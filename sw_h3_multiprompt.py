"""SW H3 多段提示词一体机（单节点版）

================================================================================
★ 这个节点替代了什么
================================================================================
原来那套 22 号工作流，为了让 4 段不同提示词走同一个循环，需要：

    4× MiniMaxH3ReferenceToVideo   （4 段条件编译）
    4× SW_H3FreeTextEncoder        （卸载 CLIP）
    4× SW_H3ConditionStrength
    1× CreateList                  （打包 4 条 CONDITIONING）
    1× StartLoop / EndLoop         （分发 + 累积）
    1× SW_H3_SegPlan_NCG               （算帧数/token 数）
    1× SW_H3_SegBridge_NCG             （循环内 latent 续接）
    1× KSampler                    （循环内采样）
    1× SW_H3_SegConcat_NCG             （latent 拼接）
    ────────────────────────────────────────────
    共 15 个节点 + 官方 4 个 H3 条件节点

本节点把它们全部收进 execute() 内部：

    SW_H3MultiPrompt_NCG
      ├─ 5 个提示词输入口（prompt_1 .. prompt_5）
      ├─ 参考图 Autogrow（person_a / person_b / background 任意张数）
      ├─ 内部：算帧数 → 编码 N 段条件 → 卸载 CLIP → 逐段采样+续接
      │        → latent 拼接 → 一次解码
      └─ 输出：video(IMAGE) / latent(LATENT) / 帧数 / 秒数 / 报告

================================================================================
★ 「自动几段」是怎么算的
================================================================================
ComfyUI 的节点输入在**图加载时就固定**，节点内部拿不到「哪个口被连了线」。
所以判定规则是「**哪个口填了非空文本**」：

    prompt_1 非空 → 1 段
    prompt_1..3 非空 → 3 段
    prompt_1..5 非空 → 5 段
    遇到第一个空口就停（后面填了也不算，报警告）

段数来源优先级：
    segment_count = "auto"  → 按上面规则自动数（默认）
    segment_count = "1".."5" → 手动锁定，忽略空口

★ 最低 3 段：auto 数出 < 3 段时**直接报错**（不是悄悄补段）。
  本工作流面向 15 秒以上成片，2 段（14.4s）达不到；确实只想试效果时
  把 segment_count 手动设成 "1" 或 "2" 即可绕过。

================================================================================
★ 三个和原工作流不一样的、更好的地方
================================================================================
1. **参考图只编码一次**
   原工作流每个 H3 节点都自己 `vae.encode(参考图)`，4 段 = 4 次重复编码。
   这里 ref_blocks 算一次，N 段共用（model_base.py:2188 的 payload 只读）。

2. **CLIP 只卸载一次**
   原工作流 4 个 SW_H3FreeTextEncoder；这里是「N 段全编完 → 卸一次 → 开采」。

3. **段间续接在 latent 域，且不经过 VAE**
   取上段尾部 2 个 token（=5 帧）注入本段 frame_idx=0 的 minimax_keyframes
   （model_base.py:2188 `payload["keyframes"]`），与官方 MiniMaxH3AddGuide
   等价但吃 latent 不吃像素。

================================================================================
★ 帧网格（逐条来自官方源码）
================================================================================
    comfy/ldm/minimax/model.py:30   FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    comfy_extras/nodes_minimax_h3.py:37   align_frame_count: 合法帧数 = 5 + 17n
    comfy_extras/nodes_minimax_h3.py:43   video_latent_t = ((f-5)//17)*5 + 2

    7 秒 → 175 帧（7.29s），52 token。段间锚定 5 帧（2 token）。
    N 段总帧数 = 175 + 170×(N-1) = 170N + 5，仍严格落在 5+17n 上。

================================================================================
★ 音轨（H3 是 audio+video 联合生成）
================================================================================
H3 的 DiT 同时去噪 video 与 stereo audio 两条 latent 流（model.py 有 audio_out，
采样序列末两段永远是 target audio + target video）。提供 ref_audio 后，模型会在
生成的 audio latent 里复现对白语音。本节点：
  · video 流 → video_vae.decode() → 画面（IMAGE）
  · audio 流 → audio_vae.decode() → 波形（AUDIO），随 video 一起送 CreateVideo
  · 因此成片是带声音的，无需后期对轨（除非你想换声 / 加 BGM）。
★ 必须接 audio_vae 才会出声音（语音由 H3 从提示词自动生成，无需 ref_audios）；
  不接 audio_vae 则 audio 输出为 None，成片静音。提供 ref_audios 时模型会在 audio latent
  里复现参考语音。
"""

from __future__ import annotations

import logging
import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 官方常数（与 comfy_extras/nodes_minimax_h3.py 逐字一致）
# --------------------------------------------------------------------------- #
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
H3_FPS = 24
FRAME_BASE = 5
FRAME_STEP = 17

# 可选段数上限
MAX_SEGMENTS = 5


# ===========================================================================
# 小工具
# ===========================================================================
def _align_frame_count(n: int) -> int:
    """向上吸附到 5+17n（与官方 align_frame_count 一致）。"""
    n = int(n)
    if n < FRAME_BASE:
        return FRAME_BASE
    while n % FRAME_STEP != FRAME_BASE:
        n += 1
    return n


def _align_frame_count_floor(n: int) -> int:
    """向下吸附到 5+17n（不小于 FRAME_BASE）。

    与 _align_frame_count 的「向上吸附」对称，用于「末段补足剩余时长」：
    我们要的是**不超过**目标的最大合法帧数，避免末段超发再砍尾。
    """
    n = int(n)
    if n <= FRAME_BASE:
        return FRAME_BASE
    k = (n - FRAME_BASE) // FRAME_STEP
    return FRAME_BASE + k * FRAME_STEP


def _video_latent_t(frame_count: int) -> int:
    """帧数 -> video latent token 数（与官方 video_latent_t 一致）。"""
    fc = int(frame_count)
    return 2 if fc <= FRAME_BASE else ((fc - FRAME_BASE) // FRAME_STEP) * 5 + 2


def _frames_of_tokens(n_tokens: int) -> int:
    """latent token 数 -> 像素帧数。"""
    return int(sum(FRAME_PER_TOKEN[k % 5] for k in range(int(n_tokens))))


def _anchor_token_count(anchor_frames: int) -> int:
    """段首锚定 frames 帧需要几个 latent token。"""
    frames = int(anchor_frames)
    if frames <= 1:
        return 1
    total = 0
    for k in range(len(FRAME_PER_TOKEN)):
        total += FRAME_PER_TOKEN[k % 5]
        if total >= frames:
            return k + 1
    return 2


def _as_int(value, default=0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, str) and value.strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _as_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "":
            return bool(default)
        if s in ("true", "1", "yes", "y", "on", "是", "开"):
            return True
        if s in ("false", "0", "no", "n", "off", "否", "关"):
            return False
        return bool(default)
    return bool(value)


def _resize(image, width, height, crop):
    """与官方 _resize 逐字一致：image [B,H,W,C] -> [B, height, width, 3]。"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _adapt_canvas(width, height):
    """与官方 adapt_canvas 逐字一致（参考视频尺寸适配）。"""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _encode_ref_audio(audio_vae, audio):
    """与官方 _encode_ref_audio 逐字一致。"""
    import torchaudio
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _build_refs(vae, audio_vae, ref_image_size, width, height,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
    """统一编码参考图/视频/音频，只跑一次，全段共用。

    返回 (ref_items, ref_blocks, counts_dict)。
    ref_items 给 tokenizer 做 <Picture>/<Video>/<Audio> 标签；
    ref_blocks 给 DiT payload 注入 latent。
    """
    ref_items, ref_blocks = [], []

    # ---- 参考图 ----
    img_count = 0
    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        ref_items.append({"type": "image", "data": resized})
        if vae is not None:
            z = vae.encode(resized)
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": z})
        img_count += 1

    # ---- 参考视频 + 配套音轨 ----
    video_count = 0
    video_audio_count = 0
    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        idx = name.rsplit("_", 1)[-1]
        soundtrack = ref_video_audios.get("ref_video_audio_" + idx)
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = _adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 参考视频至少需要 5 帧（约 0.2 秒 @24fps）")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        if soundtrack is not None:
            ref_items.append({"type": "audio"})
            video_audio_count += 1
        sample_idx = list(range(0, frames.shape[0], H3_FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        if vae is not None:
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                if audio_vae is None:
                    raise ValueError("连接了 ref_video_audio 但没有 audio_vae，无法编码参考音频")
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})
        video_count += 1

    # ---- 独立参考音频 ----
    audio_count = 0
    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        ref_items.append({"type": "audio"})
        if audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
        audio_count += 1

    counts = {"image": img_count, "video": video_count,
              "video_audio": video_audio_count, "audio": audio_count}
    return ref_items, ref_blocks, counts


def _empty_av_latent(width, height, length, batch_size=1):
    """与官方 _empty_av_latent 逐字一致。"""
    frame_count = _align_frame_count(max(5, length))
    latent_t = _video_latent_t(frame_count)
    audio_t = round(frame_count / H3_FPS * 40)     # AUDIO_LATENT_FPS = 40
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _av_parts(latent):
    """AV latent -> (video, audio)；非嵌套时返回 (tensor, None)。"""
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        return parts[0], (parts[1] if len(parts) > 1 else None)
    return samples, None


# 注：原版会构造 _zero_out(cond) 作负样本再走 CFG 路径。
# 本 NCG 版已彻底删除该路径（negative=None + cfg 固定 1.0），故不再保留此函数。


def _free_text_encoder(clip):
    """卸载 CLIP 编码器（等价 SW_H3FreeTextEncoder.free）。

    ★ 不能用 free_memory(target)：get_free_memory 已把可驱逐权重算作空闲，
      目标立刻达成 → 什么都不卸。必须 model_unload() 模型对象本身。
    """
    unloaded = 0
    try:
        want = getattr(clip, "patcher", None)
        for lm in list(getattr(comfy.model_management, "current_loaded_models", [])):
            if want is not None and getattr(lm, "model", None) is not want:
                continue
            try:
                lm.model_unload()
                if lm in comfy.model_management.current_loaded_models:
                    comfy.model_management.current_loaded_models.remove(lm)
                unloaded += 1
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover
        logger.warning("[SW-H3Multi] targeted unload failed: %s", exc)

    if unloaded == 0:
        try:
            comfy.model_management.unload_all_models()
            unloaded = -1
        except Exception as exc:  # pragma: no cover
            logger.warning("[SW-H3Multi] unload_all_models failed: %s", exc)

    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    return unloaded


def _patch_sigma_shift(model, shift_video, shift_audio):
    """等价官方 MiniMaxH3SigmaShift（nodes_minimax_h3.py:366）。"""
    import comfy.model_sampling

    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV,
                                 comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(m.model.model_config)
    model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", model_sampling)

    to = m.model_options["transformer_options"] = \
        m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    return m


# ===========================================================================
# 主节点
# ===========================================================================
class SW_H3MultiPrompt_NCG(io.ComfyNode):
    """海螺 H3 多段提示词一体机：一个节点出 15~35 秒长视频。

    填几个提示词口就跑几段（自动），最低 3 段。
    节点内部完成：条件编译 → CLIP 卸载 → 逐段采样 + latent 续接 →
    latent 拼接 → 一次 VAE 解码。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SW_H3MultiPrompt_NCG",
            display_name="SW H3 多段提示词一体机",
            category="SW/H3Segments",
            description=(
                "一个节点搞定 15 秒以上长视频。5 个提示词口，填几个跑几段（最低 3 段）。"
                "内部：算帧数 → 逐段编译条件（参考图只编码一次）→ 卸载 CLIP 省显存 → "
                "逐段采样并在 latent 域续接 → 拼接 → 一次性 VAE 解码。"
                "输出 video 可直接接 CreateVideo/SaveVideo。"
                "★ 成片自带声音：H3 是 audio+video 联合生成，接上 audio_vae 后语音由模型从提示词自动产出，"
                "无需后期对轨（除非想换声 / 加 BGM）。"
            ),
            inputs=[
                # ---- 模型 ----
                io.Model.Input("model", tooltip="H3 权重（UNETLoader 或 LoRA 之后的 model）"),
                io.Clip.Input("clip", tooltip="Qwen3-VL 文本编码器。节点编完条件后会自动卸载它。"),
                io.Vae.Input("vae", tooltip="H3 video VAE。最后做一次性解码。"),
                io.Vae.Input("audio_vae", tooltip="H3 audio VAE。参考音频需要它来编码。"),

                # ---- 5 个外部条件口（可选） ----
                # 想像截图里那样外接 CLIPTextEncode 时连这里。连了 cond_N 就跳过
                # prompt_N 的内部编码，直接拿外部条件并注入参考资产。
                io.Conditioning.Input("cond_1", optional=True, tooltip="第 1 段外部条件。连接后优先使用，不再用 prompt_1 内部编码。"),
                io.Conditioning.Input("cond_2", optional=True, tooltip="第 2 段外部条件。"),
                io.Conditioning.Input("cond_3", optional=True, tooltip="第 3 段外部条件。"),
                io.Conditioning.Input("cond_4", optional=True, tooltip="第 4 段外部条件。"),
                io.Conditioning.Input("cond_5", optional=True, tooltip="第 5 段外部条件。"),

                # ---- 5 个提示词口 ----
                # ★ 必须是 required（不能 optional），否则 io.ComfyNode 生成的
                #   INPUT_TYPES 会把 optional widget 全部挪到 required 后面，
                #   导致 widgets_values 与前端/后端顺序不一致，全部参数错位。
                #   空字符串仍被节点视为「未填写」，不影响自动段数判定。
                #   如果对应 cond_N 已连接，prompt_N 会被忽略。
                io.String.Input(
                    "prompt_1", multiline=True, dynamic_prompts=True,
                    tooltip="第 1 段提示词。★ 必填（或与 cond_1 二选一）。相对时间码从 0 秒开始。"),
                io.String.Input(
                    "prompt_2", multiline=True, dynamic_prompts=True,
                    tooltip="第 2 段提示词。接上/填了才启用第 2 段（或与 cond_2 二选一）。"),
                io.String.Input(
                    "prompt_3", multiline=True, dynamic_prompts=True,
                    tooltip="第 3 段提示词。接上/填了才启用第 3 段（或与 cond_3 二选一）。"),
                io.String.Input(
                    "prompt_4", multiline=True, dynamic_prompts=True,
                    tooltip="第 4 段提示词。接上/填了才启用第 4 段（或与 cond_4 二选一）。"),
                io.String.Input(
                    "prompt_5", multiline=True, dynamic_prompts=True,
                    tooltip="第 5 段提示词。接上/填了才启用第 5 段（或与 cond_5 二选一）。最多 5 段。"),

                io.Combo.Input(
                    "segment_count",
                    options=["auto", "1", "2", "3", "4", "5"], default="auto",
                    tooltip=(
                        "auto = 按上面 5 个口里「填了非空文本」的数量自动决定，"
                        "遇到第一个空口停止。手动值则锁定段数、忽略空口。"
                        "★ auto 数出 < 3 段会直接报错：15 秒以上至少要 3 段。"
                        "只想试效果可手动设 1 或 2。")),

                # ---- 参考图 ----
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="参考图。提示词里用 <Picture 1> <Picture 2> ... 引用，顺序按这里的口序。"),
                        prefix="ref_image_", min=0, max=9)),

                # ---- 参考视频 ----
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="参考视频帧序列（IMAGE）。提示词里用 <Video 1> <Video 2> ... 引用。"),
                        prefix="ref_video_", min=0, max=3)),

                # ---- 参考视频配套音轨 ----
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="与 ref_video_N 同号的参考视频音轨。需要 audio_vae。"),
                        prefix="ref_video_audio_", min=0, max=3)),

                # ---- 独立参考音频 ----
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio",
                            tooltip="独立参考音频。提示词里用 <Audio 1> <Audio 2> ... 引用。需要 audio_vae。"),
                        prefix="ref_audio_", min=0, max=3)),

                # ---- 画布与时长 ----
                io.Int.Input("width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768, min=32, max=8192, step=32),
                io.Float.Input(
                    "segment_seconds", default=7.0, min=0.5, max=20.0, step=0.1,
                    tooltip=("每段目标秒数。会自动吸附到 H3 合法帧数网格 5+17n"
                             "（7 秒 → 175 帧 = 7.29 秒）。")),

                # ---- 采样 ----
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True,
                             tooltip="起始种子。每段会 +递增种子，避免段间噪声雷同。"),
                io.Boolean.Input(
                    "递增种子", default=True,
                    tooltip="第 2 段用 seed+1、第 3 段 seed+2…… 段间噪声不同，过渡更自然。"),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Combo.Input("sampler_name",
                               options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler",
                               options=comfy.samplers.KSampler.SCHEDULERS, default="beta",
                               tooltip="★ beta = H3 官方 BasicScheduler 用的调度器，务必保持。"
                                       "simple 在高噪声段跨步过大（6 步时首段噪声高 82%），"
                                       "蒸馏 LoRA 下会让latent 冲出分布 → 画面泛紫。"
                                       "只有在做对比实验时才改回 simple。"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),

                # ---- 条件细节 ----
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="match = 按成片面积等比缩小参考图（快）。"
                                       "max = 2048 短边，人物一致性最好但慢很多。"),
                io.Float.Input("visual_strength", default=0.999, min=0.0, max=1.0, step=0.001,
                               tooltip="参考图 latent 的保真度。1.0 最贴参考图，调低让模型有自由度。"),
                io.Float.Input("audio_strength", default=1.0, min=0.0, max=1.0, step=0.001,
                               tooltip="参考音频 latent 的保真度。没用参考音频时无影响。"),
                io.Combo.Input("anchor_frames", options=["5", "1"], default="5",
                               tooltip="段间锚定帧数。5 帧（推荐）保证拼接后总帧数仍落在网格上；"
                                       "1 帧衔接更紧但总帧数会偏。"),

                # ---- 采样分布 ----
                io.Boolean.Input("内置sigma_shift", default=True,
                                 tooltip="节点内部给 model 打 sigma shift（官方 MiniMaxH3SigmaShift 等价物）。"
                                         "★ 如果你在节点外面已经接了 MiniMaxH3SigmaShift，这里要关掉。"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),

                # ---- 输出控制 ----
                io.Boolean.Input("内部解码", default=True,
                                 tooltip="True = 节点内直接出 IMAGE（可接 CreateVideo）。"
                                         "False = 只出 LATENT，自己接官方 VAE Decode。"),
                io.Float.Input("裁剪到秒数", default=0.0, min=0.0, max=600.0, step=0.5,
                                tooltip=("0 = 不裁（用每段自然时长，对白最完整）。填 26 时：默认让「最后一段」"
                                         "只生成补足剩余时长的长度（末段对白完整保留），而不是先满段再砍尾；"
                                         "视频与音频同步裁剪，避免音视频错位。若段数过多前段已超目标，则回退砍尾并警告。")),
            ],
            outputs=[
                io.Image.Output(display_name="video"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="段数"),
                io.Int.Output(display_name="成片帧数"),
                io.Float.Output(display_name="成片秒数"),
                io.Int.Output(display_name="每段帧数"),
                io.String.Output(display_name="报告"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    # ---------------------------------------------------------------- #
    @classmethod
    def execute(cls, model, clip, vae, audio_vae=None,
                cond_1=None, cond_2=None, cond_3=None, cond_4=None, cond_5=None,
                prompt_1=None, prompt_2=None, prompt_3=None,
                prompt_4=None, prompt_5=None,
                segment_count="auto", ref_images=None,
                ref_videos=None, ref_video_audios=None, ref_audios=None,
                width=1344, height=768, segment_seconds=7.0,
                seed=0, 递增种子=True, steps=20,
                sampler_name="euler", scheduler="beta", denoise=1.0,
                ref_image_size="match",
                visual_strength=0.999, audio_strength=1.0,
                anchor_frames="5",
                内置sigma_shift=True, shift_video=12.0, shift_audio=3.0,
                内部解码=True, 裁剪到秒数=0.0) -> io.NodeOutput:

        # ================= 1. 段数判定 =================
        cond_slots = [cond_1, cond_2, cond_3, cond_4, cond_5]
        prompt_slots = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5]
        texts = [(str(p).strip() if p is not None else "") for p in prompt_slots]
        # 有效 = 外部 cond 已连，或 prompt 填了非空文本
        active = [(cond_slots[i] is not None) or bool(texts[i]) for i in range(MAX_SEGMENTS)]

        # auto：数「从第 1 口起连续有效」的长度。遇到空口即停 —— 段号必须连续，
        # 否则后段会拿着空条件去采样。
        auto_count, stop_reason = 0, ""
        for idx, ok in enumerate(active):
            if ok:
                auto_count += 1
            else:
                later = [j + 1 for j in range(idx + 1, MAX_SEGMENTS) if active[j]]
                if later:
                    stop_reason = ("第 %d 段既没接 cond 也没填 prompt，但第 %s 段有效 —— 段号必须连续，"
                                   "auto 模式只认前 %d 段"
                                   % (idx + 1, "/".join(str(j) for j in later), auto_count))
                break

        def _describe_filled():
            parts = []
            for i in range(MAX_SEGMENTS):
                if cond_slots[i] is not None:
                    parts.append("第%d段(cond)" % (i + 1))
                elif texts[i]:
                    parts.append("第%d段(prompt)" % (i + 1))
            return "、".join(parts) or "（全空）"

        mode = str(segment_count).strip().lower()
        if mode == "auto":
            n_seg = auto_count
            if n_seg < 3:
                raise ValueError(
                    "[SW-H3Multi] 自动数出 %d 段有效输入，但本节点面向 15 秒以上长视频，"
                    "最低需要 3 段（3 段 = 21.5 秒）。\n"
                    "  · 你填了：%s\n"
                    "%s"
                    "  · 解决办法：① 把 segment_count 手动设成 \"1\" 或 \"2\" 先试效果；\n"
                    "            ② 补齐第 %d 段 cond 或 prompt（段号要连续）。"
                    % (n_seg, _describe_filled(),
                       ("  · 注意：%s\n" % stop_reason) if stop_reason else "",
                       n_seg + 1))
        else:
            n_seg = _as_int(mode, 3)
            n_seg = max(1, min(MAX_SEGMENTS, n_seg))
            missing = [i + 1 for i in range(n_seg) if not active[i]]
            if missing:
                raise ValueError(
                    "[SW-H3Multi] segment_count 锁了 %d 段，但第 %s 段既没接 cond 也没填 prompt。\n"
                    "  · 已填：%s"
                    % (n_seg, "/".join(str(j) for j in missing), _describe_filled()))

        n_seg = max(1, min(MAX_SEGMENTS, n_seg))

        # 为每段准备：外部 cond（优先）或内部 prompt
        external_conds = [cond_slots[i] for i in range(n_seg)]
        prompts_for_internal = [
            None if cond_slots[i] is not None else texts[i]
            for i in range(n_seg)
        ]

        # ================= 2. 帧数规划 =================
        width = _as_int(width, 1344)
        height = _as_int(height, 768)
        seg_seconds = _as_float(segment_seconds, 7.0)
        anchor = _as_int(anchor_frames, 5)
        if anchor not in (1, 5):
            anchor = 5

        seg_frames = _align_frame_count(max(FRAME_BASE, int(round(seg_seconds * H3_FPS))))
        seg_tokens = _video_latent_t(seg_frames)
        anchor_tokens = _anchor_token_count(anchor)
        if anchor >= seg_frames:
            anchor, anchor_tokens = 5, _anchor_token_count(5)
        step_frames = seg_frames - anchor
        total_frames = seg_frames + step_frames * (n_seg - 1)

        # ---- 末段补足（解决「最后一段语音没说全」）---------------------------
        # 当「裁剪到秒数」小于自然总时长时，让最后一段只生成「补足剩余」的长度，
        # 而不是先生成满段再砍尾 —— 否则砍掉的恰好是末段尾部的对白语音。
        # 仅当 前 n-1 段自然总时长 <= 目标帧数 时生效（末段还能缩得下）；
        # 否则（如 5 段硬塞 26 秒）前 n-1 段已超限，只能回退「砍尾」并在报告里警告。
        crop_target = int(round(_as_float(裁剪到秒数, 0.0) * H3_FPS)) \
            if _as_float(裁剪到秒数, 0.0) > 0 else 0
        seg_frames_list = [seg_frames] * n_seg
        last_shortened = False
        tail_chop_fallback = False
        if crop_target > 0 and n_seg >= 2:
            lead_frames = seg_frames + step_frames * (n_seg - 2)   # 前 n-1 段自然总帧数
            remainder = crop_target - lead_frames
            if remainder >= FRAME_BASE:
                seg_frames_list[-1] = _align_frame_count_floor(remainder)
                last_shortened = True
            else:
                # 前 n-1 段已经超出目标，末段缩到最小也装不下 -> 只能砍尾
                tail_chop_fallback = True

        # ================= 3. model patch =================
        if _as_bool(内置sigma_shift, True):
            model = _patch_sigma_shift(model,
                                       _as_float(shift_video, 12.0),
                                       _as_float(shift_audio, 3.0))

        # ================= 4. 参考资产只编码一次 =================
        ref_items, ref_blocks, ref_counts = _build_refs(
            vae, audio_vae, ref_image_size, width, height,
            ref_images=ref_images, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios)
        print("[SW-H3Multi] 参考资产：图 %d 张 / 视频 %d 个（含音轨 %d） / 独立音频 %d 条"
              "（只编码一次，%d 段共用）"
              % (ref_counts["image"], ref_counts["video"], ref_counts["video_audio"],
                 ref_counts["audio"], n_seg), flush=True)

        # ================= 5. 逐段编译条件（CLIP 在这一步） =================
        conds = []
        for i in range(n_seg):
            ext_cond = external_conds[i]
            text = prompts_for_internal[i]
            if ext_cond is not None:
                # 深拷贝，避免修改用户传入的条件；然后注入参考资产
                cond = [[entry[0], dict(entry[1])] for entry in ext_cond]
                print("[SW-H3Multi] 段 %d/%d 使用外部 cond（跳过内部 CLIP 编码）"
                      % (i + 1, n_seg), flush=True)
            elif text:
                tokens = clip.tokenize(text, minimax_ref_items=ref_items)
                cond = clip.encode_from_tokens_scheduled(tokens)
            else:
                raise ValueError("[SW-H3Multi] 第 %d 段既没有外部 cond 也没有 prompt，无法编译条件。"
                                 % (i + 1))
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
            if _as_float(visual_strength, 0.999) < 1.0 or _as_float(audio_strength, 1.0) < 1.0:
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_visual_cond_noise_aug": _as_float(visual_strength, 0.999),
                    "minimax_audio_cond_noise_aug": _as_float(audio_strength, 1.0),
                })
            conds.append(cond)
            print("[SW-H3Multi] 段 %d/%d 条件就绪（%s）"
                  % (i + 1, n_seg,
                     "外部 cond" if ext_cond is not None else "内部编码 %d 字" % len(text)),
                  flush=True)

        del ref_items, ref_blocks

        # ================= 6. 卸载 CLIP（一次） =================
        # 采样前必然不再需要 CLIP（条件已全部编译完，且采样不再走文本编码器），
        # 所以这一段无条件执行，不受「内部解码」开关影响。
        how = _free_text_encoder(clip)
        print("[SW-H3Multi] CLIP 编码器已卸载（%s）"
              % ("全部模型" if how == -1 else "%d 个模型对象" % how), flush=True)

        # ================= 7. 逐段采样 + latent 续接 =================
        seed0 = _as_int(seed, 0)
        stride = 1 if _as_bool(递增种子, True) else 0
        steps_i = _as_int(steps, 20)
        denoise_f = _as_float(denoise, 1.0)
        total_steps = max(1, steps_i * n_seg)
        bar = comfy.utils.ProgressBar(total_steps)
        done = 0

        pieces, prev_video = [], None

        for i, cond in enumerate(conds):
            seg_seed = (seed0 + stride * i) % 0x10000000000000000

            # 段 i>0：把上段尾部 anchor 个 token 注入本段 frame_idx=0
            if prev_video is not None and prev_video.shape[2] > anchor_tokens:
                tail = prev_video[:, :, -anchor_tokens:, :, :].clone()
                kf = list(cond[0][1].get("minimax_keyframes", []))
                kf.append({"resolved_frame_index": 0, "latent": tail})
                cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": kf})

            # ================= 无 CFG 路径（NCG 版核心） =================
            # 官方 BasicGuider 只接 model + conditioning，结构上不可能有 negative，
            # 其 cfg 恒为 1.0（comfy_extras/nodes_custom_sampler.py 的 Guider_Basic）。
            # 这里照官方做法：negative 传 None + cfg 固定 1.0。
            # → samplers.py:610 `math.isclose(cond_scale, 1.0)` 命中，
            #   uncond_ 直接置None，calc_cond_batch 过滤掉它，
            #   **每步只前向 cond 一次，等于省一半算力**，且不做任何信号缩放。
            negative = None

            latent, frames_this = _empty_av_latent(width, height, seg_frames_list[i])
            noise = comfy.sample.prepare_noise(latent["samples"], seg_seed)

            def _cb(p, x0, x, total, noisy_samples=None, _done=done):
                bar.update_absolute(_done + p + 1)
                comfy.model_management.throw_exception_if_processing_interrupted()
                return None

            sampled = comfy.sample.sample(
                model, noise, steps_i, 1.0, sampler_name, scheduler,
                cond, negative, latent["samples"],
                denoise=denoise_f, disable_noise=False,
                callback=_cb, seed=seg_seed)

            video, audio = _av_parts({"samples": sampled})
            pieces.append((video, audio))
            prev_video = video
            done += steps_i

            print("[SW-H3Multi] 段 %d/%d 采样完成：%d 帧 / %d token（seed=%d）"
                  % (i + 1, n_seg, frames_this, int(video.shape[2]), seg_seed), flush=True)

        # ================= 8. latent 拼接 =================
        videos, audios, per_seg = [], [], []
        for index, (video, audio) in enumerate(pieces):
            keep_from = 0 if index == 0 else min(anchor_tokens, max(0, video.shape[2] - 1))
            piece = video[:, :, keep_from:, :, :]
            videos.append(piece)
            per_seg.append(int(piece.shape[2]))
            if audio is not None:
                take = 0 if index == 0 else min(int(round(anchor * FRAME_RESCALE)),
                                                max(0, audio.shape[-1] - 1))
                audios.append(audio[..., take:])

        merged_video = torch.cat(videos, dim=2) if len(videos) > 1 else videos[0]
        total_tokens = int(merged_video.shape[2])
        real_frames = _frames_of_tokens(total_tokens)

        merged_audio = None
        if audios:
            merged_audio = torch.cat(audios, dim=-1) if len(audios) > 1 else audios[0]
            target = int(round(real_frames * FRAME_RESCALE))
            if merged_audio.shape[-1] > target:
                merged_audio = merged_audio[..., :target]
            elif merged_audio.shape[-1] < target:
                pad = target - merged_audio.shape[-1]
                merged_audio = torch.cat(
                    [merged_audio, merged_audio[..., -1:].repeat(1, 1, 1, pad)], dim=-1)

        merged = {"samples": comfy.nested_tensor.NestedTensor((merged_video, merged_audio))
                  if merged_audio is not None else merged_video}
        merged.pop("noise_mask", None)

        # ================= 9. 一次解码 =================
        # VAE 时间分块参数保持官方默认（token_drop=3，混合区 5 帧）。
        # 实测加大混合区（token_drop=1）会引入画面跳动，得不偿失。
        # ⚠️ 画面泛紫的真正根因是 scheduler：simple 在高噪声段跨步过大，
        #    蒸馏 LoRA 下 latent 冲出分布。改用官方 beta 即可根治，
        #    与本处的解码分块无关（详见 README「画面泛紫」章节）。
        video_batch = None
        audio_out = None
        out_frames = real_frames
        out_seconds = real_frames / H3_FPS
        do_decode = _as_bool(内部解码, True)
        cropped = False

        if do_decode:
            video_only = merged_video
            print("[SW-H3Multi] 一次性解码 %d token / %d 帧（VAE 自带时间分块，边界无缝）"
                  % (total_tokens, real_frames), flush=True)
            images = vae.decode(video_only)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            video_batch = images
            out_frames = int(images.shape[0])
            out_seconds = out_frames / H3_FPS

            # ---- 音频流解码：H3 的 audio latent 必须靠 audio_vae 解出波形 ----
            if merged_audio is not None and audio_vae is not None:
                wav = audio_vae.decode(merged_audio)
                wav = wav.movedim(-1, 1)   # [B,2,L] -> [B,L,2]（ComfyUI AUDIO 格式）
                audio_out = {"waveform": wav, "sample_rate": 32000}
                print("[SW-H3Multi] 音频流已解码：%d 采样 @32kHz，将随 video 一起送 CreateVideo"
                      % int(wav.shape[1]), flush=True)
            elif merged_audio is not None:
                print("[SW-H3Multi] 注意：未连接 audio_vae，audio 流不解码（成片为静音视频）。"
                      "想出声音请把 H3 audio VAE 接到 audio_vae。", flush=True)

            # ---- 音频无条件同步到实际视频帧数 --------------------------------
            # 若 VAE 实际输出帧数与预计算不一致（任何来源的差异），音频严格
            # 切到实际视频时长，杜绝 CreateVideo 合成时截掉尾部音频
            # （「末段语音说不全」）。官方默认参数下两者应一致，此处不触发。
            if audio_out is not None:
                need = int(round(out_frames / H3_FPS * audio_out["sample_rate"]))
                w = audio_out["waveform"]
                if w.shape[1] > need:
                    audio_out = {"waveform": w[:, :need, :],
                                 "sample_rate": audio_out["sample_rate"]}

            # ---- 同步裁剪：视频与音频一起裁到「裁剪到秒数」-------------------
            # 关键修复：之前只切 video_batch、不切 audio_out，导致音频比视频长，
            # CreateVideo 合成时把超出视频长度的尾部音频（末段对白）强行截掉，
            # 表现为「最后一段语音没说全」。现在音频随视频同步裁，杜绝错位。
            if crop_target > 0 and crop_target < out_frames:
                want = crop_target
                video_batch = video_batch[:want]
                out_frames = want
                out_seconds = out_frames / H3_FPS
                if audio_out is not None:
                    need = int(round(out_frames / H3_FPS * audio_out["sample_rate"]))
                    w = audio_out["waveform"]
                    if w.shape[1] > need:
                        audio_out = {"waveform": w[:, :need, :],
                                     "sample_rate": audio_out["sample_rate"]}
                cropped = True

        # ================= 10. 报告 =================
        grid_ok = (total_tokens - 2) % 5 == 0
        lines = [
            "================ SW H3 多段一体机 ================",
            "段数    : %d 段（来源：%s）%s"
            % (n_seg, "auto 自动判定" if mode == "auto" else "手动锁定",
               ("  ⚠️ " + stop_reason) if stop_reason else ""),
            "每段    : %d 帧 / %.4f 秒（%.2fs 请求 → 吸附到 5+17n 网格）"
            % (seg_frames, seg_frames / H3_FPS, seg_seconds),
            "锚定    : %d 帧 = %d token（第 2 段起开头复刻上段末帧，拼接时丢弃）"
            % (anchor, anchor_tokens),
            "末段    : %s"
            % ("补足到 %d 帧（目标 %d 秒；末段对白完整保留，不再砍尾）"
               % (seg_frames_list[-1], round(crop_target / H3_FPS))
               if last_shortened else
               ("⚠️ 砍尾回退：前 %d 段已超目标，末段尾部被裁（对白可能不完整），建议减段数或放宽时长"
                % (n_seg - 1) if tail_chop_fallback else "标准 %d 帧" % seg_frames)),
            "每段保留: " + ", ".join("段%d %d token" % (i + 1, n) for i, n in enumerate(per_seg)),
            "合并后  : %d token → %d 帧 / %.4f 秒" % (total_tokens, real_frames, real_frames / H3_FPS),
            "网格校验: %s" % ("✅ 5c+2，与训练网格一致" if grid_ok else "⚠️ 偏离 5c+2"),
            "参考资产: 图 %d 张 / 视频 %d 个（含音轨 %d） / 独立音频 %d 条（只编码 1 次，%d 段共用）"
            % (ref_counts["image"], ref_counts["video"], ref_counts["video_audio"],
               ref_counts["audio"], n_seg),
            "采样    : %d 步 × %d 段，%s / %s，★无 CFG（cond 单路，负样本跳过），seed=%d%s"
            % (steps_i, n_seg, sampler_name, scheduler, seed0,
               "（逐段 +1）" if stride else "（全段同 seed）"),
            "显存    : CLIP 编码器已在采样前卸载（%s）"
            % ("已卸" if how != 0 else "未能定位，已尝试全卸"),
        ]
        if do_decode:
            lines.append("输出    : IMAGE %d 帧 / %.4f 秒%s"
                         % (out_frames, out_seconds,
                            "（音视频同步裁剪）" if cropped else ""))
        else:
            lines.append("输出    : LATENT（内部解码已关闭，请接官方 VAE Decode）")
        lines += [
            "★ 音频流：%s"
            % ("已解码并随 video 输出（CreateVideo 合成带声视频；声音由 H3 从提示词自动生成，"
               "若接了 ref_audio 则复现参考语音）"
               if audio_out is not None else
               "未解码（未接 audio_vae → 静音视频；接上 H3 audio VAE 即可出声）"),
            "===============================================",
        ]
        report = "\n".join(lines)
        print("[SW-H3Multi]\n" + report, flush=True)

        return io.NodeOutput(
            video_batch, merged, n_seg, out_frames,
            round(out_seconds, 6), seg_frames, report, audio_out,
        )


NODE_CLASS_MAPPINGS = {
    "SW_H3MultiPrompt_NCG": SW_H3MultiPrompt_NCG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SW_H3MultiPrompt_NCG": "SW H3 多段一体机【无CFG·NCG】",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
