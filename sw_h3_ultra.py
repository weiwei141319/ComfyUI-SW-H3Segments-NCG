"""SW 海螺 H3 长视频「超级进化版」单节点

================================================================================
★ 一句话
================================================================================
把「模型输入 / 素材输入 / 创建视频」之外的**所有东西**收进一个节点：
帧数规划、参考资产编码、CLIP 文本/视觉编码、逐段采样、latent 续接、
参考接力回灌、外部歌音轨切片对齐、一次性解码、音视频同步。

================================================================================
★ 和老节点 SW_H3MultiPrompt_NCG 的区别
================================================================================
┌────────────────┬──────────────────────┬───────────────────────────────┐
│                │ SW_H3MultiPrompt_NCG     │ SW_H3Ultra_NCG（本节点）           │
├────────────────┼──────────────────────┼───────────────────────────────┤
│ 提示词口        │ 固定 5 个 widget      │ ★ Autogrow 口，1~12 段随接随长 │
│ 段数上限        │ 5 段（~35 秒）        │ ★ 12 段（~85 秒）              │
│ 外部整首歌驱动  │ ✗                     │ ★ 逐段切片注入 + 成片音轨替换  │
│ 参考图 VAE 编码 │ 只算 1 次全段共用     │ ★ 一次 / 每段重编 / 参考接力   │
│ 参考接力回灌    │ ✗                     │ ★ 上段画面回灌为新参考图       │
│ 锚定帧数        │ 固定 5 帧             │ ★ 5 / 22 帧可选                │
│ 音轨策略        │ 只能 H3 自產          │ ★ H3 / 外部歌曲 / 混合         │
└────────────────┴──────────────────────┴───────────────────────────────┘

================================================================================
★ 关键答疑：参考图「多次编码」到底有没有用？
================================================================================
参考图在本里走**两条完全独立的路**，必须先分清是哪一条：

路线 A：CLIP 视觉塔（Qwen3-VL）
    `clip.tokenize(prompt, minimax_ref_items=[...])` → 让文本侧"看清是谁"。
    → 本节点**本来就每段都跑 N 次**（每段 prompt 不同、LD 也不同）。
      这从来不是被优化掉的部分。

路线 B：VAE latent（ref_blocks 里那条 latent）
    进 DiT 打包序列当 ref_img token（`comfy/ldm/minimax/model.py:395` 起）。
    → 老节点把它只算一次，N 段共用同一份 tensor。

**重复算路线 B 有没有用？**
    没有。两个理由，都能在官方源码里查证：

    1) `vae.encode()` 是确定性的 → 重跑 4 次得到的是**逐位相同**的 tensor。
    2) 参考 token 的 RoPE 位置与「第几段」无关：
         comfy/ldm/minimax/model.py:353
             cursor = float(text_len)
             for blk in refs: cursor += _ref_t_span(blk)
             ...
             pos.append(_video_grid(latent_t, frame, cursor))   # target 起点
       参考 token 的位置只由 **text_len + refs 的 span** 决定，
       target 长度、段号都不参与 → 复用与重编在模型眼里完全等价。

    ⚠️ 所以「多次编码」本身不改善画质。本节点仍提供 `每段重编` 选项，
       但报告里会明写「结果与一次编码逐位相同，仅供你自己验证」。

**那后面几段画质真的劣化怎么办？→ 用「参考接力」**
    真正丢的不是「参考图被看过几次」，而是**后续段看不到上一段实际生成的样子**：
    段间只靠 5 帧（2 个 latent token）的首帧锚定拉住，而这一段还有各自不同的
    提示词在牵引画面，所以第 3、4 段的细节/肤色会顺着各自的去噪轨迹漂开。

    `参考接力`把上一段末端画面**直接变成下一段的新 ref_img token**：

        video[:, :, -1:, :, :]   shape [B,24,1,H/16,W/16]
        → {"kind":"image", "latent_h":H//16, "latent_w":W//16, "latent": 切片}

    这在 PackedLayout 里就是标准的 image ref（单帧一个 spatial group），
    而且是**纯 latent 操作：不跑 VAE、不跑 CLIP、几乎零成本**。

    三种接力模式：
      off          不动
      latent       纯 latent 切片回灌（推荐，免费）
      full         额外把自己 VAE 解码回像素，再让 Qwen 视觉塔也看到
                   （最强，但 CLIP 必须常驻，多了 16GB 左右的显存）

================================================================================
★ 帧网格（逐条来自官方源码）
================================================================================
    comfy/ldm/minimax/model.py:31       FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    comfy_extras/nodes_minimax_h3.py:37 align_frame_count: 合法帧数 = 5 + 17n
    comfy_extras/nodes_minimax_h3.py:43 video_latent_t = ((f-5)//17)*5 + 2
    音频 latent 帧率 40Hz → audio_t = round(frames / 24 * 40)，FRAME_RESCALE = 5/3

================================================================================
★ 安全约束（开发期）
================================================================================
本文件只做文件编辑 + 离线静态校验（脚本在 output/ 目录），
**绝不发出 /api/prompt、绝不真跑采样**。实际效果由用户在显卡上验证。
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

SW_H3_ULTRA_VERSION = "v1.0-20261004"

# Autogrow 提示词口数量上限
MAX_PROMPTS = 12
PROMPT_NAMES = ["prompt_%d" % i for i in range(1, MAX_PROMPTS + 1)]

AUDIO_SAMPLE_RATE = 32000   # H3 audio VAE 工作采样率

# --------------------------------------------------------------------------- #
# ★ 自包含工具层：一份都不从别的 SW 模块 import。
#
#   原因：本包里 sw_h3_multiprompt.py 正被别的会话同时编辑（2026-10-04 07:44
#   那次改动就把 _smooth_temporal_chunks 删掉了）。超级进化版必须做到
#   「别人怎么改都不影响我」，所以下面全部按官方源码逐字重写一份。
#   权威出处：comfy_extras/nodes_minimax_h3.py + comfy/ldm/minimax/model.py
# --------------------------------------------------------------------------- #
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
H3_FPS = 24
FRAME_BASE = 5
FRAME_STEP = 17
AUDIO_LATENT_FPS = 40


def _align_frame_count(n: int) -> int:
    """向上吸附到 5+17n（官方 align_frame_count）。"""
    n = int(n)
    if n < FRAME_BASE:
        return FRAME_BASE
    while n % FRAME_STEP != FRAME_BASE:
        n += 1
    return n


def _align_frame_count_floor(n: int) -> int:
    """向下吸附到 5+17n（用于「末段补足剩余时长」，不能超发）。"""
    n = int(n)
    if n <= FRAME_BASE:
        return FRAME_BASE
    return FRAME_BASE + ((n - FRAME_BASE) // FRAME_STEP) * FRAME_STEP


def _video_latent_t(frame_count: int) -> int:
    """帧数 -> video latent token 数（官方 video_latent_t）。"""
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
    """官方 _resize：[B,H,W,C] -> [B,height,width,3]。"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _adapt_canvas(width, height):
    """官方 adapt_canvas（参考视频画布适配）。"""
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
    """官方 _encode_ref_audio：-> ([1,32,2,T], T)。"""
    import torchaudio
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _build_refs(vae, audio_vae, ref_image_size, width, height,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
    """统一编码参考图/视频/音频。

    ★ 注意 key 顺序：Autogrow 传进来是 dict，'ref_image_10' 的字典序会排在
      'ref_image_2' 前面，所以这里显式按数字后缀排序，保证 <Picture N> 编号稳定。

    返回 (ref_items, ref_blocks, counts)。
      ref_items  → 给 tokenizer 生成 <Picture>/<Video>/<Audio> 标签
      ref_blocks → 给 DiT payload 注入 latent
    """
    def _num(key):
        tail = str(key).rsplit("_", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return 1 << 30

    ref_items, ref_blocks = [], []
    img_count = video_count = video_audio_count = audio_count = 0

    for name, img in sorted((ref_images or {}).items(), key=lambda kv: _num(kv[0])):
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
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": vae.encode(resized)})
        img_count += 1

    ref_video_audios = ref_video_audios or {}
    for name, frames_in in sorted((ref_videos or {}).items(), key=lambda kv: _num(kv[0])):
        if frames_in is None:
            continue
        idx = name.rsplit("_", 1)[-1]
        soundtrack = ref_video_audios.get("ref_video_audio_" + idx)
        vh, vw = frames_in.shape[1], frames_in.shape[2]
        cw, ch = _adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(frames_in, cw, ch, "disabled")
        n = frames.shape[0]
        if n < FRAME_BASE:
            raise ValueError("MiniMax H3 参考视频至少需要 5 帧（约 0.2 秒 @24fps）")
        while n % FRAME_STEP != FRAME_BASE:
            n -= 1
        frames = frames[:n]
        if soundtrack is not None:
            ref_items.append({"type": "audio"})
            video_audio_count += 1
        sample_idx = list(range(0, frames.shape[0], H3_FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_idx],
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

    for name, audio in sorted((ref_audios or {}).items(), key=lambda kv: _num(kv[0])):
        if audio is None:
            continue
        ref_items.append({"type": "audio"})
        if audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})
        audio_count += 1

    return ref_items, ref_blocks, {"image": img_count, "video": video_count,
                                   "video_audio": video_audio_count, "audio": audio_count}


def _empty_av_latent(width, height, length, batch_size=1):
    """官方 _empty_av_latent：视频 24ch / 音频 32ch 的 AV latent。"""
    frame_count = _align_frame_count(max(FRAME_BASE, length))
    latent_t = _video_latent_t(frame_count)
    audio_t = round(frame_count / H3_FPS * AUDIO_LATENT_FPS)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _av_parts(latent):
    """AV latent -> (video, audio)。"""
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        return parts[0], (parts[1] if len(parts) > 1 else None)
    return samples, None


# 注：原版会构造 _zero_out(cond) 作负样本再走 CFG 路径。
# 本 NCG 版已彻底删除该路径（negative=None + cfg 固定 1.0），故不再保留此函数。


def _free_text_encoder(clip):
    """卸载 CLIP 编码器。

    ★ 不能用 free_memory(target)：get_free_memory 会把可驱逐权重算作空闲，
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
        logger.warning("[SW-H3Ultra] targeted unload failed: %s", exc)

    if unloaded == 0:
        try:
            comfy.model_management.unload_all_models()
            unloaded = -1
        except Exception as exc:  # pragma: no cover
            logger.warning("[SW-H3Ultra] unload_all_models failed: %s", exc)

    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    return unloaded


def _patch_sigma_shift(model, shift_video, shift_audio):
    """等价官方 MiniMaxH3SigmaShift。"""
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
# 工具
# ===========================================================================
def _text_of(value) -> str:
    """把 4 种可能的入参（None / '' / primitive * 包装 / str）统一成干净字符串。

    ┴ 重要：ComfyUI 前端把「已转成 input 但未连线」的 widget 序列化成空串 ''，
      不能直接 bool 判断，也不能直接当文本用。
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        # primitive 节点包装出来的 [value, *]
        value = value[0] if value else ""
    s = str(value)
    return s.strip()


def _sorted_prompts(prompts) -> list[tuple[int, str]]:
    """把 Autogrow 传来的 dict 按**数字后缀**排好序。

    ⚠️ 不能直接 dict.values()：'prompt_10' 的字典序会排在 'prompt_2' 前面，
       12 段时段的顺序会全乱。
    """
    out = []
    for key, value in (prompts or {}).items():
        name = str(key)
        tail = name.rsplit("_", 1)[-1] if "_" in name else name
        try:
            idx = int(tail)
        except ValueError:
            continue
        out.append((idx, _text_of(value)))
    out.sort(key=lambda pair: pair[0])
    return out


def _slice_song(song, start_sec: float, dur_sec: float):
    """从整条歌里切出一段。返回 {"waveform": [1,C,L], "sample_rate": sr}。

    尾部不够长就补静音（`AudioConcat` 重复拼会绕回歌开头，不能那么干）。
    """
    if song is None:
        return None
    waveform = song["waveform"]
    sr = int(song["sample_rate"])
    if waveform.ndim == 3:
        channels = waveform.shape[1]
        total = waveform.shape[2]
    else:  # pragma: no cover - 防御
        return None
    start = max(0, int(round(start_sec * sr)))
    need = max(1, int(round(dur_sec * sr)))
    usable = max(0, total - start)
    take = min(need, usable)
    piece = waveform[:, :, start:start + take]
    if take < need:
        pad = torch.zeros((1, channels, need - take), dtype=piece.dtype, device=piece.device)
        piece = torch.cat([piece, pad], dim=2)
    return {"waveform": piece, "sample_rate": sr}


def _resample_audio(audio, target_sr: int):
    """重采样到目标采样率（torchaudio 在 ComfyUI 里是标配）。"""
    if audio is None:
        return None
    sr = int(audio["sample_rate"])
    if sr == target_sr:
        return audio
    import torchaudio
    wav = torchaudio.functional.resample(audio["waveform"], sr, target_sr)
    return {"waveform": wav, "sample_rate": target_sr}


def _relay_pixel(vae, tail_tokens):
    """把上段末端 latent 解回像素，给 Qwen 视觉塔用（仅接力=full 时走）。

    入参是上段末端的 latent token 切片 [B,C,T,H/16,W/16]；
    单 token 过 H3 VAE 的时间卷积缺上下文，取最后 2 个 token 更稳。
    失败返回 None，调用方自动降级为纯 latent 接力。
    """
    # ★ 不要用未定义的 video_latent（原版这里的遮蔽行是 bug，会直接 NameError）。
    #   入参本身已经是 relay_history 里切好的末端片段，再从尾部取 2 个 token 即可。
    if tail_tokens.shape[2] > 2:
        tail_tokens = tail_tokens[:, :, -2:, :, :]
    try:
        pix = vae.decode(tail_tokens)
    except Exception as exc:  # pragma: no cover - 依赖真实权重
        print("[SW-H3Ultra] 参考接力：末端 latent 解码失败（%s），自动降级为纯 latent 接力" % exc,
              flush=True)
        return None
    if pix.ndim == 5:
        pix = pix.reshape(-1, pix.shape[-3], pix.shape[-2], pix.shape[-1])
    return pix[-1:]          # 取最后一帧，[1,H,W,C]


def _relay_block(past_tail, height: int, width: int):
    """构造 image ref 块。

    ⚠️ PackedLayout 的 image 分支以单帧为一个 spatial group：
         n = _frame_grid(latent_h, latent_w).shape[0]
       行的个数由 spatial 决定，所以 latent 的 **latent_t 必须是 1**，
       否则 rows 数量对不上 layout。这里显式切成最后一帧。
    """
    frame = past_tail[:, :, -1:, :, :]
    return {"kind": "image", "latent_h": height // 16, "latent_w": width // 16,
            "latent": frame}


def _mix_audio(primary, secondary, ratio_secondary: float):
    """把两条音轨按权重混合到 primary 的采样率上。"""
    if primary is None:
        return secondary
    if secondary is None or ratio_secondary <= 0.0:
        return primary
    sr = int(primary["sample_rate"])
    other = _resample_audio(secondary, sr)
    a = primary["waveform"]
    b = other["waveform"]
    if a.ndim != 3:
        a = a.unsqueeze(0)
    if b.ndim != 3:
        b = b.unsqueeze(0)
    length = min(a.shape[2], b.shape[2])
    a = a[:, :, :length]
    b = b[:, :, :length]
    channels = min(a.shape[1], b.shape[1])
    a = a[:, :channels, :]
    b = b[:, :channels, :]
    mixed = (1.0 - ratio_secondary) * a + ratio_secondary * b
    return {"waveform": mixed, "sample_rate": sr}


def _trim_audio_to(audio, seconds: float):
    """把音轨严格裁到 seconds 秒（不够就尾部粘最后一帧，不绕回开头）。"""
    if audio is None:
        return None
    sr = int(audio["sample_rate"])
    need = int(round(seconds * sr))
    wav = audio["waveform"]
    if wav.shape[-1] == need:
        return audio
    if wav.shape[-1] > need:
        return {"waveform": wav[..., :need], "sample_rate": sr}
    pad_shape = list(wav.shape)
    pad_shape[-1] = need - wav.shape[-1]
    padded_tail = wav[..., -1:].expand(*pad_shape).contiguous()
    return {"waveform": torch.cat([wav, padded_tail], dim=-1), "sample_rate": sr}


# ===========================================================================
# 主节点
# ===========================================================================
class SW_H3Ultra_NCG(io.ComfyNode):
    """SW 海螺 H3 长视频超级进化版：一个节点出 15 ~ 85 秒长视频。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SW_H3Ultra_NCG",
            display_name="SW 海螺H3长视频超级进化版【无CFG·NCG】",
            category="SW/H3Segments",
            description=(
                "一个节点干完：帧数规划 → 参考资产编码 → 逐段编译条件 → 逐段采样"
                " → latent 续接 → 参考接力回灌 → 外部歌音轨切片对齐 → 一次性 VAE 解码。"
                "提示词用 Autogrow 口（prompt_1..12，接几个跑几段），"
                "接歌曲则对口型并替换成片音轨。除模型/素材/建视频外不再需要任何中间节点。"),
            inputs=[
                # ---------------- 模型 ----------------
                io.Model.Input("model", tooltip="H3 权重（可接 LoRA / SageAttn 之后的 model）"),
                io.Clip.Input("clip", tooltip="Qwen3-VL 文本编码器（含视觉塔，参考图要走它）"),
                io.Vae.Input("vae", tooltip="H3 video VAE，最后一次性解码"),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="H3 audio VAE。不接 → 成片静音、歌也无法编码"),

                # ---------------- 提示词 Autogrow ----------------
                io.Autogrow.Input(
                    "prompts", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.String.Input(
                            "prompt", multiline=True, dynamic_prompts=True,
                            tooltip="一段提示词。引用参考图用 <Picture 1> / <Picture 2>…"),
                        names=PROMPT_NAMES, min=1)),
                io.Combo.Input(
                    "segment_count",
                    options=["auto"] + [str(i) for i in range(1, MAX_PROMPTS + 1)],
                    default="auto",
                    tooltip="auto = 按提示词口实际接了几个来定（必须连续）。手动值时锁定段数。"),

                # ---------------- 参考资产 ----------------
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="参考图。提示词里用 <Picture 1> <Picture 2>… 引用"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video", tooltip="参考视频帧序列（IMAGE），用 <Video 1>… 引用"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio", tooltip="与 ref_video_N 同号的配套音轨"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio", tooltip="独立参考音频，用 <Audio 1>… 引用"),
                        prefix="ref_audio_", min=0, max=3)),

                # ---------------- 外部整首歌 / 配音 ----------------
                io.Audio.Input(
                    "song", optional=True,
                    tooltip="外部整首歌或配音。节点会按每段的起点秒自动切片，"
                            "既当该段的条件（对口型），也可直接当最终成片音轨"),
                io.Combo.Input(
                    "音轨来源", options=["H3生成", "外部歌曲", "混合"], default="外部歌曲",
                    tooltip="H3生成 = 用模型自己产的音（不接song时只能用这个）；"
                            "外部歌曲 = 成片音轨直接换成 song（唱歌MV推荐）；"
                            "混合 = 两条按下面权重混合"),
                io.Float.Input(
                    "歌曲音量", default=0.85, min=0.0, max=1.0, step=0.01,
                    tooltip="仅「混合」模式生效：song 在混合音轨里的占比"),

                # ---------------- 参考编码策略 ----------------
                io.Combo.Input(
                    "参考图编码", options=["一次", "每段重编"], default="一次",
                    tooltip="一次 = 参考图 VAE 编码算一遍，全段共用（快）；"
                            "每段重编 = 每段重新编码同一张参考图。"
                            "⚠️ 官方 vae.encode 是确定性的，两种结果逐位相同，"
                            "选「每段重编」只是拿时间换一个你自己验证的机会"),
                io.Combo.Input(
                    "参考接力", options=["off", "latent", "full"], default="latent",
                    tooltip="★ 真正改善后续段画质的一档：把上一段末端画面回灌成下一段的新参考图。"
                            "off = 不回灌；"
                            "latent = 纯 latent 切片回灌（免费，推荐）；"
                            "full = 另外再解码回像素给 Qwen 视觉塔看（最强，CLIP 需常驻）"),
                io.Int.Input(
                    "接力窗口", default=1, min=1, max=4, step=1,
                    tooltip="回灌最近几段的画面。1 = 只回灌上一段（开销恒定，推荐）；"
                            "调大 = 参考 token 变多，注意序列长度"),

                # ---------------- 画布与时长 ----------------
                io.Int.Input("width", default=768, min=32, max=8192, step=32),
                io.Int.Input("height", default=1344, min=32, max=8192, step=32),
                io.Float.Input(
                    "segment_seconds", default=7.0, min=0.5, max=20.0, step=0.1,
                    tooltip="每段目标秒数，自动吸附到 H3 合法帧数网格 5+17n"),
                io.Combo.Input(
                    "anchor_frames", options=["5", "22"], default="5",
                    tooltip="段间锚定帧数。5 帧（2 token）标准；"
                            "22 帧（7 token）段间拉得更紧、更难漂，代价是成片少 17 帧/接口"),

                # ---------------- 采样 ----------------
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Boolean.Input("递增种子", default=True,
                                 tooltip="第 N 段用 seed+(N-1)，段间噪声不同，过渡更自然"),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                               default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS,
                               default="beta",
                               tooltip="★ beta = H3 官方 BasicScheduler 的调度器，务必保持。"
                                       "simple 在高噪声段跨步过大，蒸馏 LoRA 下会让"
                                       "latent 冲出分布 → 画面泛紫。"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),

                # ---------------- 条件细节 ----------------
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="match = 按成片面积等比缩（快）；"
                                       "max = 2048 短边，人物一致性最好但慢很多"),
                io.Float.Input("visual_strength", default=0.999, min=0.0, max=1.0, step=0.001),
                io.Float.Input("audio_strength", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Boolean.Input("内置sigma_shift", default=True),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),

                # ---------------- 输出 ----------------
                io.Boolean.Input("内部解码", default=True,
                                 tooltip="False = 只出 LATENT，自己接官方 VAE Decode"),
                io.Boolean.Input("卸载CLIP", default=True,
                                 tooltip="采样前卸掉 CLIP 省显存。"
                                         "★ 参考接力=full 时会被自动忽略（后面还要用视觉塔）"),
                io.Float.Input("裁剪到秒数", default=0.0, min=0.0, max=600.0, step=0.5,
                               tooltip="0 = 不裁。填具体秒数时优先让末段少生成一点而不是砍尾"),
            ],
            outputs=[
                io.Image.Output(display_name="video"),
                io.Audio.Output(display_name="audio"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="段数"),
                io.Int.Output(display_name="成片帧数"),
                io.Float.Output(display_name="成片秒数"),
                io.Int.Output(display_name="每段帧数"),
                io.String.Output(display_name="报告"),
            ],
        )

    # ---------------------------------------------------------------- #
    @classmethod
    def execute(cls, model, clip, vae, audio_vae=None,
                prompts=None, segment_count="auto",
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
                song=None, 音轨来源="外部歌曲", 歌曲音量=0.85,
                参考图编码="一次", 参考接力="latent", 接力窗口=1,
                width=768, height=1344, segment_seconds=7.0, anchor_frames="5",
                seed=0, 递增种子=True, steps=20,
                sampler_name="euler", scheduler="beta", denoise=1.0,
                ref_image_size="match", visual_strength=0.999, audio_strength=1.0,
                内置sigma_shift=True, shift_video=12.0, shift_audio=3.0,
                内部解码=True, 卸载CLIP=True, 裁剪到秒数=0.0) -> io.NodeOutput:

        width = _as_int(width, 768)
        height = _as_int(height, 1344)
        anchor = _as_int(anchor_frames, 5)
        if anchor not in (5, 22):
            anchor = 5
        接力窗口 = max(1, min(4, _as_int(接力窗口, 1)))
        relay_mode = str(参考接力 or "off").strip().lower()
        reencode_each = str(参考图编码 or "一次").strip() == "每段重编"

        # ================= 1. 段数判定 =================
        ordered = _sorted_prompts(prompts)
        filled = {idx: text for idx, text in ordered if text}
        mode = str(segment_count or "auto").strip().lower()

        if mode == "auto":
            n_seg = 0
            for i in range(1, MAX_PROMPTS + 1):
                if filled.get(i):
                    n_seg = i
                else:
                    break
            later = [i for i in filled if i > n_seg + 1]
            stop_reason = ("第 %s 段有提示词但第 %d 段是空的 —— 段号必须连续"
                           % ("/".join(str(i) for i in later), n_seg + 1)) if later else ""
            if n_seg < 1:
                raise ValueError(
                    "[SW-H3Ultra] 一个提示词口都没接。请把 prompt_1 连上一个文本节点"
                    "（PrimitiveStringMultiline / SW_TextConcat 之类），本节点的提示词口是 Autogrow 输入端口，"
                    "不支持在节点面板里直接打字。")
        else:
            n_seg = max(1, min(MAX_PROMPTS, _as_int(mode, 1)))
            stop_reason = ""
            missing = [i for i in range(1, n_seg + 1) if not filled.get(i)]
            if missing:
                raise ValueError(
                    "[SW-H3Ultra] segment_count 锁了 %d 段，但第 %s 段的提示词口是空的。"
                    % (n_seg, "/".join(str(i) for i in missing)))

        texts = [filled.get(i, "") for i in range(1, n_seg + 1)]

        # ================= 2. 帧数规划 =================
        seg_frames = _align_frame_count(max(FRAME_BASE, int(round(_as_float(segment_seconds, 7.0) * H3_FPS))))
        seg_tokens = _video_latent_t(seg_frames)
        anchor_tokens = _anchor_token_count(anchor)
        if anchor >= seg_frames:
            anchor, anchor_tokens = 5, _anchor_token_count(5)
        step_frames = seg_frames - anchor
        crop_target = _as_float(裁剪到秒数, 0.0)
        crop_target = int(round(crop_target * H3_FPS)) if crop_target > 0 else 0

        seg_frames_list = [seg_frames] * n_seg
        last_shortened, tail_chop_fallback = False, False
        if crop_target > 0 and n_seg >= 2:
            lead_frames = seg_frames + step_frames * (n_seg - 2)
            remainder = crop_target - lead_frames
            if remainder >= FRAME_BASE:
                seg_frames_list[-1] = _align_frame_count_floor(remainder)
                last_shortened = True
            else:
                tail_chop_fallback = True

        # 每段在成片里的起点秒（给歌切片用）
        # 拼接时会丢掉每段的 anchor 帧，所以步长 = seg_frames - anchor
        starts, acc_frames = [], 0
        for i in range(n_seg):
            starts.append(acc_frames / float(H3_FPS))
            acc_frames += seg_frames_list[i] - anchor

        # ================= 3. model patch =================
        if _as_bool(内置sigma_shift, True):
            model = _patch_sigma_shift(model, _as_float(shift_video, 12.0),
                                       _as_float(shift_audio, 3.0))

        # ================= 4. 参考资产 =================
        # 老节点在这里只编码一次；本节点支持「每段重编」，但结果逐位相同（见模块 docstring）
        def _build_once():
            return _build_refs(vae, audio_vae, ref_image_size, width, height,
                               ref_images=ref_images, ref_videos=ref_videos,
                               ref_video_audios=ref_video_audios, ref_audios=ref_audios)

        ref_items, ref_blocks, ref_counts = _build_once()
        static_item_count = len(ref_items)
        print("[SW-H3Ultra] 参考资产：图 %d / 视频 %d（含音轨 %d）/ 音频 %d 条；"
              "编码模式=%s；接力=%s（窗口 %d）"
              % (ref_counts["image"], ref_counts["video"], ref_counts["video_audio"],
                 ref_counts["audio"],
                 "每段重编" if reencode_each else "一次", relay_mode,
                 接力窗口 if relay_mode != "off" else 0), flush=True)

        # ================= 5. 逐段采样 =================
        seed0 = _as_int(seed, 0)
        stride = 1 if _as_bool(递增种子, True) else 0
        steps_i = _as_int(steps, 20)
        denoise_f = _as_float(denoise, 1.0)
        total_steps = max(1, steps_i * n_seg)
        bar = comfy.utils.ProgressBar(total_steps)
        done = 0

        pieces, prev_video = [], None
        relay_history = []          # 最近几段的末端 latent（给接力用）
        clip_unloaded = False
        relay_full_ok = (relay_mode == "full")

        for i in range(n_seg):
            seg_seed = (seed0 + stride * i) % 0x10000000000000000

            # ---- 5a. 每段的参考资产（重编模式在这里生效）--------------
            if reencode_each and i > 0:
                seg_items, seg_blocks, seg_counts = _build_once()
            else:
                seg_items, seg_blocks = ref_items, ref_blocks

            items = list(seg_items)
            blocks = list(seg_blocks)

            # ---- 5b. 参考接力：past段末端画面 -> 新 ref_img token ----
            if relay_mode != "off" and prev_video is not None:
                for past in relay_history[-接力窗口:]:
                    blocks.append(_relay_block(past, height, width))
                    if relay_full_ok:
                        pix = _relay_pixel(vae, past)
                        if pix is not None:
                            items.append({"type": "image", "data": pix})

            # ---- 5c. 文本/视觉编码 ----
            tokens = clip.tokenize(texts[i], minimax_ref_items=items)
            cond = clip.encode_from_tokens_scheduled(tokens)

            if blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks})
            if _as_float(visual_strength, 0.999) < 1.0 or _as_float(audio_strength, 1.0) < 1.0:
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_visual_cond_noise_aug": _as_float(visual_strength, 0.999),
                    "minimax_audio_cond_noise_aug": _as_float(audio_strength, 1.0),
                })

            # ---- 5d. 首帧锚定（video latent） + 歌曲切片（audio latent）
            latent, frames_this = _empty_av_latent(width, height, seg_frames_list[i])
            kf = {}
            if prev_video is not None and prev_video.shape[2] > anchor_tokens:
                kf["resolved_frame_index"] = 0
                kf["latent"] = prev_video[:, :, -anchor_tokens:, :, :].clone()

            song_piece = None
            if song is not None and audio_vae is not None:
                song_piece = _slice_song(song, starts[i], seg_frames_list[i] / float(H3_FPS))
                audio_latent, rt = _encode_ref_audio(audio_vae, song_piece)
                samples = latent["samples"]
                max_rt = math.floor(samples.tensors[1].shape[-1] -
                                    FRAME_RESCALE * 0)
                max_rt = max(1, int(max_rt))
                if rt > max_rt:
                    audio_latent = audio_latent[..., :max_rt].clone()
                kf["resolved_frame_index"] = 0
                kf["audio_latent"] = audio_latent
            if kf:
                cond = node_helpers.conditioning_set_values(
                    cond, {"minimax_keyframes": [kf]})

            # ================= 无 CFG 路径（NCG 版核心） =================
            # 照官方 BasicGuider：negative=None + cfg 固定 1.0。
            # samplers.py:610 命中 isclose(1.0) → uncond_=None → 只前向 cond，省一半算力。
            negative = None
            noise = comfy.sample.prepare_noise(latent["samples"], seg_seed)

            def _cb(p, x0, x, total, noisy_samples=None, _done=done):
                bar.update_absolute(_done + p + 1)
                comfy.model_management.throw_exception_if_processing_interrupted()
                return None

            # ---- 5e. CLIP 卸载 ----
            # 最后一段的条件编完之后就再也不需要 CLIP 了，趁开采前卸掉省显存。
            # 接力=full 例外：后面还要用它的视觉塔看中继画面。
            if (_as_bool(卸载CLIP, True) and not relay_full_ok
                    and not clip_unloaded and i == n_seg - 1):
                how = _free_text_encoder(clip)
                clip_unloaded = True
                print("[SW-H3Ultra] CLIP 编码器已卸载（%s）"
                      % ("全部模型" if how == -1 else "%d 个模型对象" % how), flush=True)

            sampled = comfy.sample.sample(
                model, noise, steps_i, 1.0, sampler_name, scheduler,
                cond, negative, latent["samples"],
                denoise=denoise_f, disable_noise=False, callback=_cb, seed=seg_seed)

            video, audio = _av_parts({"samples": sampled})
            pieces.append((video, audio))
            prev_video = video
            if relay_mode != "off":
                keep = min(2, int(video.shape[2]))
                relay_history.append(video[:, :, -keep:, :, :].clone())
                relay_history = relay_history[-接力窗口:]
            done += steps_i

            print("[SW-H3Ultra] 段 %d/%d 完成：%d 帧 / %d token（seed=%d%s）"
                  % (i + 1, n_seg, frames_this, int(video.shape[2]), seg_seed,
                     "，接力已回灌" if relay_mode != "off" and i > 0 else ""), flush=True)

        if not clip_unloaded and _as_bool(卸载CLIP, True) and not relay_full_ok:
            how = _free_text_encoder(clip)
            clip_unloaded = True
            print("[SW-H3Ultra] CLIP 编码器已卸载（%s）"
                  % ("全部模型" if how == -1 else "%d 个模型对象" % how), flush=True)
        if relay_full_ok:
            print("[SW-H3Ultra] 参考接力=full：全程保留 CLIP（每段都要用视觉塔），"
                  "显存换一致性。跑完记得确认没有 OOM。", flush=True)

        # ================= 6. latent 拼接 =================
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
            target_len = int(round(real_frames * FRAME_RESCALE))
            if merged_audio.shape[-1] > target_len:
                merged_audio = merged_audio[..., :target_len]
            elif merged_audio.shape[-1] < target_len:
                pad = target_len - merged_audio.shape[-1]
                merged_audio = torch.cat(
                    [merged_audio, merged_audio[..., -1:].repeat(1, 1, 1, pad)], dim=-1)

        merged = {"samples": comfy.nested_tensor.NestedTensor((merged_video, merged_audio))
                  if merged_audio is not None else merged_video}
        merged.pop("noise_mask", None)

        # ================= 7. 解码 =================
        video_batch, audio_out = None, None
        out_frames, out_seconds = real_frames, real_frames / float(H3_FPS)
        do_decode = _as_bool(内部解码, True)
        cropped = False
        smooth_info = None

        if do_decode:
            # VAE 时间分块保持官方默认。泛紫根因是 scheduler（simple→beta），
            # 不是解码分块—— 此处不再改token_drop，避免画面跳动。
            images = vae.decode(merged_video)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            video_batch = images
            out_frames = int(images.shape[0])
            out_seconds = out_frames / float(H3_FPS)

            h3_audio = None
            if merged_audio is not None and audio_vae is not None:
                wav = audio_vae.decode(merged_audio).movedim(-1, 1)
                h3_audio = {"waveform": wav, "sample_rate": AUDIO_SAMPLE_RATE}
            elif merged_audio is not None:
                print("[SW-H3Ultra] 没接 audio_vae，H3 音轨无法解码（若用外部歌曲则不影响）", flush=True)

            # ---- 音轨策略 ----
            songs_trim = _trim_audio_to(song, out_seconds) if song is not None else None
            source = str(音轨来源 or "H3生成").strip()
            if source == "外部歌曲":
                audio_out = songs_trim
            elif source == "混合":
                audio_out = _mix_audio(songs_trim, h3_audio, _as_float(歌曲音量, 0.85))
            else:
                audio_out = h3_audio
            if audio_out is not None:
                audio_out = _trim_audio_to(audio_out, out_seconds)

            if crop_target > 0 and crop_target < out_frames:
                video_batch = video_batch[:crop_target]
                out_frames = crop_target
                out_seconds = out_frames / float(H3_FPS)
                if audio_out is not None:
                    audio_out = _trim_audio_to(audio_out, out_seconds)
                cropped = True

        # ================= 8. 报告 =================
        grid_ok = (total_tokens - 2) % 5 == 0
        relay_desc = {"off": "关闭", "latent": "潜空间切片回灌（零额外开销）",
                      "full": "潜空间 + Qwen 视觉塔（CLIP 常驻）"}.get(relay_mode, str(参考接力))
        lines = [
            "============ SW 海螺H3长视频超级进化版 %s ============" % SW_H3_ULTRA_VERSION,
            "段数    : %d 段（来源：%s）%s"
            % (n_seg, "auto" if mode == "auto" else "手动锁定",
               ("  ⚠️ " + stop_reason) if stop_reason else ""),
            "每段    : %d 帧 / %.4f 秒（%.2fs 请求 → 5+17n 网格）"
            % (seg_frames, seg_frames / float(H3_FPS), _as_float(segment_seconds, 7.0)),
            "锚定    : %d 帧 = %d token（第 2 段起开头复刻上段末帧，拼接时丢弃）"
            % (anchor, anchor_tokens),
            "末段    : %s"
            % ("补足到 %d 帧（目标 %d 秒）" % (seg_frames_list[-1], round(crop_target / H3_FPS))
               if last_shortened else
               ("⚠️ 砍尾回退：前 %d 段已超目标，建议减段数或放宽时长" % (n_seg - 1)
                if tail_chop_fallback else "标准 %d 帧" % seg_frames)),
            "每段保留: " + ", ".join("段%d %d token" % (i + 1, n) for i, n in enumerate(per_seg)),
            "合并后  : %d token → %d 帧 / %.4f 秒" % (total_tokens, real_frames, real_frames / float(H3_FPS)),
            "网格校验: %s" % ("✅ 5c+2" if grid_ok else "⚠️ 偏离 5c+2"),
            "参考资产: 图 %d / 视频 %d（含音轨 %d）/ 音频 %d 条"
            % (ref_counts["image"], ref_counts["video"], ref_counts["video_audio"],
               ref_counts["audio"]),
            "参考编码: %s%s"
            % ("每段重编（⚠️ 结果与一次编码逐位相同，仅供验证）" if reencode_each else "一次，全段共用",
               "；static ref_items %d 个" % static_item_count),
            "参考接力: %s" % relay_desc,
            "歌曲驱动: %s" % ("已接 song，逐段切片对齐 + 成片音轨=%s" % 音轨来源
                              if song is not None else "未接（纯 H3 自產音）"),
            "采样    : %d 步 × %d 段，%s / %s，★无 CFG（cond 单路，负样本跳过），seed=%d%s"
            % (steps_i, n_seg, sampler_name, scheduler, seed0,
               "（逐段 +1）" if stride else "（全段同 seed）"),
            "CLIP    : %s" % ("已卸载" if clip_unloaded else
                              ("接力=full，全程常驻" if relay_full_ok else "未卸载")),
            "解码    : %s" % (("已解码 %d 帧" % out_frames) if do_decode else
                              "未解码（内部解码关闭，请接官方 VAE Decode）"),
            "成片    : %d 帧 / %.4f 秒%s / 音轨 %s"
            % (out_frames, out_seconds, "（已裁剪）" if cropped else "",
               ("有 %.1fs 音轨" % (audio_out["waveform"].shape[-1] / float(audio_out["sample_rate"])))
               if audio_out is not None else "无"),
            "===============================================",
        ]
        report = "\n".join(lines)
        print("[SW-H3Ultra]\n" + report, flush=True)

        return io.NodeOutput(
            video_batch, audio_out, merged, n_seg, out_frames,
            round(out_seconds, 6), seg_frames, report,
        )


NODE_CLASS_MAPPINGS = {"SW_H3Ultra_NCG": SW_H3Ultra_NCG}
NODE_DISPLAY_NAME_MAPPINGS = {"SW_H3Ultra_NCG": "SW 海螺H3长视频进化版 NCG【无CFG】"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
