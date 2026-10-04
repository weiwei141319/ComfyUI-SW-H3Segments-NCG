"""SW H3 分段 latent 拼接节点包（每 7 秒一段，最多 5 段，最后一起解码）。

================================================================================
★ 这套东西解决什么
================================================================================
H3 单段只能出 ~15 秒。要做 35 秒，常规做法是「每段单独采样 + 每段单独 VAE 解码
+ 像素域拼接」。本包改走 latent 域：

    每段只采样，不解码  →  段与段用 latent 尾部 token 续接  →  最后拼成一个大
    latent  →  官方 VAEDecode 一次性解码

省掉 N-1 次全片解码 + N 次 guide 重编码，自定义节点只有 3 个。

================================================================================
★ 帧网格（逐条来自官方源码，不是猜的）
================================================================================
    comfy/ldm/minimax/model.py:30
        FRAME_PER_TOKEN = (1, 4, 4, 4, 4)      # latent token k 占几帧，按 k%5 循环
    comfy/ldm/minimax/model.py:31
        FRAME_RESCALE = 5.0 / 3.0              # 1 像素帧 = 5/3 个音频 latent
    comfy_extras/nodes_minimax_h3.py:37
        def align_frame_count(n):              # 合法帧数 = 5 + 17n
    comfy_extras/nodes_minimax_h3.py:43
        def video_latent_t(frame_count):       # = 2 if f<=5 else ((f-5)//17)*5+2

    推得：**每 5 个 latent token 精确对应 17 帧**（1+4+4+4+4），且第 1 个 token
    只占 1 帧。所以帧数 F = 17c + 5  ⟺  latent token 数 T = 5c + 2。

    7 秒 = 168 帧 → 吸附到 5+17n 得 **175 帧**（7.29s），T = 52。

================================================================================
★ 拼接为什么不会错位
================================================================================
段长取 175 帧（c=10, T=52），段间锚定 5 帧（= 前 2 个 token）。

    段 1        tokens[0:52]   = 175 帧（开头 5 帧是参考图种子，成片要留）
    段 2..N     tokens[2:52]   = 170 帧（开头 5 帧是上段末 5 帧的复刻，丢掉）

    总 token = 52 + 50×(N-1) = 50N + 2 = 5×(10N) + 2
    总帧数   = 175 + 170×(N-1) = 170N + 5 = 17×(10N) + 5   ✅ 仍落在网格上

★ 锚定位置编码天然对齐（关键巧合，已核对）：
    段长 T = 5c+2，最后 2 个 token 的下标是 5c 和 5c+1 → k%5 = 0, 1；
    段首 2 个 token 的下标是 0 和 1 → k%5 = 0, 1。
    **两者 k%5 相同**，所以把上段尾部 2 个 token 注入到本段 frame_idx=0 时，
    时间位置编码天然对得上，不需要重排。

    N 段时长：1→7.29s  2→14.38s  3→21.46s  4→28.54s  5→35.63s

================================================================================
★ 为什么可以「一次性解码」超长 latent
================================================================================
    comfy/ldm/minimax/vae.py:408
        class MiniMaxH3VideoVAE: comfy_has_chunked_io = True
    comfy/ldm/minimax/vae.py:704
        def decode_temporal(self, z, output_buffer=None):
            # 按 tokens_chunk_size=5 分块，带 token_overlap=2 重叠，chunk 之间
            # 用 blend() 线性混合 → 任意长度都能解，边界无缝，且逐块释放显存

所以把 252 个 token 的超长 latent 直接丢给官方 VAEDecode 就行，VAE 自己分块。

★ 官方 VAEDecode 只解视频流（正好对上「音轨用原曲」的需求）：
    nodes.py: VAEDecode.decode()
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]        # ← 只取 video，audio 那路不解码
"""

from __future__ import annotations

import logging

import torch

import comfy.nested_tensor
import node_helpers
from comfy_api.latest import io

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# H3 官方常数（与本机的 comfy.ldm.minimax.model / comfy_extras.nodes_minimax_h3
# 逐字一致；导入失败时回落到本地副本，保证节点仍能注册）
# --------------------------------------------------------------------------- #
try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:  # pragma: no cover - 只在上游结构变化时触发
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

H3_FPS = 24
FRAME_BASE = 5
FRAME_STEP = 17


def _align_frame_count(n) -> int:
    """向上吸附到 5+17n（与官方 align_frame_count 一致）。"""
    n = int(n)
    if n < FRAME_BASE:
        return FRAME_BASE
    while n % FRAME_STEP != FRAME_BASE:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    """帧数 -> video latent token 数（与官方 video_latent_t 一致）。"""
    fc = int(frame_count)
    return 2 if fc <= FRAME_BASE else ((fc - FRAME_BASE) // FRAME_STEP) * 5 + 2


def _frames_of_tokens(n_tokens: int) -> int:
    """latent token 数 -> 像素帧数。FRAME_PER_TOKEN 按 k%5 循环累加。"""
    return int(sum(FRAME_PER_TOKEN[k % 5] for k in range(int(n_tokens))))


def _anchor_token_count(anchor_frames: int) -> int:
    """段首锚定 frames 帧需要几个 latent token（从段首第 0 个 token 起算）。"""
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
    """容错取值：''/None 一律回落，杜绝前端空串把整条 prompt 废掉。"""
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


def _unwrap(value, default=None):
    """剥掉框架回灌通道（current_iteration_value）包的那层壳。

    LoopIteration.current_iteration_value 声明为 is_output_list=True，
    下游拿到的是 list 包装。实测形态可能是 [latent] 或 [[latent]]，
    这里按「剥到出现带 samples 的 dict 为止」处理，兼容两种。
    """
    seen = 0
    while (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and seen < 4
        and not (isinstance(value[0], dict) and "samples" in value[0])
    ):
        value = value[0]
        seen += 1
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return default
    return value


def _av_parts(latent):
    """AV latent -> (video, audio)；不是嵌套张量时返回 (tensor, None)。"""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        video = parts[0]
        audio = parts[1] if len(parts) > 1 else None
        return video, audio
    return samples, None


# ===========================================================================
# 1. SW_H3_SegPlan_NCG —— 循环外：分段计划
# ===========================================================================
class SW_H3_SegPlan_NCG(io.ComfyNode):
    """把「每段 7 秒 × 最多 5 段」编译成官方节点能直接吃的参数。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SW_H3_SegPlan_NCG",
            display_name="SW H3 分段计划【无CFG·NCG】",
            category="SW/H3Segments",
            description=(
                "算出每段帧数（吸附到 5+17n 网格）、每段 latent token 数、成片总帧数。"
                "「单段帧数」直接接官方 MiniMax H3 图生视频/参考生视频的 length；"
                "「段数」接 StartLoop（For 模式的 max_iteration 前请先过 PrimitiveInt）。"
            ),
            inputs=[
                io.Int.Input(
                    "width", default=1344, min=32, max=8192, step=32,
                    tooltip="输出宽度，32 的倍数。长视频建议先跑 960x544 压内存。",
                ),
                io.Int.Input(
                    "height", default=768, min=32, max=8192, step=32,
                    tooltip="输出高度，32 的倍数。",
                ),
                io.Float.Input(
                    "segment_seconds", default=7.0, min=0.5, max=20.0, step=0.1,
                    tooltip="每段目标秒数。7 秒会吸附到 175 帧（7.29s），因为 H3 合法帧数必须是 5+17n。",
                ),
                io.Int.Input(
                    "segment_count", default=5, min=1, max=5, step=1,
                    tooltip="分段数量，最多 5 段。1段=7.29s / 2段=14.38s / 3段=21.46s / 4段=28.54s / 5段=35.63s。",
                ),
                io.Combo.Input(
                    "anchor_frames", options=["5", "1"], default="5",
                    tooltip="段间锚定帧数（= 下一段开头复刻上一段末尾的帧数，拼接时丢掉）。"
                            "★ 5 帧是推荐值：此时总 token 数 = 50N+2，仍严格落在 5+17n 网格上；"
                            "1 帧只在你想让段间衔接更紧时用，但总 token 会偏离网格。",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="单段帧数"),
                io.Int.Output(display_name="段数"),
                io.Int.Output(display_name="单段token"),
                io.Int.Output(display_name="成片帧数"),
                io.Float.Output(display_name="成片秒数"),
                io.Int.Output(display_name="锚定帧数"),
                io.Int.Output(display_name="宽"),
                io.Int.Output(display_name="高"),
                io.String.Output(display_name="计划报告"),
            ],
        )

    @classmethod
    def execute(cls, width=1344, height=768, segment_seconds=7.0,
                segment_count=5, anchor_frames="5") -> io.NodeOutput:
        width = _as_int(width, 1344)
        height = _as_int(height, 768)
        segment_seconds = _as_float(segment_seconds, 7.0)
        segment_count = max(1, min(5, _as_int(segment_count, 5)))
        anchor = _as_int(anchor_frames, 5)
        if anchor not in (1, 5):
            anchor = 5

        target_frames = max(FRAME_BASE, int(round(segment_seconds * H3_FPS)))
        seg_frames = _align_frame_count(target_frames)
        seg_tokens = _video_latent_t(seg_frames)

        if anchor >= seg_frames:
            anchor = 5

        step_frames = seg_frames - anchor
        total_frames = seg_frames + step_frames * (segment_count - 1)
        total_seconds = total_frames / H3_FPS

        anchor_tokens = _anchor_token_count(anchor)
        total_tokens = seg_tokens + (seg_tokens - anchor_tokens) * (segment_count - 1)
        grid_ok = (total_tokens - 2) % 5 == 0

        window = " | ".join(
            ("段1 [0-%d) %d帧" % (seg_frames, seg_frames))
            if i == 0
            else ("段%d 丢首%d帧 取%d帧" % (i + 1, anchor, step_frames))
            for i in range(segment_count)
        )

        report = "\n".join([
            "===== SW H3 分段计划 =====",
            "每段    : %d 帧 / %.4f s  (%.3fs 请求 → 吸附到 5+17n)"
            % (seg_frames, seg_frames / H3_FPS, segment_seconds),
            "latent  : 每段 %d token  (%d 帧 = 5×%d+2)"
            % (seg_tokens, seg_frames, (seg_frames - FRAME_BASE) // FRAME_STEP),
            "锚定    : %d 帧 = %d token（下段开头复刻，拼接时丢弃）" % (anchor, anchor_tokens),
            "分段窗口: " + window,
            "成片    : %d 帧 / %.4f s  (%d 段)" % (total_frames, total_seconds, segment_count),
            "总token : %d %s" % (total_tokens, "✅ 落在 5c+2 网格" if grid_ok else "⚠️ 偏离网格，仅 5 帧锚定能保证"),
            "画布    : %dx%d  → latent %dx%d" % (width, height, height // 16, width // 16),
            "★ 只在 latent 域拼接，最后一次性 VAE 解码（VAE 自带时间分块，边界无缝混合）",
            "==========================",
        ])
        print("[SW-H3Seg]\n" + report)

        return io.NodeOutput(
            seg_frames, segment_count, seg_tokens, total_frames,
            round(total_seconds, 6), anchor, width, height, report,
        )


# ===========================================================================
# 2. SW_H3_SegBridge_NCG —— 循环体内：把上一段的尾部 latent 注入本段条件
# ===========================================================================
class SW_H3_SegBridge_NCG(io.ComfyNode):
    """循环体内续接节点：拿上一段的尾部 latent token 当本段的开头锚点。

    等价于官方 MiniMaxH3AddGuide 干的事，但**吃 latent 不吃像素**，所以全程
    不需要 VAE 编码/解码。

    依据 nodes_minimax_h3.py:160：
        kf["latent"] = vae.encode(kf.pop("image"))
    keyframe 条件本质就是一个 latent 张量，塞进去即可。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SW_H3_SegBridge_NCG",
            display_name="SW H3 分段续接（循环体内）",
            category="SW/H3Segments",
            description=(
                "放在循环体内。从 StartLoop 的 current_iteration_value 拿到上一段采样结果，"
                "切下尾部 latent token 注入本段条件（minimax_keyframes，frame_idx=0），"
                "让本段开头严格复刻上段结尾。首段无上一段时原样透传。"
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent",
                                tooltip="本段的空 AV latent（官方 H3 图生视频/参考生视频节点产出）。"),
                io.Latent.Input(
                    "prev_latent", optional=True,
                    tooltip="接 StartLoop 的 current_iteration_value（上一段采样结果）。首轮为空。",
                ),
                io.Int.Input(
                    "anchor_frames", default=5, min=1, max=5, step=4,
                    tooltip="锚定多少帧（1 或 5）。必须与 SW_H3_SegConcat_NCG 的一致。",
                ),
                io.Boolean.Input(
                    "续接音频", default=False,
                    tooltip="把上一段的尾部音频 latent 也注入本段。成片音轨用原曲时保持关闭即可。",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.Boolean.Output(display_name="是否续接段"),
                io.String.Output(display_name="续接报告"),
            ],
        )

    @classmethod
    def execute(cls, positive, latent, prev_latent=None, anchor_frames=5,
                续接音频=False) -> io.NodeOutput:
        anchor = _as_int(anchor_frames, 5)
        if anchor not in (1, 5):
            anchor = 5
        carry_audio = _as_bool(续接音频, False)

        prev = _unwrap(prev_latent, None)
        if prev is None or not isinstance(prev, dict) or "samples" not in prev:
            return io.NodeOutput(
                positive, latent, False,
                "首段：无上一段 latent，不做续接（开头由参考图/首帧锚定）。",
            )

        video, audio = _av_parts(prev)
        if video is None:
            return io.NodeOutput(positive, latent, False, "上一段无 video latent，跳过续接。")

        tokens = _anchor_token_count(anchor)
        if video.shape[2] <= tokens:
            return io.NodeOutput(
                positive, latent, False,
                "上段视频 latent 只有 %d 个 token，不足以切出 %d 个锚定 token，跳过续接。"
                % (video.shape[2], tokens),
            )

        tail_video = video[:, :, -tokens:, :, :].clone()

        keyframe = {"resolved_frame_index": 0, "latent": tail_video}

        if carry_audio and audio is not None:
            # 1 像素帧 = FRAME_RESCALE 个音频 latent
            take = int(round(anchor * FRAME_RESCALE))
            take = max(1, min(take, audio.shape[-1]))
            keyframe["audio_latent"] = audio[..., -take:].clone()

        existing = list(positive[0][1].get("minimax_keyframes", []))
        existing.append(keyframe)
        positive = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": existing}
        )

        report = (
            "续接：注入上段尾部 %d token（=%d 帧）作为本段 frame_idx=0 锚点%s。"
            "上段 token 下标 k%%5=%s 与本段段首一致，时间位置编码天然对齐。"
            % (
                tokens, anchor,
                "，同时注入尾部音频 latent" if carry_audio else "",
                ", ".join(str((video.shape[2] - tokens + i) % 5) for i in range(tokens)),
            )
        )
        print("[SW-H3Seg] " + report)
        return io.NodeOutput(positive, latent, True, report)


# ===========================================================================
# 3. SW_H3_SegConcat_NCG —— 循环外：N 段 latent 拼成一个
# ===========================================================================
class SW_H3_SegConcat_NCG(io.ComfyNode):
    """循环外拼接节点：把 EndLoop 攒下的 N 段 latent 沿时间维拼成一个大 latent。

    接法：EndLoop 的 accumulate 打开，output_value 接每段的采样输出；
    EndLoop 的 outputs 就是「每段 latent 组成的列表」，直接接到这里。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SW_H3_SegConcat_NCG",
            display_name="SW H3 分段拼接（一次解码）",
            category="SW/H3Segments",
            description=(
                "把 N 段 H3 latent 沿时间维拼成一个：第 1 段全留，第 2 段起丢掉开头"
                "的锚定 token（那是上段末帧的复刻）。输出直接接官方 VAE Decode —— "
                "H3 的 VAE 自带时间分块（comfy_has_chunked_io），会自己滑窗解码并混合重叠区，"
                "所以任意长度都能一次性解出来，边界无缝。"
            ),
            is_input_list=True,
            inputs=[
                io.Latent.Input(
                    "latents",
                    tooltip="接 EndLoop 的 outputs（accumulate 打开，收集每一段）。",
                ),
                io.Int.Input(
                    "anchor_frames", default=5, min=1, max=5, step=4,
                    tooltip="锚定帧数（1 或 5）。必须与 SW_H3_SegBridge_NCG 的一致，否则段间会重复或跳帧。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="总帧数"),
                io.Float.Output(display_name="总秒数"),
                io.String.Output(display_name="拼接报告"),
            ],
        )

    @classmethod
    def execute(cls, latents, anchor_frames=5) -> io.NodeOutput:
        # is_input_list=True ⇒ 框架已包成 list；保险起见再规范一次
        if isinstance(latents, dict):
            latents = [latents]
        elif isinstance(latents, (list, tuple)):
            latents = [item for item in latents if isinstance(item, dict) and "samples" in item]
        if not latents:
            raise ValueError("[SW-H3Seg] SegConcat 没收到任何 latent，请检查 EndLoop 的 accumulate 是否打开、output_value 是否已接。")

        anchor = _as_int(anchor_frames, 5)
        if anchor not in (1, 5):
            anchor = 5
        drop_tokens = _anchor_token_count(anchor)

        videos = []
        audios = []
        per_segment = []
        for index, item in enumerate(latents):
            video, audio = _av_parts(item)
            if video is None:
                raise ValueError("[SW-H3Seg] 第 %d 段没有 video latent。" % (index + 1))
            keep_from = 0 if index == 0 else min(drop_tokens, max(0, video.shape[2] - 1))
            piece = video[:, :, keep_from:, :, :]
            videos.append(piece)
            per_segment.append(int(piece.shape[2]))
            if audio is not None:
                take = 0 if index == 0 else min(int(round(anchor * FRAME_RESCALE)),
                                                max(0, audio.shape[-1] - 1))
                audios.append(audio[..., take:])

        merged_video = torch.cat(videos, dim=2) if len(videos) > 1 else videos[0]
        total_tokens = int(merged_video.shape[2])
        total_frames = _frames_of_tokens(total_tokens)

        merged_audio = None
        if audios:
            merged_audio = torch.cat(audios, dim=-1) if len(audios) > 1 else audios[0]
            # 帧→音频 latent 不是整数倍，段多了会累积漂移；这里按总帧数精确校正
            target_audio = int(round(total_frames * FRAME_RESCALE))
            if merged_audio.shape[-1] > target_audio:
                merged_audio = merged_audio[..., :target_audio]
            elif merged_audio.shape[-1] < target_audio:
                pad = target_audio - merged_audio.shape[-1]
                merged_audio = torch.cat(
                    [merged_audio, merged_audio[..., -1:].repeat(1, 1, 1, pad)], dim=-1
                )

        out = dict(latents[0])
        if merged_audio is not None:
            out["samples"] = comfy.nested_tensor.NestedTensor((merged_video, merged_audio))
        else:
            out["samples"] = merged_video
        out.pop("noise_mask", None)

        grid_ok = (total_tokens - 2) % 5 == 0
        report = "\n".join([
            "===== SW H3 分段拼接 =====",
            "收到    : %d 段" % len(latents),
            "每段保留: " + ", ".join(
                ("段%d %d token" % (i + 1, n)) for i, n in enumerate(per_segment)),
            "丢弃    : 第 2 段起各丢 %d 个锚定 token（=%d 帧，上段末帧的复刻）"
            % (drop_tokens, anchor),
            "合并后  : %d token → %d 帧 / %.4f s"
            % (total_tokens, total_frames, total_frames / H3_FPS),
            "网格    : %s" % ("✅ 5c+2，与训练网格一致" if grid_ok else "⚠️ 偏离 5c+2"),
            "音频    : " + ("%d latent（已按总帧数校正）" % merged_audio.shape[-1]
                            if merged_audio is not None else "无"),
            "★ 下一步接官方 VAE Decode：VAE 自带时间分块（5 token 一块、2 token 重叠混合），"
            "一次性解完，段边界无缝。" ,
            "==========================",
        ])
        print("[SW-H3Seg]\n" + report)

        return io.NodeOutput(out, total_frames, round(total_frames / H3_FPS, 6), report)


NODE_CLASS_MAPPINGS = {
    "SW_H3_SegPlan_NCG": SW_H3_SegPlan_NCG,
    "SW_H3_SegBridge_NCG": SW_H3_SegBridge_NCG,
    "SW_H3_SegConcat_NCG": SW_H3_SegConcat_NCG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SW_H3_SegPlan_NCG": "SW H3 分段计划【无CFG·NCG】",
    "SW_H3_SegBridge_NCG": "SW H3 分段续接 NCG（循环体内）",
    "SW_H3_SegConcat_NCG": "SW H3 分段拼接 NCG（一次解码）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
