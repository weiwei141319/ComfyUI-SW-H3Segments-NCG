# ComfyUI-SW-H3Segments-NCFG（无 CFG 对照版）

> **这是 `ComfyUI-SW-H3Segments` 的 A/B 对照分支。**
> 唯一差异：采样时**彻底删除 CFG 路径**（`negative=None` + `cfg` 固定 1.0），
> 完全对齐官方 `BasicGuider` 链路。用来和原版做同参数对比。

---

## 一、为什么会有这个版本

原版 `ComfyUI-SW-H3Segments` 虽然默认 `cfg=1.0`，但**保留了 CFG 代码路径**：
构造一个 `_zero_out(cond)`（空条件）当negative，并把它和 cond 一起丢进
`comfy.sample.sample()`。一旦你把 cfg 调到 1.0 以下，就会命中
`samplers.py:598` 的混合公式：

```
cfg_result = uncond_pred + (cond_pred − uncond_pred) × cfg
```

代入空条件 `uncond = 0`：

```
cfg=0.8  →  out = 0 + (cond − 0) × 0.8  =  cond × 0.8
```

**每一步去噪结果被整体乘到 80%** —— 这不是"引导"，是**信号衰减**。
在 4 步蒸馏 LoRA 下，模型输出本已偏离原始分布，再乘 0.8 →
色度向零塌陷→ 低饱和区（皮肤 / 天空 / 草坪）先泛紫。

**本版把这条路彻底删掉**，回到官方做法。

---

## 二、官方是怎么做的

官方 H3 链路用的是 `BasicGuider`：

```python
# comfy_extras/nodes_custom_sampler.py
class CFGGuider:
    def __init__(self, model_patcher):
        self.cfg = 1.0# ← 恒为 1.0

class Guider_Basic(CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})   # ← 只塞 positive
```

`BasicGuider` 的接口里**根本没有 negative**，`cfg` 也恒为 1.0（没人去 `set_cfg`）。
所以官方是**纯条件去噪、零CFG**。

本版照此实现，命中 ComfyUI 的官方优化：

```python
# comfy/samplers.py  sampling_function 第 610 行
if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
    uncond_ = None       # ← 直接丢掉负样本通道
```

`calc_cond_batch` 里`if cond is not None:` 会跳过 None 通道，
**每步只前向 cond 一次**。

> 注：官方 H3 节点（`comfy_extras/nodes_minimax_h3.py`）**没有**设
> `disable_cfg1_optimization`（只有 k_diffusion 采样器内部那条路径会设，而本插件不走那条），
> 所以这个优化在本地能正常命中。已实测确认。

---

## 三、实测证据

用官方钩子 `sampler_calc_cond_batch_function` 截住 conds 构造结果：

| 语义 | conds[0] | conds[1] | 需前向通道数 |
|---|---|---|---|
| 原版 `negative=zero_out(cond)`, cfg=0.8 | conditioning | conditioning | **2** |
| **本版** `negative=None`, cfg=1.0 | conditioning | `None` | **1** |

→ **前向次数减半，等于省一半采样算力。**

数值上`cfg_function` 拿到 `uncond_pred = 0`（占位零）代入公式：

```
out = 0 + (cond_pred − 0) × 1.0 = cond_pred
```

**完全等于 cond 预测，不做任何缩放** —— 画质回到官方基线。

---

## 四、与原版的完整差异

| | 原版 H3Segments | 本版 NCG |
|---|---|---|
| `cfg` 输入口 | 有 | **已删除** |
| negative | `_zero_out(cond)` | **`None`** |
| 采样入口 | `comfy.sample.sample()` | **`sw_h3_basic_sample.basic_sample()`** |
| 采样 `cond_scale` | 用户传入 | **固定 1.0** |
| 前向通道数 | 2 | **1** |
| `scheduler` 默认 | `beta` | `beta`（相同） |
| 段拼接 / anchor / sigma_shift | 相同 | 相同 |

除了这几点，**其他逻辑与原版逐行相同** —— 这样 A/B 对比才有意义。

### ★★ 关键：为什么不能直接 `comfy.sample.sample(negative=None, cfg=1.0)`

这是本版踩过的**真坑**（线上 `TypeError: 'NoneType' object is not iterable`）。
`comfy/sample.py:sample()` 内部构造的是**硬编码的 CFGGuider**：

```python
# comfy/sample.py:78
sampler = comfy.samplers.KSampler(...)
return sampler.sample(noise, positive, negative, cfg=cfg, ...)   # ← negative 照样往下传

# comfy/samplers.py:1449
cfg_guider = CFGGuider(model)
cfg_guider.set_conds(positive, negative)          # ← 无条件塞 negative

# comfy/samplers.py:1195
def set_conds(self, positive, negative):
    self.inner_set_conds({"positive": positive, "negative": negative})

# comfy/samplers.py:1203 → comfy/sampler_helpers.py:72
if self.model_patcher.is_dynamic() and cond_has_hooks(conds[k]):
                                                  # ↑ cond=None → `for c in cond` → TypeError
```

**cfg=1.0 也救不了**：因为它崩在 `set_conds` 的条件转换里，还没走到 cfg 计算。

官方 H3 链路压根不调 `comfy.sample.sample`，它走
`Guider_Basic`（`comfy_extras/nodes_custom_sampler.py:797`）：

```python
class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):                  # ← 签名里根本没有 negative
        self.inner_set_conds({"positive": positive})
```

`original_conds` 因此**没有 `"negative"` 这个键**，
`predict_noise` 遍历 `original_conds` 构造 conds 列表时只有1 路。

本版的 `sw_h3_basic_sample.py` 就是复刻 `comfy/sample.py:sample()` 的全部逻辑
（KSampler 建 sigmas → guider.sample → intermediate_device cast），
**唯一区别是把 CFGGuider 换成 Guider_Basic**。取不到官方类时本地fallback 一份。


###节点类名映射

| 原版 | 本版 |
|---|---|
| `SW_H3MultiPrompt` | `SW_H3MultiPrompt_NCG` |
| `SW_H3Ultra` | `SW_H3Ultra_NCG` |
| `SW_H3_SegPlan` | `SW_H3_SegPlan_NCG` |
| `SW_H3_SegBridge` | `SW_H3_SegBridge_NCG` |
| `SW_H3_SegConcat` | `SW_H3_SegConcat_NCG` |

> 两套插件**可以同时安装**（类名不冲突），节点显示名带`【无CFG·NCG】` 标识，
> 方便在搜索面板区分。

---

## 五、现成工作流

已按A/B 对照要求做好一份可直接拖进 ComfyUI 的工作流：

| 文件 | 说明 |
|---|---|
| `24SW_H3_多段一体机_NCG无CFG_4段_3图_竖版.json` | 4 段 / 3 参考图 / 768×1344 竖版 / 26 秒 |

**它就是 24 号原版工作流的逐字节对照版**，只动了一体机节点本身：

- `type`：`SW_H3MultiPrompt` → `SW_H3MultiPrompt_NCG`
- 删掉 `cfg` 输入口和 `widgets_values` 里那一格`0.8`
- 其余 16 个节点、17 条连线、4 段提示词、3 张参考图、
  `seed=1234567890`、`steps=6`、`euler` / `beta`、`裁剪到秒数=26` 全部逐项相同

所以你只要把 24 号和这份各跑一次，**唯一变量就是 CFG 路径**。

已离线校验通过（64/64）：官方 `validate_prompt` 通过、整图端到端执行无错误、
4 次采样全部 `cfg=1.0` 且 `negative=None`、成片 617 帧 / 25.71 秒、音视频同步。

> 装这份工作流前先确认 ComfyUI 已重启、`SW_H3MultiPrompt_NCG` 能在节点搜索里搜到。
> 若搜不到，说明插件没加载成功（看启动日志有没有报错）。

---

## 五★、v7 新增：一条总时长搞定 30 秒长视频

这一版把「填几个手动参数」改成**填一个总秒数**，段数 / 每段帧数 / 裁剪点全部自动规划。

### 1. 固定分段方案（目标总时长 = 唯一驱动）

| 目标总时长 | 段数 | 每段 |
|---|---|---|
| ≤ 15 秒 | 1 | 10.5s |
| 16 ~ 20.5 秒 | 2 | 10.5 + 10.5 |
| 21 ~ 31 秒 | 3 | **10.5 + 10.5 + 10** |
| > 31 秒 | 4 ~ 6 | 继续扩展，取「能覆盖目标」的最小段数 |

- 段间保留 **5 帧锚定重叠**（段 2 起点 = 段 1 末帧 - 5），拼完正好接上
- 末段**超出裁掉、不缩短** —— 保证末段对白与收尾完整
- 31 秒实测：段1 `0→10.833s` / 段2 `10.625→21.458s` / 段3 `21.250→31.375s`，
  自然成片 753 帧，按目标裁到 **744 帧 = 31.000s**

### 2. 按段切分参考音频（对口型 / 歌词漂移的根因修复）

新开关 **`按段切分参考音频`**（默认开）。

| | 旧行为 | 新行为 |
|---|---|---|
| 参考音频怎么编 | 整首编成**一个** `ref_audio` block 注入每一段 | 按每段在成片里的**全局时间窗**切出该窗音频 |
| 模型知道什么 | 「这首歌是这个人唱的」 | 「我现在生成的这 10 秒 = 原曲的哪 10 秒」 |
| 结果 | 第 2/3 段口型漂、歌词重复 | 逐帧音素贴合 |

实现：把该段音频编成 `audio_latent`，带 `resolved_frame_index=0` 放进该段的
`minimax_keyframes`。PackedLayout 对 keyframes 走
`cond_t = cursor + FRAME_RESCALE * resolved_frame_index`，
生成贴在 **target 时间轴**上的 `cond_audio` 段，模型由此获得逐帧对齐。

> 长视频对口型飘、歌词重复，基本都是这条没做导致的。

### 3. 前端：提示词口按段数动态显隐

`segment_count` 填 3 时，`prompt_4` / `prompt_5` **自动隐藏**（不再占地方）；
填 `auto` 时按已填的提示词口渐进显示下一个。

> 浏览器看不到效果先硬刷新（Ctrl+F5）。
> 扩展 JS 必须在 `web/` **顶层** —— 放 `web/js/` 子目录会让
> `import "../../scripts/app.js"` 解析成 `/extensions/scripts/app.js` → 404 → 扩展静默不加载。

---

## 六、安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/weiwei141319/ComfyUI-SW-H3Segments-NCG.git
```

重启 ComfyUI，节点搜索 `NCG` 即可看到。

> 视频分段切割/合并是**另一个独立包** `ComfyUI-SW-VideoSplit`（与本包无依赖），需要时单独安装。

---

## 七、A/B 怎么比

想公平对比，两个节点用**完全相同**的：

- 参考图 / 参考视频 / 参考音频
- 各段提示词、`segment_seconds`、`anchor_frames`
- `width` / `height`（分辨率）
- `seed` / `递增种子`（**务必固定种子**，否则噪声不同没法比）
- `steps` / `sampler_name` / `scheduler`（都设 `euler` / `beta`）
- `内置sigma_shift` / `shift_video` / `shift_audio`
- `内部解码` / `裁剪到秒数`

差异只有一处：**原版走 CFG 路径，本版不走**。

预期观察点：

1. **耗时**：本版应快接近一倍（采样阶段）
2. **画面**：本版不会有 `cond×0.8` 的整体衰减，色彩应更饱满
3. **泛紫**：若两者都不紫，说明 `beta` 已经是主因（已验证）；若原版仍紫、本版不紫，则 CFG 路径是叠加因素

---

## 八、许可证

保留所有权利。未经作者书面许可，不得复制、分发或用于商业用途。
