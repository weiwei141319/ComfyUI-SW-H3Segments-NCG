# -*- coding: utf-8 -*-
"""NCG 版的无 CFG 采样封装。

★ 为什么不能直接用 `comfy.sample.sample(..., negative=None, cfg=1.0)`：

`comfy/sample.py:sample()` 内部构造的是**硬编码的 CFGGuider**：

    # comfy/sample.py:78
    sampler = comfy.samplers.KSampler(...)
    return sampler.sample(noise, positive, negative, cfg=cfg, ...)

    # comfy/samplers.py:1449
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)     # ← 无条件传 negative

    # comfy/samplers.py:1195
    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    # comfy/samplers.py:1203 → sampler_helpers.py:72
    cond_has_hooks(conds[k])  →  for c in cond    # ← cond=None 直接 TypeError

所以哪怕 cfg=1.0，只要 negative 真是 None，`comfy.sample.sample` 就会炸：
`TypeError: 'NoneType' object is not iterable`。

官方 H3 链路不走 `comfy.sample.sample`，它走
`Guider_Basic`（`comfy_extras/nodes_custom_sampler.py:797`）：

    class Guider_Basic(comfy.samplers.CFGGuider):
        def set_conds(self, positive):                  # ← 签名只有 positive
            self.inner_set_conds({"positive": positive})

`original_conds` 里因此**没有 "negative" 这个键**，
`CFGGuider.predict_noise` → `sampling_function` 只拿到 1 路条件，
`calc_cond_batch` 的 `if cond is not None` 也不会去碰它。

本模块提供 `basic_sample()`，复刻 `comfy/sample.py:sample()` 的全部逻辑，
唯一区别是把 CFGGuider 换成 Guider_Basic。
"""

from __future__ import annotations

import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy_extras.nodes_custom_sampler as _official_guiders


def _get_basic_guider_cls():
    """取官方 Guider_Basic；取不到就本地复刻一份（避免版本漂移）。"""
    cls = getattr(_official_guiders, "Guider_Basic", None)
    if cls is not None:
        return cls

    class _GuiderBasicFallback(comfy.samplers.CFGGuider):
        def set_conds(self, positive):          # noqa: D401
            self.inner_set_conds({"positive": positive})

    return _GuiderBasicFallback


def basic_sample(model, noise, steps, sampler_name, scheduler,
                 positive, latent_image, denoise=1.0, disable_noise=False,
                 start_step=None, last_step=None, force_full_denoise=False,
                 noise_mask=None, sigmas=None, callback=None,
                 disable_pbar=False, seed=None):
    """等价 `comfy.sample.sample(..., negative=None, cfg=1.0)`，但走 Guider_Basic。

    参数与官方 `comfy/sample.py:sample()` 完全一致（只是去掉了 negative / cfg）。
    返回值与官方一致：已在 intermediate_device 上cast 过的 latent。
    """
    sampler = comfy.samplers.KSampler(
        model, steps=steps, device=model.load_device,
        sampler=sampler_name, scheduler=scheduler,
        denoise=denoise, model_options=model.model_options,
    )

    if sigmas is None:
        sigmas = sampler.sigmas
    if last_step is not None and last_step < (len(sigmas) - 1):
        sigmas = sigmas[:last_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0
    if start_step is not None:
        if start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
        else:
            return (latent_image if latent_image is not None
                    else noise.new_zeros(noise.shape))

    # ★ 唯一的改动：Guider_Basic 而非 CFGGuider
    guider = _get_basic_guider_cls()(model)
    guider.set_conds(positive)
    guider.set_cfg(1.0)

    samples = guider.sample(
        noise, latent_image, comfy.samplers.sampler_object(sampler.sampler),
        sigmas, denoise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=seed,
    )
    return samples.to(device=comfy.model_management.intermediate_device(),
                      dtype=comfy.model_management.intermediate_dtype())


__all__ = ["basic_sample"]
