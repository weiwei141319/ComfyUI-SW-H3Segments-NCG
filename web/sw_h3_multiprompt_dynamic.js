/**
 * SW_H3MultiPrompt_NCG —— 提示词口动态显隐（v2 加固版）
 *
 * 规则：
 *   · segment_count = "1".."5"（手动锁定）→ 只显示前 N 个口，其余隐藏
 *     （与后端「锁定段数、忽略空口」语义一致；被藏口里的旧文本仍序列化保留）。
 *   · segment_count = "auto" → 渐进显示：已填非空的口 + 下一个空口。
 *
 * 验证方法：F12 控制台应看到
 *   [SW-H3Segments] 动态显隐扩展已加载        ← 页面加载即有
 *   [SW-H3Segments] SW_H3MultiPrompt_NCG: 显示前 N 个提示词口
 * 第一行都没有 = 浏览器还在用缓存的旧页面 → Ctrl+F5 硬刷新。
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "SW_H3MultiPrompt_NCG";
const PROMPTS = ["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"];
const LOG = (...a) => console.log("[SW-H3Segments]", ...a);

LOG("动态显隐扩展已加载");

function chainCallback(object, property, callback) {
  if (object == undefined) {
    console.error("[SW-H3Segments] 尝试给不存在的属性挂回调:", property);
    return;
  }
  if (property in object) {
    const callbackOrig = object[property];
    object[property] = function () {
      const r = callbackOrig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
    };
  } else {
    object[property] = callback;
  }
}

function chainWidgetCallback(widget, cb) {
  const orig = widget.callback;
  widget.callback = function () {
    const r = orig?.apply(this, arguments);
    cb();
    return r;
  };
}

function getWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function isFilled(node, name) {
  return String(getWidget(node, name)?.value ?? "").trim() !== "";
}

/** 应该可见的 prompt 口数量 */
function visibleCount(node) {
  const seg = getWidget(node, "segment_count");
  if (!seg) return PROMPTS.length;

  const mode = String(seg.value);
  if (mode !== "auto") {
    const k = parseInt(mode, 10);
    if (Number.isFinite(k)) return Math.min(PROMPTS.length, Math.max(1, k));
    return PROMPTS.length;
  }

  let n = 0;
  while (n < PROMPTS.length && isFilled(node, PROMPTS[n])) n++;
  return Math.min(PROMPTS.length, n + 1);
}

function applyVisibility(node, quiet) {
  if (!node?.widgets) return;
  const maxShow = visibleCount(node);
  let changed = false;
  PROMPTS.forEach((name, i) => {
    const w = getWidget(node, name);
    if (!w) return;
    const hide = i >= maxShow;
    if (!!w.hidden !== hide) {
      w.hidden = hide;
      // 双保险：部分渲染路径读 options.hidden
      if (w.options) w.options.hidden = hide;
      // 旧画布渲染兜底：隐藏时不占高度
      w.computeSize = hide ? () => [0, -4] : undefined;
      changed = true;
    }
  });
  if (changed && !quiet) {
    LOG(`${node.__sw_log_name ?? NODE_NAME}: 显示前 ${maxShow} 个提示词口`);
    try {
      const h = node.computeSize(node.size[0])[1];
      node.setSize([node.size[0], h]);
    } catch (e) { /* computeSize 不可用时只刷画布 */ }
    node.onResize?.(node.size);
    node.setDirtyCanvas?.(true, true);
    app.canvas?.setDirtyCanvas?.(true, true);
    // 触发前端响应式重布局（Vue widget 视图层）
    app.graph?.change?.();
  }
}

app.registerExtension({
  name: "SW.H3MultiPrompt.DynamicPromptSlots",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    LOG("注册节点钩子:", NODE_NAME);

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      applyVisibility(node, true); // 静默应用，不打日志

      const seg = getWidget(node, "segment_count");
      if (seg) chainWidgetCallback(seg, () => applyVisibility(node));

      // prompt_N 填写变化 → 影响 auto 模式下 prompt_{N+1} 的显隐
      for (let i = 0; i < PROMPTS.length - 1; i++) {
        const p = getWidget(node, PROMPTS[i]);
        if (p) chainWidgetCallback(p, () => applyVisibility(node));
      }
    });

    // 节点加入画布后再应用一次（此时 widget/视图状态最完整）
    chainCallback(nodeType.prototype, "onAdded", function (graph) {
      this.__sw_log_name = NODE_NAME + "#" + this.id;
      applyVisibility(this, true);
      return graph;
    });

    // 加载工作流时按恢复后的值重算
    chainCallback(nodeType.prototype, "onConfigure", function () {
      applyVisibility(this, true);
    });
  },
});
