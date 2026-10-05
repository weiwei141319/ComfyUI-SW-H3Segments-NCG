from .sw_h3_segments import (
    NODE_CLASS_MAPPINGS as _SEG_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _SEG_NAMES,
)
from .sw_h3_multiprompt import (
    NODE_CLASS_MAPPINGS as _MULTI_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _MULTI_NAMES,
)

from .sw_h3_ultra import (
    NODE_CLASS_MAPPINGS as _ULTRA_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _ULTRA_NAMES,
)
NODE_CLASS_MAPPINGS = {
    **_SEG_CLASS, **_MULTI_CLASS, **_ULTRA_CLASS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_SEG_NAMES, **_MULTI_NAMES, **_ULTRA_NAMES,
}

# 注：视频拆分已独立成专门的插件包 ComfyUI-SW-VideoSplit（custom_nodes 同级目录），
#     不在本包里重复提供。
# 前端扩展目录：prompt_1..5 按段数动态显隐（见 web/sw_h3_multiprompt_dynamic.js）
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
