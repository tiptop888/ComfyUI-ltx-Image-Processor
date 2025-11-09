"""
ComfyUI ltx 图片处理插件 - 初始化文件
包含带有ltx字样的节点，实现加载图片和保存图片名称一致的功能
"""

from .ltx_image_filename_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']