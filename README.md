# ComfyUI ltx Image Processor

一个强大的ComfyUI图像处理插件，支持批量加载、处理和保存图像，同时保持原始文件名和格式。

## 功能特性

- ✅ **批量图像处理**：支持批量加载和处理不同尺寸的图像
- ✅ **保持原始文件名**：处理后的图像保持原始文件名和扩展名
- ✅ **多种尺寸调整模式**：灵活处理不同尺寸图像
- ✅ **保持宽高比**：所有操作都会保持原始图像比例，避免变形
- ✅ **智能填充**：使用最少的填充区域来适配图像尺寸

## 安装

1. 将此仓库克隆或下载到你的ComfyUI的`custom_nodes`目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-ltx-Image-Processor.git
```

2. 重启ComfyUI

## 节点列表

### LTX Load Image with Filename
- 单张图像加载，返回图像数据、遮罩、文件名和扩展名

### LTX Load Images from Directory with Filenames
- **核心功能节点**，支持多种尺寸调整模式：
  - `none`：保持原始尺寸（仅适用于相同尺寸图片）
  - `resize`：拉伸至目标尺寸（可能导致变形）
  - `smart_resize_crop`：智能缩放（保持宽高比，居中填充）
  - `common_size_fit`：使用最常见尺寸作为画布，大的等比缩放，小的居中填充

### LTX Save Image with Original Filename
- 使用原始文件名和格式保存图像

### LTX Save Image with Original Info
- 使用加载时的扩展名信息保存图像

### LTX Process Batch with Original Names
- 批量处理保持原始名称和格式

### LTX Load And Process Single Image
- 用于循环处理，实现加载一张、处理一张、保存一张

## 使用方法

### 批量处理不同尺寸图像

1. 使用 `LTX Load Images from Directory with Filenames` 节点
2. 选择 `common_size_fit` 模式
3. 节点会自动：
   - 统计目录中所有图像的尺寸
   - 选择出现频率最高的尺寸作为目标尺寸
   - 将大的图像等比缩放至目标尺寸内
   - 将小的图像居中放置并用背景色填充
   - 所有操作都保持原始宽高比

### 智能尺寸适配

- 使用 `smart_resize_crop` 模式可以将所有图像智能适配到指定尺寸
- 图像始终保持宽高比，通过缩放和居中填充来适应
- 不会裁剪任何内容，避免丢失图像信息

## 特点

- **无图像变形**：所有尺寸调整操作都严格保持宽高比
- **最小填充**：智能算法尽量减少不必要的填充区域
- **原始格式保持**：保存时自动使用原始文件格式
- **批量处理**：支持批量处理大量图像
- **ComfyUI兼容**：无缝集成到ComfyUI工作流中

## 配置选项

- `directory`：输入图像目录路径
- `start_index`：开始处理的文件索引
- `image_load_cap`：最大加载图像数量
- `resize_mode`：尺寸调整模式
- `target_width/target_height`：目标尺寸
- `fill_color`：填充颜色（黑色/白色/灰色）

## 注意事项

1. 当使用 `none` 模式时，如果目录中有不同尺寸的图像，将会报错
2. `common_size_fit` 模式会自动选择最常见的图像尺寸作为基准
3. 所有尺寸调整操作都保持原始宽高比，防止图像变形
4. 建议使用 `smart_resize_crop` 或 `common_size_fit` 模式处理不同尺寸的图像

## 支持的格式

- PNG, JPG, JPEG
- BMP, TIFF, WEBP
- GIF

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进此插件。

## 作者

程序大师