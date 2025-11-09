"""
ComfyUI ltx 图片处理插件
作者: 程序大师
功能: 实现加载一张、处理一张、保存一张，且保存文件名与加载文件名完全一致
节点名称包含ltx字样，不修改原有节点，只添加新节点
自动识别图片格式，保存时使用原始格式
保持图片原尺寸，不进行尺寸调整
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional
import folder_paths
from PIL import Image, ImageOps
from typing import Dict, List, Tuple, Any


def smart_resize_and_crop(img, target_width, target_height, fill_color=(0, 0, 0)):
    """
    智能调整图像尺寸，始终保持宽高比：
    - 如果图像小于目标尺寸：按比例缩放并填充空白（居中）
    - 如果图像大于目标尺寸：按比例缩放至目标尺寸内（居中），避免裁剪
    """
    original_width, original_height = img.size
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    # 计算缩放后的尺寸，保持宽高比
    if original_ratio > target_ratio:
        # 图片更宽，基于宽度缩放
        new_width = target_width
        new_height = int(original_height * (target_width / original_width))
    else:
        # 图片更高，基于高度缩放
        new_height = target_height
        new_width = int(original_width * (target_height / original_height))

    # 缩放图片
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建目标尺寸的新图片，填充背景
    new_img = Image.new("RGB", (target_width, target_height), fill_color)
    # 计算居中位置
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    # 将调整后的图片粘贴到中心
    new_img.paste(img_resized, (x, y))
    
    return new_img


class LTX_LoadImageWithFilename:
    """
    ltx加载图片并返回文件名 - 不修改原节点，仅新增
    自动识别图片格式，保持图片原尺寸
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            }
        }

    CATEGORY = "ltx/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "extension")
    FUNCTION = "load_image"

    def load_image(self, image):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        img = Image.open(image_path)

        # 获取文件扩展名
        _, ext = os.path.splitext(image)
        ext = ext.lower().lstrip('.')

        # 处理图片
        output_images = []
        output_masks = []

        # 转换为RGB
        for i in range(1):  # 单张图片
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            output_images.append(img_tensor)

            # 处理mask
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((img.height, img.width), dtype=torch.float32)  # 使用原图尺寸
            output_masks.append(mask.unsqueeze(0))

        # 合并图片和mask
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)

        # 返回图片、mask、文件名和扩展名
        return (output_image, output_mask, image, ext)


class LTX_LoadImagesFromDirWithFilename:
    """
    ltx从目录加载图片并返回文件名列表 - 不修改原节点，仅新增
    自动识别图片格式，保持图片原尺寸
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "./input", "placeholder": "图片目录路径"}),
                "start_index": ("INT", {"default": 0, "min": 0}),
                "image_load_cap": ("INT", {"default": 0, "min": 0}),
                "resize_mode": (["none", "resize", "smart_resize_crop", "common_size_fit"], {"default": "none", "label": "尺寸调整模式", 
                    "tooltip": "none=保持原始尺寸(仅适用于相同尺寸图片), resize=拉伸至目标尺寸, smart_resize_crop=智能缩放(保持宽高比，居中填充), common_size_fit=使用最常见尺寸作为画布"}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "fill_color": (["black", "white", "gray"], {"default": "black", "label": "填充颜色", 
                    "tooltip": "当使用智能缩放时，用于填充空白区域的颜色"})
            }
        }

    CATEGORY = "ltx/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "masks", "filenames", "count", "extensions")
    FUNCTION = "load_images_from_dir"

    def load_images_from_dir(self, directory, start_index, image_load_cap, resize_mode, target_width, target_height, fill_color):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        # 支持的格式
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif')

        # 获取图片文件
        image_files = []
        for file in os.listdir(directory):
            if file.lower().endswith(supported_formats):
                image_files.append(file)

        image_files = sorted(image_files)

        # 应用索引和限制
        if image_load_cap > 0:
            image_files = image_files[start_index:start_index + image_load_cap]
        else:
            image_files = image_files[start_index:]

        if not image_files:
            raise ValueError(f"目录中没有找到图片文件: {directory}")

        # 如果是common_size_fit模式，需要先确定目标尺寸
        if resize_mode == "common_size_fit":
            # 统计尺寸出现频率
            size_count = {}
            for file in image_files:
                image_path = os.path.join(directory, file)
                with Image.open(image_path) as img:
                    size = img.size
                    size_count[size] = size_count.get(size, 0) + 1
            
            # 选择最常见的尺寸作为目标尺寸
            if size_count:
                target_size = max(size_count.items(), key=lambda x: x[1])[0]
                actual_target_width, actual_target_height = target_size
            else:
                # 如果无法确定，使用默认值
                actual_target_width, actual_target_height = target_width, target_height
        else:
            actual_target_width, actual_target_height = target_width, target_height

        images = []
        masks = []
        extensions = []

        # 将字符串填充色转换为RGB值
        fill_color_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128)
        }
        fill_color_rgb = fill_color_map.get(fill_color, (0, 0, 0))

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            img = Image.open(image_path)

            # 获取扩展名
            _, ext = os.path.splitext(image_file)
            ext = ext.lower().lstrip('.')
            extensions.append(ext)

            # 根据选择的模式处理图片
            processed_img = img
            processed_mask = None
            
            if resize_mode == "resize":
                # 传统模式：直接调整到目标尺寸（可能导致拉伸变形）
                processed_img = img.resize((actual_target_width, actual_target_height), Image.Resampling.LANCZOS)
                
                # 处理mask (调整后尺寸)
                if 'A' in img.getbands():
                    mask_resized = img.getchannel('A').resize((actual_target_width, actual_target_height), Image.Resampling.LANCZOS)
                    mask_array = np.array(mask_resized).astype(np.float32) / 255.0
                    processed_mask = 1. - torch.from_numpy(mask_array)
                else:
                    processed_mask = torch.zeros((actual_target_height, actual_target_width), dtype=torch.float32)
            elif resize_mode == "smart_resize_crop":
                # 智能模式：小于目标尺寸则缩放填充，大于目标尺寸则居中裁剪
                processed_img = smart_resize_and_crop(img, actual_target_width, actual_target_height, fill_color_rgb)
                
                # 对于mask的处理，我们同样需要应用相同的变换
                if 'A' in img.getbands():
                    # 对透明通道也应用相同处理
                    mask_img = img.getchannel('A')
                    # 为透明度通道使用白色填充以保持不透明区域的正确性
                    processed_mask_img = smart_resize_and_crop(mask_img.convert("L").convert("RGB"), actual_target_width, actual_target_height, (255, 255, 255))
                    processed_mask_np = np.array(processed_mask_img.convert('L')).astype(np.float32) / 255.0
                    processed_mask = 1. - torch.from_numpy(processed_mask_np)
                else:
                    processed_mask = torch.zeros((actual_target_height, actual_target_width), dtype=torch.float32)
            elif resize_mode == "common_size_fit":
                # 使用最常见尺寸作为画布，大的等比缩放，小的居中填充
                original_width, original_height = img.size
                target_ratio = actual_target_width / actual_target_height
                original_ratio = original_width / original_height

                if original_width > actual_target_width or original_height > actual_target_height:
                    # 图片比目标尺寸大，进行等比缩放
                    if original_ratio > target_ratio:
                        # 图片更宽，基于宽度缩放
                        new_width = actual_target_width
                        new_height = int(original_height * (actual_target_width / original_width))
                    else:
                        # 图片更高，基于高度缩放
                        new_height = actual_target_height
                        new_width = int(original_width * (actual_target_height / original_height))

                    # 缩放图片
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # 创建目标尺寸的画布，居中放置缩放后的图片
                    processed_img = Image.new("RGB", (actual_target_width, actual_target_height), fill_color_rgb)
                    x = (actual_target_width - new_width) // 2
                    y = (actual_target_height - new_height) // 2
                    processed_img.paste(img_resized, (x, y))
                elif original_width < actual_target_width or original_height < actual_target_height:
                    # 图片比目标尺寸小，居中填充
                    processed_img = Image.new("RGB", (actual_target_width, actual_target_height), fill_color_rgb)
                    x = (actual_target_width - original_width) // 2
                    y = (actual_target_height - original_height) // 2
                    processed_img.paste(img, (x, y))
                else:
                    # 图片与目标尺寸相同，直接使用
                    processed_img = img

                # 处理mask
                if 'A' in img.getbands():
                    # 对透明通道也应用相同的处理
                    mask_img = img.getchannel('A')
                    
                    if original_width > actual_target_width or original_height > actual_target_height:
                        # 对mask进行相同的缩放和居中处理
                        if original_ratio > target_ratio:
                            new_mask_width = actual_target_width
                            new_mask_height = int(original_height * (actual_target_width / original_width))
                        else:
                            new_mask_height = actual_target_height
                            new_mask_width = int(original_width * (actual_target_height / original_height))

                        mask_resized = mask_img.resize((new_mask_width, new_mask_height), Image.Resampling.LANCZOS)
                        processed_mask_img = Image.new("L", (actual_target_width, actual_target_height), 0)
                        x = (actual_target_width - new_mask_width) // 2
                        y = (actual_target_height - new_mask_height) // 2
                        processed_mask_img.paste(mask_resized, (x, y))
                    elif original_width < actual_target_width or original_height < actual_target_height:
                        processed_mask_img = Image.new("L", (actual_target_width, actual_target_height), 0)
                        x = (actual_target_width - original_width) // 2
                        y = (actual_target_height - original_height) // 2
                        processed_mask_img.paste(mask_img, (x, y))
                    else:
                        processed_mask_img = mask_img
                    
                    mask_array = np.array(processed_mask_img).astype(np.float32) / 255.0
                    processed_mask = 1. - torch.from_numpy(mask_array)
                else:
                    processed_mask = torch.zeros((actual_target_height, actual_target_width), dtype=torch.float32)
            else:
                # "none" 模式：保持原始尺寸
                if 'A' in img.getbands():
                    mask_array = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    processed_mask = 1. - torch.from_numpy(mask_array)
                else:
                    processed_mask = torch.zeros((img.height, img.width), dtype=torch.float32)

            # 转换图片为RGB并转换为tensor
            if resize_mode in ["resize", "smart_resize_crop", "common_size_fit"]:
                # 已经调整到统一尺寸，直接转换
                img_rgb = processed_img.convert("RGB")
                img_array = np.array(img_rgb).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,]
            else:
                # 保持原始尺寸
                img_rgb = processed_img.convert("RGB")
                img_array = np.array(img_rgb).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,]

            images.append(img_tensor)
            masks.append(processed_mask.unsqueeze(0))

        # 合并所有图片和mask
        if images:
            # 检查所有图片是否具有相同尺寸后再拼接
            if len(images) > 1:
                first_shape = images[0].shape[1:]  # 获取第一个图像的尺寸 (H, W, C)
                all_same_size = all(img.shape[1:] == first_shape for img in images)

                if all_same_size:
                    # 所有图片尺寸相同，可以安全拼接
                    output_images = torch.cat(images, dim=0)
                    output_masks = torch.cat(masks, dim=0)
                else:
                    # 如果在"none"模式下图片尺寸不同，提示用户
                    if resize_mode == "none":
                        # 在"none"模式下，如果尺寸不同，给出错误提示
                        raise ValueError(f"在'none'模式下检测到不同尺寸的图片: {[img.shape[1:3] for img in images]}。请确保'none'模式下所有图片尺寸相同，或使用其他模式。")
                    else:
                        # 在其他模式下，所有图片应该已经是相同尺寸
                        output_images = torch.cat(images, dim=0)
                        output_masks = torch.cat(masks, dim=0)
            else:
                # 只有一张图片，直接使用
                output_images = images[0]
                output_masks = masks[0]
        else:
            # 如果没有图片，返回空tensor
            output_images = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            output_masks = torch.zeros((1, 1, 1), dtype=torch.float32)

        # 返回图片、mask、文件名列表、数量和扩展名列表
        return (output_images, output_masks, json.dumps(image_files), len(image_files), json.dumps(extensions))





class LTX_SaveImageWithOriginalFilename:
    """
    ltx使用原始文件名和格式保存图片 - 不修改原节点，仅新增
    自动使用原始格式保存，无需选择，保持原尺寸
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_filename": ("STRING", {"default": "image.png"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ltx/Image"

    def save_images(self, images, original_filename, prompt=None, extra_pnginfo=None):
        output_path = self.output_dir
        os.makedirs(output_path, exist_ok=True)

        # 检查 original_filename 是否是 JSON 格式的字符串数组
        # 如果是 JSON 数组，从中获取第一个文件名用于扩展名
        try:
            parsed_filenames = json.loads(original_filename)
            if isinstance(parsed_filenames, list) and len(parsed_filenames) > 0:
                # 如果是文件名列表，使用第一个文件名的扩展名，并在保存时使用相应的索引文件名
                first_filename = parsed_filenames[0]
                name, orig_ext = os.path.splitext(first_filename)
                orig_ext = orig_ext.lower().lstrip('.')  # 获取原始扩展名
            else:
                # 如果不是数组或数组为空，按原始逻辑处理
                name, orig_ext = os.path.splitext(original_filename)
                orig_ext = orig_ext.lower().lstrip('.')  # 获取原始扩展名
        except json.JSONDecodeError:
            # 如果不是有效的 JSON，按原始逻辑处理
            name, orig_ext = os.path.splitext(original_filename)
            orig_ext = orig_ext.lower().lstrip('.')  # 获取原始扩展名

        results = []

        for idx, img_tensor in enumerate(images):
            # 转换tensor到numpy
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 构造保存文件名，使用原始扩展名
            # 如果 original_filename 是 JSON 数组，尝试使用对应的文件名
            try:
                parsed_filenames = json.loads(original_filename)
                if isinstance(parsed_filenames, list) and idx < len(parsed_filenames):
                    # 使用数组中的相应文件名
                    current_filename = parsed_filenames[idx]
                    name, _ = os.path.splitext(current_filename)
                    save_filename = current_filename
                else:
                    # 如果数组中没有对应索引的文件名，使用原始逻辑
                    if len(images) > 1:
                        save_filename = f"{name}_{idx}.{orig_ext}"
                    else:
                        save_filename = f"{name}.{orig_ext}"
            except json.JSONDecodeError:
                # 如果不是有效的 JSON，按原始逻辑处理
                if len(images) > 1:
                    save_filename = f"{name}_{idx}.{orig_ext}"
                else:
                    save_filename = f"{name}.{orig_ext}"

            file_path = os.path.join(output_path, save_filename)

            # 根据原始格式保存，不压缩，保持原尺寸
            # 使用PIL支持的格式确保文件扩展名正确
            if orig_ext in ["jpg", "jpeg"]:
                img.save(file_path, format="JPEG", quality=100, subsampling=0)  # 无损保存
            elif orig_ext == "png":
                img.save(file_path, format="PNG", compress_level=0)  # 无压缩保存
            elif orig_ext in ["tiff", "tif"]:
                img.save(file_path, format="TIFF", compression=None)
            elif orig_ext == "webp":
                img.save(file_path, format="WEBP", quality=100, method=6)  # 高质量保存
            elif orig_ext == "bmp":
                img.save(file_path, format="BMP")  # BMP格式
            elif orig_ext == "gif":
                img.save(file_path, format="GIF")  # GIF格式
            else:
                # 对于未知格式，使用PNG作为默认格式
                # 重新构造文件路径使用PNG扩展名
                fallback_filename = f"{os.path.splitext(save_filename)[0]}.png"
                fallback_file_path = os.path.join(output_path, fallback_filename)
                img.save(fallback_file_path, format="PNG")
                # 更新results以反映实际保存的文件名
                save_filename = fallback_filename

            results.append({
                "filename": save_filename,
                "subfolder": "",
                "type": self.type
            })

        return {"ui": {"images": results}}


class LTX_SaveImageWithOriginalInfo:
    """
    ltx使用原始文件名和扩展名保存图片 - 更高级的版本
    使用从加载节点传来的扩展名信息，保持原尺寸
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_filename": ("STRING", {"default": "image.png"}),
                "original_extension": ("STRING", {"default": "png"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ltx/Image"

    def save_images(self, images, original_filename, original_extension, prompt=None, extra_pnginfo=None):
        output_path = self.output_dir
        os.makedirs(output_path, exist_ok=True)

        # 解析原始文件名
        name, _ = os.path.splitext(original_filename)

        results = []

        for idx, img_tensor in enumerate(images):
            # 转换tensor到numpy
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 构造保存文件名，使用传入的扩展名
            if len(images) > 1:
                save_filename = f"{name}_{idx}.{original_extension}"
            else:
                save_filename = f"{name}.{original_extension}"

            file_path = os.path.join(output_path, save_filename)

            # 根据原始格式保存，不压缩，保持原尺寸
            # 使用PIL支持的格式确保文件扩展名正确
            if original_extension in ["jpg", "jpeg"]:
                img.save(file_path, format="JPEG", quality=100, subsampling=0)  # 无损保存
            elif original_extension == "png":
                img.save(file_path, format="PNG", compress_level=0)  # 无压缩保存
            elif original_extension in ["tiff", "tif"]:
                img.save(file_path, format="TIFF", compression=None)
            elif original_extension == "webp":
                img.save(file_path, format="WEBP", quality=100, method=6)  # 高质量保存
            elif original_extension == "bmp":
                img.save(file_path, format="BMP")  # BMP格式
            elif original_extension == "gif":
                img.save(file_path, format="GIF")  # GIF格式
            else:
                # 对于未知格式，使用PNG作为默认格式
                # 重新构造文件路径使用PNG扩展名
                fallback_filename = f"{os.path.splitext(save_filename)[0]}.png"
                fallback_file_path = os.path.join(output_path, fallback_filename)
                img.save(fallback_file_path, format="PNG")
                # 更新results以反映实际保存的文件名
                save_filename = fallback_filename

            results.append({
                "filename": save_filename,
                "subfolder": "",
                "type": self.type
            })

        return {"ui": {"images": results}}


class LTX_ProcessBatchWithOriginalNames:
    """
    ltx批量处理保持原始名称和格式 - 不修改原节点，仅新增
    保持原始文件名、格式和尺寸
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {"default": "./input", "placeholder": "输入目录"}),
                "output_directory": ("STRING", {"default": "", "placeholder": "输出目录，留空使用默认"}),
                "start_index": ("INT", {"default": 0, "min": 0}),
                "image_load_cap": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "count")
    FUNCTION = "process_batch"
    CATEGORY = "ltx/Image"

    def process_batch(self, input_directory, output_directory, start_index, image_load_cap):
        # 加载图片
        loader = LTX_LoadImagesFromDirWithFilename()
        images, masks, filenames_str, count, extensions_str = loader.load_images_from_dir(
            input_directory, start_index, image_load_cap, "none", 512, 512, "black"
        )

        # 确定输出目录
        if output_directory and output_directory.strip():
            output_path = output_directory
        else:
            output_path = folder_paths.get_output_directory()

        os.makedirs(output_path, exist_ok=True)

        # 解析文件名列表和扩展名列表
        filenames = json.loads(filenames_str)
        extensions = json.loads(extensions_str)

        # 模拟处理（这里可以根据需要添加实际处理逻辑）
        # 目前只是保存原始图片，保持名称、格式和尺寸
        results = []

        for idx, img_tensor in enumerate(images):
            if idx < len(filenames):
                orig_name = filenames[idx]
                name, _ = os.path.splitext(orig_name)

                # 使用原始扩展名
                if idx < len(extensions):
                    ext = extensions[idx]
                else:
                    ext = "png"  # 默认扩展名
                save_filename = f"{name}.{ext}"
            else:
                save_filename = f"processed_{idx}.png"

            # 转换tensor到图片
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file_path = os.path.join(output_path, save_filename)

            # 根据原始格式保存，保持原尺寸
            if ext in ["jpg", "jpeg"]:
                img.save(file_path, quality=100, subsampling=0)
            elif ext == "png":
                img.save(file_path, compress_level=0)
            elif ext in ["tiff", "tif"]:
                img.save(file_path, compression=None)
            elif ext == "webp":
                img.save(file_path, quality=100, method=6)
            elif ext == "bmp":
                img.save(file_path)
            elif ext == "gif":
                img.save(file_path)
            else:
                # 对于未知格式，尝试以最兼容的格式保存
                img.save(file_path)

            results.append(save_filename)

        return (images, json.dumps(results), len(results))


class LTX_LoadAndProcessSingleImage:
    """
    ltx加载单张图片 - 用于循环处理，实现加载一张、处理一张、保存一张
    保持原始文件名、格式和尺寸
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "./input", "placeholder": "图片目录路径"}),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    CATEGORY = "ltx/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "extension")
    FUNCTION = "load_single_image"

    def load_single_image(self, directory, index):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        # 支持的格式
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif')

        # 获取图片文件
        image_files = []
        for file in os.listdir(directory):
            if file.lower().endswith(supported_formats):
                image_files.append(file)

        image_files = sorted(image_files)

        if index >= len(image_files):
            raise IndexError(f"索引 {index} 超出范围，目录中只有 {len(image_files)} 张图片")

        image_file = image_files[index]
        image_path = os.path.join(directory, image_file)
        img = Image.open(image_path)

        # 获取扩展名
        _, ext = os.path.splitext(image_file)
        ext = ext.lower().lstrip('.')

        # 转换图片
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        # 处理mask
        if 'A' in img.getbands():
            mask_array = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask_tensor = 1. - torch.from_numpy(mask_array)
        else:
            mask_tensor = torch.zeros((img.height, img.width), dtype=torch.float32)

        # 返回图片、mask、文件名和扩展名
        return (img_tensor, mask_tensor.unsqueeze(0), image_file, ext)


class LTX_SaveImageProcessedWithOriginalFilename:
    """
    ltx使用原始文件名保存处理后的图片 - 改进版本
    保持原始文件名和格式，确保与加载的文件名一致
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_filename": ("STRING", {"default": "image.png"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ltx/Image"

    def save_images(self, images, original_filename, prompt=None, extra_pnginfo=None):
        output_path = self.output_dir
        os.makedirs(output_path, exist_ok=True)

        # 检查 original_filename 是否是 JSON 格式的字符串数组
        # 如果是 JSON 数组，从中获取第一个文件名用于扩展名
        try:
            parsed_filenames = json.loads(original_filename)
            if isinstance(parsed_filenames, list) and len(parsed_filenames) > 0:
                # 如果是文件名列表，使用第一个文件名的扩展名，并在保存时使用相应的索引文件名
                first_filename = parsed_filenames[0]
                name, orig_ext = os.path.splitext(first_filename)
                orig_ext = orig_ext.lower().lstrip('.')  # 获取原始扩展名
            else:
                # 如果不是数组或数组为空，按原始逻辑处理
                name, orig_ext = os.path.splitext(original_filename)
                orig_ext = orig_ext.lower().lstrip('.')  # 获取原始扩展名
        except json.JSONDecodeError:
            # 如果不是有效的 JSON，按原始逻辑处理
            name, orig_ext = os.path.splitext(original_filename)
            orig_ext = orig_ext.lower().lstrip('.')  # 获取原始扩展名

        results = []

        for idx, img_tensor in enumerate(images):
            # 转换tensor到numpy
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 构造保存文件名，使用原始扩展名
            # 如果有多张图片，则在文件名中添加索引以避免覆盖
            if len(images) > 1:
                name_part, ext_part = os.path.splitext(original_filename)
                save_filename = f"{name_part}_{idx}{ext_part}"
            else:
                save_filename = original_filename  # 确保文件名和扩展名完全一致

            file_path = os.path.join(output_path, save_filename)

            # 根据原始格式保存，不压缩，保持原尺寸
            # 使用PIL支持的格式确保文件扩展名正确
            if orig_ext in ["jpg", "jpeg"]:
                img.save(file_path, format="JPEG", quality=100, subsampling=0)  # 无损保存
            elif orig_ext == "png":
                img.save(file_path, format="PNG", compress_level=0)  # 无压缩保存
            elif orig_ext in ["tiff", "tif"]:
                img.save(file_path, format="TIFF", compression=None)
            elif orig_ext == "webp":
                img.save(file_path, format="WEBP", quality=100, method=6)  # 高质量保存
            else:
                # 对于未知格式，使用PNG作为默认格式
                # 重新构造文件路径使用PNG扩展名
                fallback_filename = f"{os.path.splitext(save_filename)[0]}.png"
                fallback_file_path = os.path.join(output_path, fallback_filename)
                img.save(fallback_file_path, format="PNG")
                # 更新results以反映实际保存的文件名
                save_filename = fallback_filename

            results.append({
                "filename": save_filename,
                "subfolder": "",
                "type": self.type
            })

        return {"ui": {"images": results}}


# 节点映射
NODE_CLASS_MAPPINGS = {
    "LTX_LoadImageWithFilename": LTX_LoadImageWithFilename,
    "LTX_LoadImagesFromDirWithFilename": LTX_LoadImagesFromDirWithFilename,
    "LTX_SaveImageWithOriginalFilename": LTX_SaveImageWithOriginalFilename,
    "LTX_SaveImageWithOriginalInfo": LTX_SaveImageWithOriginalInfo,
    "LTX_ProcessBatchWithOriginalNames": LTX_ProcessBatchWithOriginalNames,
    "LTX_LoadAndProcessSingleImage": LTX_LoadAndProcessSingleImage,
    "LTX_SaveImageProcessedWithOriginalFilename": LTX_SaveImageProcessedWithOriginalFilename,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX_LoadImageWithFilename": "LTX Load Image with Filename",
    "LTX_LoadImagesFromDirWithFilename": "LTX Load Images from Directory with Filenames",
    "LTX_SaveImageWithOriginalFilename": "LTX Save Image with Original Filename",
    "LTX_SaveImageWithOriginalInfo": "LTX Save Image with Original Info",
    "LTX_ProcessBatchWithOriginalNames": "LTX Process Batch with Original Names",
    "LTX_LoadAndProcessSingleImage": "LTX Load And Process Single Image",
    "LTX_SaveImageProcessedWithOriginalFilename": "LTX Save Image Processed with Original Filename",
}