#!/usr/bin/env python3
"""
简短测试脚本：使用本地的 `qwen3_vl_visionzip.py` 中的模型类，加载 Hugging Face 上的权重，
并对给定图片做一次视觉前向以验证修改后的模型代码能否正常运行。

用法示例：
python test_qwen3_vl_local.py --image /path/to/your.jpg --model Qwen/Qwen3-VL-2B
"""
import sys
import os
import argparse
import torch
from PIL import Image

import transformers.utils as _transformers_utils

def _noop_auto_docstring(*args, **kwargs):
    # Decorator no-op to bypass transformers auto_docstring issues on Python 3.12
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(obj):
        return obj

    return decorator

_transformers_utils.auto_docstring = _noop_auto_docstring

# 确保工作目录在 sys.path 前面，这样可以 import 本地的 qwen3_vl_visionzip 模块
sys.path.insert(0, os.path.dirname(__file__) or os.getcwd())

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration

# 从本地文件导入模型类（文件：qwen3_vl_visionzip.py）
from qwen3_vl_visionzip import Qwen3VLForConditionalGeneration as LocalQwen3VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file to test")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="HF model id (default: Qwen/Qwen3-VL-2B-Instruct)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载 processor（用于图像预处理）
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model)

    # 使用本地定义的模型类从 Hugging Face hub 加载权重
    print("Loading model (using local class)...")
    model = LocalQwen3VLForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    model.config.visionzip_dominant_ratio = 0.15
    model.config.visionzip_contextual_ratio = 0.05

    image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    print("Building chat inputs via processor.apply_chat_template...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise RuntimeError("processor did not return 'pixel_values'. Check processor/model compatibility.")
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("processor did not return 'image_grid_thw'. Make sure the processor is compatible with the model.")
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    input_sequence_length = inputs["input_ids"].shape[1]

    print("Profiling Visionzip compression with the base model...")
    with torch.no_grad():
        _ = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    before = getattr(model.model, "_last_visionzip_image_tokens_before", None)
    after = getattr(model.model, "_last_visionzip_image_tokens_after", None)
    image_range = getattr(model.model, "_last_visionzip_image_range", (None, None))
    keep_mask = getattr(model.model, "_last_visionzip_keep_mask", None)
    if before is not None and before > 0:
        after_value = after if after is not None else 0
        removed = before - after_value
        kept_pct = (after_value / before) * 100
        print(
            f"Visionzip pruned {removed} image tokens (kept {after_value}/{before}, {kept_pct:.1f}% remain)."
        )
        if image_range[0] is not None and image_range[1] is not None:
            span = image_range[1] - image_range[0] + 1
            print(f"Placeholder span {image_range[0]}-{image_range[1]} covers {span} tokens.")
    else:
        print("Visionzip stats unavailable; no image placeholders were spotted in this batch.")
    if keep_mask is not None:
        print(
            f"Sequence length after Visionzip cropping: {int(keep_mask.sum().item())} / {input_ids.shape[1]} tokens."
        )

    # 调用模型的 get_image_features 先一次前向，验证输入准备
    print("Running image forward (get_image_features)...")
    with torch.no_grad():
        image_embeds, deepstack = model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)

    if isinstance(image_embeds, (list, tuple)):
        print("image_embeds: list with lengths:", [e.shape for e in image_embeds])
    else:
        try:
            print("image_embeds shape:", image_embeds.shape)
        except Exception:
            print("image_embeds type:", type(image_embeds))

    if deepstack is None:
        print("deepstack features: None")
    else:
        print("deepstack features: list len:", len(deepstack))

    # 准备生成（最多 256 新 token）
    print("Generating text (max 256 new tokens)...")

    gen_inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=256,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        outputs = model.generate(**gen_inputs)

    trimmed_sequences = [
        seq[input_sequence_length:].tolist() for seq in outputs.sequences
    ]
    decoded = processor.batch_decode(
        trimmed_sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("Generated:", decoded)


if __name__ == "__main__":
    main()
