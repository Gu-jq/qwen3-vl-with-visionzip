#!/usr/bin/env python3
"""
Qwen3-VL 测试脚本 - 支持图生文和视觉问答
- 图生文模式：--image <path> （不提供问题）
- VQA模式：--image <path> --question <text>
- 支持VisionZip压缩：--use-visionzip --dominant-ratio 0.15 --contextual-ratio 0.05

用法示例：
1. 图生文: python test_qwen3_vl_local.py --image /path/to/image.jpg
2. VQA: python test_qwen3_vl_local.py --image /path/to/image.jpg --question "What is in this image?"
3. 使用VisionZip: python test_qwen3_vl_local.py --image /path/to/image.jpg --use-visionzip
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
    parser = argparse.ArgumentParser(description="Qwen3-VL 视觉测试 (图生文 & VQA)")
    parser.add_argument("--image", required=True, help="图像路径")
    parser.add_argument("--question", type=str, default=None, help="问题文本（可选，不提供则进行图生文）")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="HF模型ID")
    parser.add_argument("--use-visionzip", action="store_true", help="启用VisionZip压缩")
    parser.add_argument("--dominant-ratio", type=float, default=0.15, help="主要Token保留比例")
    parser.add_argument("--contextual-ratio", type=float, default=0.05, help="上下文Token保留比例")
    parser.add_argument("--max-tokens", type=int, default=256, help="生成的最大token数")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 打印配置信息
    print("=" * 80)
    print("🚀 Qwen3-VL 测试")
    print("=" * 80)
    print(f"📦 模型: {args.model}")
    print(f"💻 设备: {device}")
    print(f"🖼️  图像: {args.image}")
    
    if args.question:
        print(f"❓ 模式: VQA (视觉问答)")
        print(f"   问题: {args.question}")
    else:
        print(f"📝 模式: 图生文 (Image Captioning)")
    
    if args.use_visionzip:
        print(f"🔧 VisionZip: 已启用")
        print(f"   ├─ Dominant Ratio: {args.dominant_ratio}")
        print(f"   ├─ Contextual Ratio: {args.contextual_ratio}")
        print(f"   └─ 总保留率: {(args.dominant_ratio + args.contextual_ratio)*100:.0f}%")
    else:
        print(f"🔧 VisionZip: 未启用 (原始模型)")
    print("=" * 80)

    # 加载 processor（用于图像预处理）
    print("\n📥 加载 processor...")
    processor = AutoProcessor.from_pretrained(args.model)

    # 使用本地定义的模型类或原始模型
    print("📥 加载模型...")
    if args.use_visionzip:
        model = LocalQwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        
        model.config.visionzip_dominant_ratio = args.dominant_ratio
        model.config.visionzip_contextual_ratio = args.contextual_ratio
        print("   ✅ VisionZip模型已加载")
    else:
        model = HFQwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        print("   ✅ 原始模型已加载")

    # 加载图像
    if not os.path.exists(args.image):
        print(f"\n❌ 错误：图像文件不存在: {args.image}")
        return
    
    image = Image.open(args.image).convert("RGB")
    print(f"   图像尺寸: {image.size[0]}x{image.size[1]}")

    # 构建消息
    if args.question:
        # VQA模式
        text_prompt = args.question
    else:
        # 图生文模式
        text_prompt = "Describe the image."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    print("\n🔄 准备输入...")
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
    print(f"   输入序列长度: {input_sequence_length} tokens")

    # 如果使用VisionZip，先做一次前向获取压缩统计
    if args.use_visionzip:
        print("\n📊 VisionZip 压缩分析...")
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
            print(f"   压缩前: {before} 个图像tokens")
            print(f"   压缩后: {after_value} 个图像tokens")
            print(f"   移除: {removed} tokens ({100-kept_pct:.1f}%)")
            print(f"   实际保留率: {kept_pct:.1f}%")
            
            if image_range[0] is not None and image_range[1] is not None:
                span = image_range[1] - image_range[0] + 1
                print(f"   图像占位符范围: {image_range[0]}-{image_range[1]} (共{span} tokens)")
        else:
            print("   ⚠️  未检测到图像占位符")
        
        if keep_mask is not None:
            final_len = int(keep_mask.sum().item())
            print(f"   压缩后序列长度: {final_len} / {input_ids.shape[1]} tokens")

    # 生成回答
    print(f"\n🤖 生成回答 (最多{args.max_tokens}个新tokens)...")

    gen_inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_tokens,
        return_dict_in_generate=True,
        do_sample=False,  # 贪婪解码保证可复现性
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
    
    # 显示结果
    print("\n" + "=" * 80)
    print("✅ 生成结果:")
    print("-" * 80)
    print(decoded)
    print("=" * 80)


if __name__ == "__main__":
    main()
