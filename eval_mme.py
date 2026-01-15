import os
import sys
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import re
import string

# --- 1. 环境准备与 Monkeypatch ---
# 针对 Python 3.12 修改 transformers 的 docstring 装饰器问题
import transformers.utils as _transformers_utils
def _noop_auto_docstring(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs: return args[0]
    def decorator(obj): return obj
    return decorator
_transformers_utils.auto_docstring = _noop_auto_docstring

# 将当前目录加入 path
HOME_DIR = "/root"
sys.path.insert(0, os.path.join(HOME_DIR, 'qwen3-vl-with-visionzip'))
sys.path.insert(0, HOME_DIR)

from vlmeval.dataset import ImageYORNDataset
from qwen3_vl_visionzip_layeradjust import Qwen3VLForConditionalGeneration as LocalQwen3VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration
from transformers import AutoProcessor

# --- 2. 评测配置 ---
MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"  # 模型权重路径或 HF ID
USE_COMPRESSION = True                   # <--- 手动切换：True 为评测压缩模型，False 为原模型
DOMINANT_RATIO = 0.55                    # 压缩比例配置
CONTEXTUAL_RATIO = 0.05
MAX_SAMPLES = None                       # 设置为整数跑快速测试，None 跑全量（2374条）

def extract_yorn(response):
    """
    提取 Yes/No 答案
    """
    response = response.lower().strip()
    # 移除标点
    response = response.translate(str.maketrans('', '', string.punctuation))
    
    if 'yes' in response and 'no' not in response:
        return 'Yes'
    if 'no' in response and 'yes' not in response:
        return 'No'
    
    # 如果都有或者都没有，尝试前缀匹配
    words = response.split()
    if words:
        if words[0] == 'yes': return 'Yes'
        if words[0] == 'no': return 'No'
        
    return None

def main():
    # 1. 自动加载 MME 数据集
    print(f"正在准备 MME 数据集...")
    dataset = ImageYORNDataset('MME')
    
    # 2. 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载模型: {MODEL_PATH} (设备: {device})...")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    if USE_COMPRESSION:
        model = LocalQwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            **model_kwargs
        ).to(device).eval()
        print(f"【运行模式】Visionzip 压缩开启 (Ratio: {DOMINANT_RATIO}, {CONTEXTUAL_RATIO})")
        model.config.visionzip_dominant_ratio = DOMINANT_RATIO
        model.config.visionzip_contextual_ratio = CONTEXTUAL_RATIO
    else:
        model = HFQwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            **model_kwargs
        ).to(device).eval()
        print("【运行模式】原生模型")

    # 3. 计分统计初始化
    scores_by_category = {}
    totals_by_category = {}

    indices = range(len(dataset))
    if MAX_SAMPLES:
        indices = indices[:MAX_SAMPLES]

    # 4. 推理循环
    for i in tqdm(indices, desc="评测中"):
        item = dataset[i]
        category = item.get('category', 'Unknown')
        if category not in scores_by_category:
            scores_by_category[category] = 0
            totals_by_category[category] = 0
            
        msgs_vl = dataset.build_prompt(item)
        
        formatted_content = []
        for m in msgs_vl:
            if m['type'] == 'image':
                # 兼容不同输入方式
                img_path = m['value']
                try:
                    formatted_content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
            elif m['type'] == 'text':
                formatted_content.append({"type": "text", "text": m['value']})
        
        if not formatted_content:
            continue

        messages = [{"role": "user", "content": formatted_content}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # 5. 评分逻辑
        pred = extract_yorn(response)
        gt = item['answer']
        
        if pred == gt:
            scores_by_category[category] += 1
        totals_by_category[category] += 1
        
        # 每隔 100 条清理一次显存
        if i % 100 == 0:
            torch.cuda.empty_cache()

    # 6. 汇总报告
    print("\n" + "="*40)
    print(f"MME 评测报告 ({'压缩版' if USE_COMPRESSION else '原始版'})")
    
    overall_correct = 0
    overall_total = 0
    
    categories = sorted(scores_by_category.keys())
    print(f"{'Category':<30} | {'Correct':<8} | {'Total':<8} | {'Acc (%)':<8}")
    print("-" * 65)
    
    for cat in categories:
        correct = scores_by_category[cat]
        total = totals_by_category[cat]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{cat:<30} | {correct:<8} | {total:<8} | {acc:>7.2f}")
        overall_correct += correct
        overall_total += total
        
    overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print("-" * 65)
    print(f"{'Overall':<30} | {overall_correct:<8} | {overall_total:<8} | {overall_acc:>7.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
