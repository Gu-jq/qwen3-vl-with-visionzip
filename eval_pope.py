import os
import sys
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import re
import string
import numpy as np

# --- 1. 环境准备与 Monkeypatch ---
# 针对 Python 3.12 修改 transformers 的 docstring 装饰器问题
import transformers.utils as _transformers_utils
def _noop_auto_docstring(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs: return args[0]
    def decorator(obj): return obj
    return decorator
_transformers_utils.auto_docstring = _noop_auto_docstring

# 将当前目录加入 path
HOME_DIR = "/root/qwen3-vl-with-visionzip"
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
MAX_SAMPLES = None                       # 设置为整数跑快速测试，None 跑全量

def extract_yorn(response):
    """
    匹配模型输出中的 Yes 或 No。
    """
    res = response.lower().strip()
    if 'yes' in res:
        return 'Yes'
    if 'no' in res:
        return 'No'
    return 'Unknown'

def cal_f1_score(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score, precision, recall

def main():
    # 1. 自动加载 POPE 数据集
    print(f"正在准备 POPE 数据集...")
    dataset = ImageYORNDataset('POPE')
    data = dataset.data
    
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

    # 3. 结果收集
    results = []
    indices = range(len(data))
    if MAX_SAMPLES:
        indices = indices[:MAX_SAMPLES]

    # 4. 推理循环
    for i in tqdm(indices, desc="评测中"):
        item = data.iloc[i]
        msgs_vl = dataset.build_prompt(item)
        
        formatted_content = []
        for m in msgs_vl:
            if m['type'] == 'image':
                img_path = m['value']
                formatted_content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
            elif m['type'] == 'text':
                formatted_content.append({"type": "text", "text": m['value']})
        
        messages = [{"role": "user", "content": formatted_content}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # 5. 评分逻辑
        pred = extract_yorn(response)
        gt = item['answer']
        
        results.append({
            'index': item['index'],
            'prediction': pred,
            'answer': gt,
            'category': item.get('category', 'Default'),
            'response': response
        })
        
        if i % 20 == 0:
            torch.cuda.empty_cache()

    # 6. 汇总报告 (POPE 专用指标)
    res_df = pd.DataFrame(results)
    y_true = np.array([1 if i == 'Yes' else 0 for i in res_df['answer']])
    y_pred = np.array([1 if i == 'Yes' else 0 for i in res_df['prediction']])
    
    f1, precision, recall = cal_f1_score(y_true, y_pred)
    accuracy = (res_df['prediction'] == res_df['answer']).mean() * 100
    yes_ratio = (res_df['prediction'] == 'Yes').mean() * 100

    print("\n" + "="*40)
    print(f"POPE 评测报告 ({'压缩版' if USE_COMPRESSION else '原始版'})")
    print(f"总样本数: {len(res_df)}")
    print(f"准确率 (Accuracy): {accuracy:.2f}%")
    print(f"精确度 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall): {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"Yes 比例 (Yes Ratio): {yes_ratio:.2f}%")
    print("="*40)

    # 如果有类别划分，打印明细
    if 'category' in res_df.columns:
        print("\n分类明细 (Accuracy):")
        cate_acc = res_df.groupby('category').apply(lambda x: (x['prediction'] == x['answer']).mean() * 100)
        for cat, acc in cate_acc.items():
            print(f"  - {cat}: {acc:.2f}%")

if __name__ == "__main__":
    main()
