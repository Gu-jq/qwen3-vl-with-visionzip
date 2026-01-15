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

# 将当前目录和 VLMEvalKit 目录加入 path
HOME_DIR = "/root"
sys.path.insert(0, os.path.join(HOME_DIR, 'VLMEvalKit'))
sys.path.insert(0, HOME_DIR)

from vlmeval.dataset import OCRBench
from qwen3_vl_visionzip_notalign import Qwen3VLForConditionalGeneration as LocalQwen3VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration
from transformers import AutoProcessor

# --- 2. 评测配置 ---
MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"  # 模型权重路径或 HF ID
USE_COMPRESSION = True                   # <--- 手动切换：True 为评测压缩模型，False 为原模型
DOMINANT_RATIO = 0.55                    # 压缩比例配置
CONTEXTUAL_RATIO = 0.05
MAX_SAMPLES = None                       # 设置为整数跑快速测试，None 跑全量（1000条左右）

def main():
    # 1. 自动加载 OCRBench 数据集
    print(f"正在准备 OCRBench 数据集...")
    dataset = OCRBench('OCRBench')
    
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

    # 3. 计分统计初始化 (按 OCRBench 官方维度)
    scores_by_category = {
        'Regular Text Recognition': 0,
        'Irregular Text Recognition': 0,
        'Artistic Text Recognition': 0,
        'Handwriting Recognition': 0,
        'Digit String Recognition': 0,
        'Non-Semantic Text Recognition': 0,
        'Scene Text-centric VQA': 0,
        'Doc-oriented VQA': 0,
        'Key Information Extraction': 0,
        'Handwritten Mathematical Expression Recognition': 0,
    }

    indices = range(len(dataset))
    if MAX_SAMPLES:
        indices = indices[:MAX_SAMPLES]

    # 4. 推理循环
    for i in tqdm(indices, desc="评测中"):
        item = dataset[i]
        msgs_vl = dataset.build_prompt(item)
        
        formatted_content = []
        for m in msgs_vl:
            if m['type'] == 'image':
                formatted_content.append({"type": "image", "image": Image.open(m['value']).convert("RGB")})
            elif m['type'] == 'text':
                formatted_content.append({"type": "text", "text": m['value']})
        
        messages = [{"role": "user", "content": formatted_content}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # 5. 评分逻辑 (模仿 VLMEvalKit 原文)
        category = item['category']
        try:
            # 答案在 TSV 中通常是 ['ans1', 'ans2'] 的字符串形式
            answers = eval(item['answer'])
        except:
            answers = [str(item['answer'])]
            
        predict = str(response)
        
        if category == 'Handwritten Mathematical Expression Recognition':
            # 数学公式识别：去空格换行后包含即中
            predict_clean = predict.strip().replace('\n', '').replace(' ', '')
            for ans in answers:
                ans_clean = str(ans).strip().replace('\n', '').replace(' ', '')
                if ans_clean in predict_clean:
                    scores_by_category[category] += 1
                    break
        else:
            # 其他分类：小写、去换行后包含即中
            predict_clean = predict.lower().strip().replace('\n', ' ')
            for ans in answers:
                ans_clean = str(ans).lower().strip().replace('\n', ' ')
                if ans_clean in predict_clean:
                    scores_by_category[category] += 1
                    break
        
        torch.cuda.empty_cache()

    # 6. 汇总报告
    total_samples = len(indices)
    recognition_score = sum([scores_by_category[k] for k in [
        'Regular Text Recognition', 'Irregular Text Recognition', 'Artistic Text Recognition',
        'Handwriting Recognition', 'Digit String Recognition', 'Non-Semantic Text Recognition'
    ]])
    
    final_score = (recognition_score + 
                   scores_by_category['Scene Text-centric VQA'] + 
                   scores_by_category['Doc-oriented VQA'] + 
                   scores_by_category['Key Information Extraction'] + 
                   scores_by_category['Handwritten Mathematical Expression Recognition'])
    
    accuracy = (final_score / (total_samples * 1.0)) * 100 if total_samples > 0 else 0

    print("\n" + "="*40)
    print(f"OCRBench 评测报告 ({'压缩版' if USE_COMPRESSION else '原始版'})")
    print(f"总样本数: {total_samples}")
    print(f"最终总分 (Final Score): {final_score}")
    print(f"标准化得分 (Final Score Norm): {final_score/10:.2f}")
    print("-" * 20)
    print(f"维度得分明细:")
    print(f"  - Text Recognition: {recognition_score}")
    print(f"  - Scene Text-centric VQA: {scores_by_category['Scene Text-centric VQA']}")
    print(f"  - Doc-oriented VQA: {scores_by_category['Doc-oriented VQA']}")
    print(f"  - Key Information Extraction: {scores_by_category['Key Information Extraction']}")
    print(f"  - Math Expression: {scores_by_category['Handwritten Mathematical Expression Recognition']}")
    print("="*40)

if __name__ == "__main__":
    main()
