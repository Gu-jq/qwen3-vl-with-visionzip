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
HOME_DIR = "/root/qwen3-vl-with-visionzip"
sys.path.insert(0, HOME_DIR)

from vlmeval.dataset import build_dataset
from qwen3_vl_visionzip_layeradjust import Qwen3VLForConditionalGeneration as LocalQwen3VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration
from transformers import AutoProcessor

# --- 2. 评测配置 ---
MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"  # 模型权重路径或 HF ID
USE_COMPRESSION = True                   # <--- 手动切换：True 为评测压缩模型，False 为原模型
DOMINANT_RATIO = 0.15                    # 压缩比例配置
CONTEXTUAL_RATIO = 0.05
MAX_SAMPLES = None                       # 设置为整数跑快速测试，None 跑全量

def extract_choice(response, options):
    """
    匹配模型输出中的 A, B, C, D 选项。
    """
    response = response.strip().upper()
    if not response: return None
    # 提取所有出现的选项字母
    matches = re.findall(f"([{options}])", response)
    if matches:
        # 启发式：如果第一个字符就是选项，直接返回；否则返回最后一个出现的匹配项
        if response[0] in options: return response[0]
        return matches[-1]
    return None

def main():
    # 1. 自动加载 RealWorldQA 数据集
    print(f"正在准备 RealWorldQA 数据集...")
    dataset = build_dataset('RealWorldQA')
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

    # 3. 计分统计初始化
    correct = 0
    total = 0
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
        # 使用更符合用户预期的简单正则匹配逻辑
        options_avail = "".join([c for c in string.ascii_uppercase if c in item and not pd.isna(item[c])])
        pred = extract_choice(response, options_avail)
        gt = item['answer']
        
        if pred == gt:
            correct += 1
        total += 1
        
        results.append({
            'index': item['index'],
            'prediction': pred,
            'answer': gt,
            'response': response
        })
        
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # 6. 汇总报告
    accuracy = (correct / total * 100) if total > 0 else 0
    print("\n" + "="*40)
    print(f"RealWorldQA 评测报告 ({'压缩版' if USE_COMPRESSION else '原始版'})")
    print(f"总样本数: {total}")
    print(f"正确数量: {correct}")
    print(f"最终准确率: {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
