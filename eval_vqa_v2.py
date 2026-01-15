"""
VQA v2.0 评测脚本 for Qwen3-VL with VisionZip

数据集准备:
-----------
1. 下载 VQA v2.0 数据集: https://visualqa.org/download.html
2. 组织文件结构如下:
   VQA_v2/
   ├── v2_OpenEnded_mscoco_val2014_questions.json
   ├── v2_mscoco_val2014_annotations.json
   ├── val2014/
   │   ├── COCO_val2014_000000000042.jpg
   │   ├── COCO_val2014_000000000073.jpg
   │   └── ...
   ├── v2_OpenEnded_mscoco_train2014_questions.json  (可选，用于训练集评测)
   ├── v2_mscoco_train2014_annotations.json          (可选)
   └── train2014/                                     (可选)

3. 修改第 74 行的 VQA_DATA_DIR 为你的数据集路径

配置与运行:
-----------
1. 修改第 80-84 行的配置参数:
   - CONFIG_NAME: 结果文件命名
   - USE_COMPRESSION: True/False (是否使用 VisionZip)
   - COMPRESSION_MODULE: "standard"/"notalign"/"layeradjust"/"mixscore"/"all"
   - DOMINANT_RATIO: 主导 token 保留比例 (如 0.15 表示 15%)
   - CONTEXTUAL_RATIO: 上下文 token 保留比例 (如 0.05 表示 5%)

2. 运行示例:
   # Baseline (无压缩)
   python eval_vqa_v2.py
   
   # VisionZip 20% 标准压缩
   # 先修改配置: USE_COMPRESSION=True, COMPRESSION_MODULE="standard", 
   #            DOMINANT_RATIO=0.15, CONTEXTUAL_RATIO=0.05
   python eval_vqa_v2.py
   
   # 后台运行
   nohup python eval_vqa_v2.py > vqa_eval.log 2>&1 &

3. 结果文件保存在 vqa_results/ 目录
"""
import os
import sys
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# --- 1. 环境准备与 Monkeypatch ---
# 针对 Python 3.12 修改 transformers 的 docstring 装饰器问题
import transformers.utils as _transformers_utils
def _noop_auto_docstring(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs: return args[0]
    def decorator(obj): return obj
    return decorator
_transformers_utils.auto_docstring = _noop_auto_docstring

# 将 VLMEvalKit 加入 path 以使用标准 VQA 评测工具
VISIONZIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(VISIONZIP_DIR, "vlmeval"))

from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration
from transformers import AutoProcessor
from vlmeval.dataset.utils.vqa_eval import process_answer

# --- 2. 评测配置（单次运行测试一种配置）---
MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
VQA_DATA_DIR = "/YueYangDi/sparky/LLM/LLM_project/dataset/VQA_v2"
OUTPUT_DIR = os.path.join(VISIONZIP_DIR, "vqa_results")
MAX_SAMPLES = 1000  # 测试样本数，None 表示全量
MAX_NEW_TOKENS = 128

# 🔧 配置参数（每次运行修改这些参数）<--- 手动修改这里
CONFIG_NAME = "Baseline"           # 配置名称（用于结果文件命名）
USE_COMPRESSION = False            # 是否使用 VisionZip 压缩
COMPRESSION_MODULE = None          # 压缩模块名称："standard" / "notalign" / "layeradjust" / "mixscore" / "all"
DOMINANT_RATIO = 1.0               # 主导 token 比例
CONTEXTUAL_RATIO = 0.0             # 上下文 token 比例

# --- 3. VQA 标准评测函数 ---
def vqa_accuracy(prediction, ground_truths):
    """
    VQA v2 官方评分标准
    如果预测答案在10个ground truth中至少出现3次，得1分
    否则得分为 min(matching_count/3, 1.0)
    """
    pred_processed = process_answer(prediction)
    gts_processed = [process_answer(gt) for gt in ground_truths]
    
    accuracies = []
    for i, gt in enumerate(gts_processed):
        other_gts = [gts_processed[j] for j in range(len(gts_processed)) if j != i]
        matching = [g for g in other_gts if g == pred_processed]
        acc = min(1.0, len(matching) / 3.0)
        accuracies.append(acc)
    
    return np.mean(accuracies) if accuracies else 0.0

# --- 4. 数据加载 ---
def load_vqa_data(split="val"):
    """加载 VQA v2 数据"""
    questions_file = os.path.join(VQA_DATA_DIR, f"v2_OpenEnded_mscoco_{split}2014_questions.json")
    annotations_file = os.path.join(VQA_DATA_DIR, f"v2_mscoco_{split}2014_annotations.json")
    image_dir = os.path.join(VQA_DATA_DIR, f"{split}2014")
    
    print(f"正在加载 VQA v2 数据集: {split}2014...")
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    # 构建 question_id 到 annotation 的映射
    id_to_annotation = {ann['question_id']: ann for ann in annotations_data['annotations']}
    
    samples = []
    for q in questions_data['questions']:
        qid = q['question_id']
        if qid not in id_to_annotation:
            continue
        
        ann = id_to_annotation[qid]
        image_id = q['image_id']
        image_filename = f"COCO_{split}2014_{image_id:012d}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            continue
        
        # 提取所有答案
        answers = [a['answer'] for a in ann['answers']]
        
        samples.append({
            'question_id': qid,
            'image_path': image_path,
            'question': q['question'],
            'answers': answers,
            'question_type': ann.get('question_type', 'unknown'),
            'answer_type': ann.get('answer_type', 'unknown'),
        })
    
    # 限制样本数量
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
    
    print(f"✅ 加载完成: {len(samples)} 个样本")
    return samples

# --- 5. 模型推理与评测 ---
def process_vision_info(messages):
    """从消息中提取图像"""
    image_inputs = []
    for message in messages:
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
    return image_inputs if image_inputs else None, None

def evaluate_model(model, processor, samples, device, config_name):
    """单样本推理模式"""
    results = []
    timing_stats = []
    compression_stats = []
    
    for sample in tqdm(samples, desc=f"评测 {config_name}"):
        try:
            # 加载图像
            image = Image.open(sample['image_path']).convert('RGB')
            
            # 构造消息 - 添加简短答案提示
            question_with_instruction = f"{sample['question']} Answer with a single word or short phrase."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question_with_instruction},
                    ],
                }
            ]
            
            # 准备输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            # 推理
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # 解码
            output_ids = [
                generated_ids[i][len(inputs.input_ids[i]):]
                for i in range(len(generated_ids))
            ]
            prediction = processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # 收集压缩统计
            if hasattr(model, '_last_visionzip_image_tokens_before'):
                compression_stats.append({
                    'tokens_before': model._last_visionzip_image_tokens_before,
                    'tokens_after': model._last_visionzip_image_tokens_after
                })
            
            # 计算准确率
            accuracy = vqa_accuracy(prediction, sample['answers'])
            
            results.append({
                'question_id': sample['question_id'],
                'question': sample['question'],
                'prediction': prediction,
                'ground_truths': sample['answers'],
                'accuracy': accuracy,
                'question_type': sample['question_type'],
                'answer_type': sample['answer_type'],
                'inference_time_ms': inference_time
            })
            timing_stats.append(inference_time)
            
        except Exception as e:
            print(f"⚠️  样本 {sample['question_id']} 处理失败: {e}")
            continue
    
    return results, timing_stats, compression_stats

# --- 6. 统计计算 ---
def compute_statistics(results, timing_stats, compression_stats, config_name):
    """计算统计信息"""
    stats = {
        'config_name': config_name,
        'total_samples': len(results),
        'overall_accuracy': np.mean([r['accuracy'] for r in results]) * 100,
        'avg_inference_time_ms': np.mean(timing_stats),
        'std_inference_time_ms': np.std(timing_stats),
    }
    
    # 按问题类型统计
    by_qtype = defaultdict(list)
    for r in results:
        by_qtype[r['question_type']].append(r['accuracy'])
    stats['accuracy_by_question_type'] = {
        qtype: np.mean(accs) * 100 for qtype, accs in by_qtype.items()
    }
    
    # 按答案类型统计
    by_atype = defaultdict(list)
    for r in results:
        by_atype[r['answer_type']].append(r['accuracy'])
    stats['accuracy_by_answer_type'] = {
        atype: np.mean(accs) * 100 for atype, accs in by_atype.items()
    }
    
    # 压缩统计
    if compression_stats:
        stats['avg_tokens_before'] = np.mean([s['tokens_before'] for s in compression_stats])
        stats['avg_tokens_after'] = np.mean([s['tokens_after'] for s in compression_stats])
        stats['compression_ratio'] = stats['avg_tokens_after'] / stats['avg_tokens_before'] * 100
    
    return stats

# --- 7. 结果打印 ---
def print_results(stats):
    """打印结果"""
    print("\n" + "="*80)
    print(f"📊 VQA v2 评测结果 - {stats['config_name']}")
    print("="*80)
    
    print(f"\n总体准确率: {stats['overall_accuracy']:.2f}%")
    print(f"平均推理时间: {stats['avg_inference_time_ms']:.1f} ± {stats['std_inference_time_ms']:.1f} ms")
    
    if 'compression_ratio' in stats:
        print(f"压缩比例: {stats['avg_tokens_before']:.1f} → {stats['avg_tokens_after']:.1f} tokens ({stats['compression_ratio']:.1f}%)")
    
    print(f"\n按问题类型准确率:")
    for qtype, acc in sorted(stats['accuracy_by_question_type'].items()):
        print(f"  - {qtype}: {acc:.2f}%")
    
    print(f"\n按答案类型准确率:")
    for atype, acc in sorted(stats['accuracy_by_answer_type'].items()):
        print(f"  - {atype}: {acc:.2f}%")
    
    print("\n" + "="*80)

# --- 8. 主函数 ---
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("🚀 开始 VQA v2 评测")
    samples = load_vqa_data(split="val")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    # 加载 processor
    print(f"📥 加载 processor: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left'  # decoder-only 模型使用 left padding
    
    # 打印当前配置
    print(f"\n{'='*80}")
    print(f"🔧 配置: {CONFIG_NAME}")
    print(f"   压缩模式: {USE_COMPRESSION}")
    if USE_COMPRESSION:
        print(f"   压缩模块: {COMPRESSION_MODULE}")
        print(f"   Token比例: dominant={DOMINANT_RATIO}, contextual={CONTEXTUAL_RATIO}")
    print(f"{'='*80}")
    
    # 加载模型
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    if USE_COMPRESSION:
        # 动态导入对应的压缩模块
        module_map = {
            "standard": "qwen3_vl_visionzip",
            "notalign": "qwen3_vl_visionzip_notalign",
            "layeradjust": "qwen3_vl_visionzip_layeradjust",
            "mixscore": "qwen3_vl_visionzip_mixscore",
            "all": "qwen3_vl_visionzip_all",
        }
        
        module_name = module_map.get(COMPRESSION_MODULE, "qwen3_vl_visionzip")
        compression_module = __import__(module_name)
        ModelClass = compression_module.Qwen3VLForConditionalGeneration
        
        model = ModelClass.from_pretrained(MODEL_PATH, **model_kwargs).to(device).eval()
        model.config.visionzip_dominant_ratio = DOMINANT_RATIO
        model.config.visionzip_contextual_ratio = CONTEXTUAL_RATIO
        print(f"✅ VisionZip 已配置 ({COMPRESSION_MODULE}): {(DOMINANT_RATIO + CONTEXTUAL_RATIO)*100:.0f}% token 保留率")
    else:
        model = HFQwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, **model_kwargs
        ).to(device).eval()
        print("✅ 原生 Baseline 模型")
    
    # 评测
    results, timing_stats, compression_stats = evaluate_model(
        model, processor, samples, device, CONFIG_NAME
    )
    
    # 计算统计
    stats = compute_statistics(results, timing_stats, compression_stats, CONFIG_NAME)
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f"vqa_{CONFIG_NAME}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({'results': results, 'stats': stats}, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细结果已保存: {results_file}")
    
    # 打印结果
    print_results(stats)

if __name__ == "__main__":
    main()
