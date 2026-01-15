# Qwen3-VL with VisionZip

> 2025秋《大语言模型原理、挑战与行业应用》期末大作业

基于 Qwen3-VL-2B-Instruct 的视觉 Token 压缩评测项目，将 [VisionZip](https://github.com/JIA-Lab-research/VisionZip) 方法应用于 Qwen3-VL 模型，实现了多种压缩策略变体，并在 MMStar、VQA v2.0、OCRBench、MME、RealWorldQA、POPE、CountBench 等数据集上进行评测。

## 📁 项目结构

```
qwen3-vl-with-visionzip/
├── README.md                              # 项目说明
├── requirements.txt                       # Python 依赖（基于 VLMEvalKit 修改）
├── qwen3_vl_visionzip.py                  # VisionZip 标准压缩实现（直接方法）
├── qwen3_vl_visionzip_notalign.py         # 分层不对齐压缩（分层独立方法）
├── qwen3_vl_visionzip_layeradjust.py      # 分层调整聚类压缩
├── qwen3_vl_visionzip_mixscore.py         # 综合权重压缩
├── qwen3_vl_visionzip_all.py              # 分层调整+综合权重压缩
├── test_qwen3_vl_local.py                 # 快速测试脚本（图生文/VQA）
├── eval_mmstar.py                         # MMStar 评测脚本
├── eval_ocrbench.py                       # OCRBench 评测脚本
├── eval_mme.py                            # MME 评测脚本
├── eval_realworldqa.py                    # RealWorldQA 评测脚本
├── eval_pope.py                           # POPE 评测脚本
├── eval_countbench.py                     # CountBench 评测脚本
├── eval_vqa_v2.py                         # VQA v2.0 评测脚本（需手动准备数据集）
└── vlmeval/                               # VLMEvalKit 工具库（克隆自官方仓库）
```

## 🛠️ 环境配置

### 1. 基础要求

- Python 3.10+
- CUDA 12.0+ (推荐 12.6)
- PyTorch 2.0+
- 16GB+ GPU 显存（推荐）

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/Gu-jq/qwen3-vl-with-visionzip.git
cd qwen3-vl-with-visionzip

# 安装依赖
pip install -r requirements.txt
```

### 3. 模型下载

脚本会自动从 Hugging Face 下载 `Qwen/Qwen3-VL-2B-Instruct` 模型权重。

## 🚀 快速开始

### Demo: 图生文 & 视觉问答

使用 `test_qwen3_vl_local.py` 进行快速测试：

```bash
# 1. 图生文（Image Captioning）
python test_qwen3_vl_local.py --image /path/to/image.jpg

# 2. 视觉问答（VQA）
python test_qwen3_vl_local.py \
    --image /path/to/image.jpg \
    --question "What is in this image?"

# 3. 启用 VisionZip 压缩（20% token 保留）
python test_qwen3_vl_local.py \
    --image /path/to/image.jpg \
    --use-visionzip \
    --dominant-ratio 0.15 \
    --contextual-ratio 0.05
```

**参数说明：**
- `--image`: 图像路径（必需）
- `--question`: 问题文本（可选，不提供则进行图生文）
- `--use-visionzip`: 启用 VisionZip 压缩
- `--dominant-ratio`: 主导 token 保留比例（默认 0.15）
- `--contextual-ratio`: 上下文 token 保留比例（默认 0.05）
- `--max-tokens`: 生成最大 token 数（默认 256）

## 📊 评测脚本

项目提供了多个数据集的评测脚本。**除 VQA v2.0 外，其他评测脚本会自动下载数据集，可直接运行。**

### 1. 通用评测（自动下载数据集）

以下评测脚本支持自动数据集下载，使用方式相同：

**支持的数据集：**
- `eval_mmstar.py` - MMStar（1500 条多模态推理题）
- `eval_ocrbench.py` - OCRBench（OCR 能力评测）
- `eval_mme.py` - MME（多模态评测）
- `eval_realworldqa.py` - RealWorldQA（真实场景问答）
- `eval_pope.py` - POPE（物体幻觉评测）
- `eval_countbench.py` - CountBench（计数能力评测）

**运行方式：**

```bash
# 1. 修改评测脚本中的配置（通常在第 30-35 行左右）
# USE_COMPRESSION = True/False
# DOMINANT_RATIO = 0.15
# CONTEXTUAL_RATIO = 0.05
# MAX_SAMPLES = None  # None 表示全量评测

# 2. 运行评测（以 MMStar 为例）
python eval_mmstar.py

# 3. 后台运行
nohup python eval_mmstar.py > mmstar_eval.log 2>&1 &
```

**配置说明：**
- `USE_COMPRESSION`: 是否启用 VisionZip（True/False）
- `DOMINANT_RATIO`: 主导 token 比例
- `CONTEXTUAL_RATIO`: 上下文 token 比例
- `MAX_SAMPLES`: 测试样本数（None 表示全量）

### 2. VQA v2.0 评测（需手动准备数据集）

评测脚本：`eval_vqa_v2.py`

#### 数据集准备

1. 下载 VQA v2.0 数据集：https://visualqa.org/download.html
2. 组织文件结构：

```
VQA_v2/
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_mscoco_val2014_annotations.json
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    └── ...
```

3. 修改 `eval_vqa_v2.py` 第 74 行的 `VQA_DATA_DIR` 为数据集路径

#### 运行评测

```bash
# 修改配置（第 80-84 行）
# CONFIG_NAME = "Baseline"
# USE_COMPRESSION = False
# COMPRESSION_MODULE = None  # "standard"/"notalign"/"layeradjust"/"mixscore"/"all"
# DOMINANT_RATIO = 1.0
# CONTEXTUAL_RATIO = 0.0

# 运行评测
python eval_vqa_v2.py

# 后台运行
nohup python eval_vqa_v2.py > vqa_eval.log 2>&1 &
```

**压缩模块说明：**
- `standard`: 标准 VisionZip 压缩
- `notalign`: 分层不对齐压缩
- `layeradjust`: 分层调整聚类压缩
- `mixscore`: 综合权重压缩
- `all`: 分层调整+综合权重压缩

评测结果保存在 `vqa_results/` 目录（JSON 格式）。

## 📝 VisionZip 压缩策略

项目实现了 5 种 VisionZip 压缩变体：

| 文件 | 策略描述 |
|------|---------|
| `qwen3_vl_visionzip.py` | 标准压缩（基于注意力重要性） |
| `qwen3_vl_visionzip_notalign.py` | 分层不对齐（不同层独立压缩） |
| `qwen3_vl_visionzip_layeradjust.py` | 分层调整聚类（基于聚类的压缩） |
| `qwen3_vl_visionzip_mixscore.py` | 综合权重（多指标融合） |
| `qwen3_vl_visionzip_all.py` | 分层调整+综合权重组合 |

## 🔗 相关链接

- **项目地址**: https://github.com/Gu-jq/qwen3-vl-with-visionzip
- **VisionZip 原始仓库**: https://github.com/JIA-Lab-research/VisionZip
- **Qwen3-VL 官方**: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- **VLMEvalKit**: https://github.com/open-compass/VLMEvalKit
- **VQA v2.0 数据集**: https://visualqa.org/download.html
