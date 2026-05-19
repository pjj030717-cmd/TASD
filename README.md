# Speculative Decoding 实验

多解码方法对比实验平台：自回归 / 标准推测解码 / FLy / FSD / TASD

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/pjj030717-cmd/-.git
cd experiments
```

### 2. 一键安装环境

```bash
bash setup.sh
```

此脚本会自动：
- 检测 CUDA 环境
- 安装 Miniconda（如未安装）
- 克隆第三方仓库（FLy、FSD）
- 创建 3 个 conda 环境

### 3. 下载模型

```bash
bash download_models.sh
```

或手动下载：
```bash
# HuggingFace
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ --local-dir ./models/Qwen2.5-72B-Instruct-AWQ
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --local-dir ./models/Qwen2.5-7B-Instruct-GPTQ-Int4

# ModelScope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ', cache_dir='./models')"
```

### 4. 运行测试

```bash
# 单 GPU 模式
conda activate ./envs/speculative
python src/main.py

# 多 GPU 模型并行（2 张卡）
python src/main.py --num_gpus 2

# 自回归基线
python src/main.py --skip_spec

# FLy 框架
conda activate ./envs/fly
export HF_ALLOW_CODE_EVAL=1
fly --model hf --model_args pretrained=./models/Qwen2.5-7B-Instruct-GPTQ-Int4,config_path=configs/fly/FLy_Qwen2.5_72b.json --tasks humaneval_instruct --batch_size 1 --apply_chat_template --confirm_run_unsafe_code

# FSD 框架
conda activate ./envs/fsd
python third_party/fsd/csqa_eval_example.py --small_model_id ./models/Qwen2.5-7B-Instruct-GPTQ-Int4 --large_model_id ./models/Qwen2.5-72B-Instruct-AWQ --fsd_div_threshold 0.4 --fsd_div_type "js_div" --num_evals 5

# TASD 框架
conda activate ./envs/speculative
python src/tasd_decode.py
```

## 项目结构

```
experiments/
├── src/                          # 核心代码
│   ├── main.py                   # 主入口
│   ├── gpu_utils.py              # GPU 检查工具
│   ├── download_models.py        # 模型下载脚本
│   ├── autoregressive_decode.py  # 自回归解码
│   ├── speculative_decode.py     # 推测解码
│   ├── tasd_decode.py            # TASD 质量驱动宽松验证推测解码
│   ├── tasd_solver.py            # TASD 参数优化求解器
│   ├── speed_bench.py            # SPEED-Bench 数据集测试
│   └── humaneval_bench.py        # HumanEval 数据集测试
├── decoding_methods/             # 各解码方法配置
│   ├── fly/                      # FLy 配置
│   ├── fsd/                      # FSD 配置
│   ├── autoregressive/           # 自回归配置
│   └── speculative/              # 推测解码配置
├── configs/                      # 配置文件
├── models/                       # 模型权重（需下载）
├── third_party/                  # 第三方仓库（setup.sh 自动克隆）
├── setup.sh                      # 一键安装脚本
├── download_models.sh            # 模型下载脚本
├── .gitignore
└── README.md
```

## 环境说明

| 环境 | 路径 | transformers | 用途 |
|------|------|--------------|------|
| FLy | `envs/fly` | 4.57.6 | FLy 松散推测解码 |
| FSD | `envs/fsd` | 4.44.0 | FSD 模糊推测解码 |
| speculative | `envs/speculative` | 5.8.1 | 自回归/推测解码/TASD |

## 模型说明

| 角色 | 模型 | 量化 | 大小 |
|------|------|------|------|
| Target | Qwen2.5-72B-Instruct | 4-bit AWQ | ~36GB |
| Draft | Qwen2.5-7B-Instruct | 4-bit GPTQ | ~3.5GB |

## TASD 框架

TASD（质量驱动的免训练宽松验证推测解码）是本项目的核心创新。

### 核心思想

- 不要求 draft 和 target 输出完全一致
- 允许在 ε 宽容度内接受 draft token
- 通过 KL 散度或概率比值判断是否接受

### 理论保证

质量损失上界：Δ ≤ k · √(ε/2)

其中 k 是草稿长度，ε 是宽容度阈值。

### 运行 TASD

```bash
# 基础运行
python src/tasd_decode.py

# HumanEval 基准测试
python src/humaneval_bench.py --max_samples 10

# SPEED-Bench 基准测试
python src/speed_bench.py
```

### 性能对比（HumanEval）

| 方法 | 速度 (tok/s) | 接受率 |
|------|-------------|--------|
| 自回归 (AR) | 11.8 | - |
| TASD (γ=5, ε=0.05) | 15.3 | 95.74% |
| FLy | 24.5 | - |

## 运行

### FLy 松散推测解码

```bash
conda activate /cloud/cloud-ssd1/-/envs/fly
export HF_ALLOW_CODE_EVAL=1
fly --model hf \
    --model_args pretrained=./models/Qwen2.5-7B-Instruct-GPTQ-Int4,config_path=decoding_methods/fly/FLy_Qwen2.5_72b.json \
    --tasks humaneval_instruct \
    --batch_size 1 \
    --apply_chat_template \
    --confirm_run_unsafe_code
```

### FSD 模糊推测解码 (完整评估)

```bash
conda activate /cloud/cloud-ssd1/-/envs/fsd
python third_party/fsd/csqa_eval_example.py \
    --small_model_id ./models/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --large_model_id ./models/Qwen2.5-72B-Instruct-AWQ \
    --fsd_div_threshold 0.4 \
    --fsd_div_type "js_div" \
    --num_evals 5
```

### EAGLE 推测解码

```bash
conda activate /cloud/cloud-ssd1/-/envs/speculative
python third_party/EAGLE/eval.py \
    --base_model ./models/llama-3.1-70b \
    --ea_model ./models/eagle-llama-3.1-70b \
    --eval_task humaneval
```

## 实验结果

实验结果保存在 `results/` 目录下：

- `experiment_comparison.md` - 方法对比
- `experiment_1_fsd_vs_sd.md` - FSD vs 标准推测解码
- `experiment_2_fly.md` - FLy 性能测试

## 开发

### 添加新的解码方法

1. 在 `src/` 下创建新的解码模块
2. 在 `main.py` 中注册新的解码器
3. 更新 `decoding_methods/` 配置

### 运行测试

```bash
# 单元测试
pytest tests/

# 集成测试
python src/tasd_test.py
```
