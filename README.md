# Speculative Decoding 实验

多解码方法对比实验平台：自回归 / 标准推测解码 / FLy / FSD / EAGLE

## 项目结构

```
experiments/
├── src/                          # 核心代码
│   ├── main.py                   # 主入口
│   ├── gpu_utils.py              # GPU 检查工具
│   ├── download_models.py        # 模型下载脚本
│   ├── autoregressive_decode.py  # 自回归解码
│   ├── speculative_decode.py     # 推测解码
│   └── fsd/                      # FSD 模块
│       ├── __init__.py
│       └── fsd_utils.py
├── decoding_methods/             # 各解码方法配置
│   └── fly/                      # FLy 松散推测解码
│       ├── FLy_Qwen2.5_72b.json
│       └── start_fly.sh
├── third_party/                  # 第三方仓库（不提交到 Git）
│   ├── fsd/                      # FSD 框架源码
│   ├── FLy-main/                 # FLy 框架源码
│   └── EAGLE/                    # EAGLE 框架源码
├── envs/                         # Conda 虚拟环境（不提交到 Git）
├── models/                       # 模型权重（不提交到 Git）
├── results/                      # 实验结果（Markdown 日志保留，JSON 不提交）
├── pyproject.toml
└── requirements.txt
```

## 环境配置

### 安装 Miniconda

```bash
# Miniconda 已安装在 /usr/local/miniconda3
source ~/.bashrc
```

### 激活对应环境

```bash
# FSD 框架 (transformers 4.44.0)
conda activate /cloud/cloud-ssd1/-/envs/fsd

# FLy 框架 (transformers 4.57.6)
conda activate /cloud/cloud-ssd1/-/envs/fly

# 自回归基线 (transformers 5.8.1)
conda activate /cloud/cloud-ssd1/-/envs/autoregressive

# 标准推测解码 (transformers 5.8.1)
conda activate /cloud/cloud-ssd1/-/envs/speculative
```

## 模型准备

### Qwen2.5 系列

| 角色 | 模型 | 量化 | 大小 |
|------|------|------|------|
| Target | Qwen2.5-72B-Instruct | 4-bit AWQ | ~36GB |
| Draft | Qwen2.5-7B-Instruct | 4-bit GPTQ | ~3.5GB |

### Llama-3.1 系列

| 角色 | 模型 | 量化 | 大小 |
|------|------|------|------|
| Target | Llama-3.1-70B-Instruct | 4-bit AWQ | ~38GB |
| Draft | Llama-3.1-8B-Instruct | 4-bit AWQ | ~5.4GB |
| EAGLE | EAGLE-Llama-3.1-70B-Instruct | FP16 | ~4GB |

### 下载模型

```bash
conda activate /cloud/cloud-ssd1/-/envs/fly
python src/download_models.py
```

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
conda activate /cloud/cloud-ssd1/-/envs/fly
cd third_party
python llama70b_eagle_test.py
```

## 解码方法对比

| 方法 | 原理 | 优势 | 适用场景 |
|------|------|------|----------|
| 自回归 | 逐 token 生成 | 质量最高 | 基线对比 |
| 标准推测 | Draft 生成 + Target 验证 | 精确匹配加速 | Draft 质量好时 |
| FLy | 语义正确即接受 | 接受率更高 | 宽松场景 |
| FSD | 散度阈值判断 | 可控精度/速度权衡 | 灵活调参 |
| EAGLE | 轻量草稿头复用隐藏层 | 无需独立 Draft 模型 | 显存受限时 |

## 实验结果

详见 [results/](results/) 目录。

### 实验 1: FSD vs 标准推测解码 (Qwen2.5-72B)

| 方法 | 生成速度 (tokens/s) | 耗时 (s) | 加速比 |
|------|---------------------|----------|--------|
| FSD (JS 散度, 阈值 0.4) | 7.3 | 17.6 | 1.53x |
| 标准推测解码 | 4.7 | 27.0 | 1.0x |

### 实验 2: FLy + ngram (Qwen2.5-72B)

| 方法 | 总 token | 总时间 (s) | 速度 (tok/s) | 加速比 |
|------|----------|------------|--------------|--------|
| 自回归 | 2560 | 259.04 | 9.88 | 1.00x |
| 标准推测解码 | 2560 | 170.01 | 15.06 | 1.52x |
| FLy + ngram | 2693 | 162.72 | 16.55 | 1.67x |

### 实验 3: Llama-3.1-70B-Instruct-AWQ 全方法对比

| 方法 | 总 token | 总时间 (s) | 速度 (tok/s) | 加速比 |
|------|----------|------------|--------------|--------|
| 自回归 | 2282 | 225.79 | 10.11 | 1.00x |
| 标准推测解码 | 2282 | 137.40 | 16.61 | 1.64x |
| FLy + ngram | 2456 | 126.14 | 19.47 | 1.93x |
| EAGLE | 2333 | 119.31 | 19.55 | 1.93x |

**关键发现：**
- EAGLE 和 FLy + ngram 并列第一，均达到约 1.93x 加速比
- EAGLE 优势：不需要独立 Draft 模型（草稿头仅 4GB vs Llama-3.1-8B-AWQ 的 5.4GB）
- EAGLE 总显存占用更低：42GB vs 43.4GB

## GPU 使用注意

- 运行前自动检查 GPU 可用性
- 不会强制终止其他用户的进程
- 多进程可共享同一 GPU（显存足够即可）
- 支持多 GPU 模型并行（`--num_gpus` 参数）
