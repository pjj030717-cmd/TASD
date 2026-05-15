# Speculative Decoding 实验

多解码方法对比实验平台：自回归 / 标准推测解码 / FLy / FSD

## 项目结构

```
experiments/
├── src/                          # 核心代码
│   ├── main.py                   # 主入口
│   ├── gpu_utils.py              # GPU 检查工具
│   ├── download_models.py        # 模型下载脚本
│   ├── autoregressive_decode.py  # 自回归解码
│   └── speculative_decode.py     # 推测解码
├── decoding_methods/             # 各解码方法配置
│   ├── fly/                      # FLy 松散推测解码
│   ├── fsd/                      # FSD 模糊推测解码
│   ├── autoregressive/           # 自回归基线
│   └── speculative/              # 标准推测解码
├── configs/                      # 配置文件
│   └── fly/                      # FLy JSON 配置
├── envs/                         # Conda 虚拟环境
│   ├── fly/                      # FLy (transformers 4.57.6)
│   ├── fsd/                      # FSD (transformers 4.44.0)
│   ├── autoregressive/           # 自回归 (transformers 5.8.1)
│   └── speculative/              # 推测解码 (transformers 5.8.1)
├── models/                       # 模型权重（不提交到 Git）
│   ├── Qwen2.5-72B-Instruct-AWQ/ # Target 模型 (4-bit AWQ)
│   └── Qwen2.5-7B-Instruct-GPTQ-Int4/  # Draft 模型 (4-bit GPTQ)
├── third_party/                  # 第三方仓库
│   ├── FLy-main/                 # FLy 框架源码
│   └── fsd/                      # FSD 框架源码
├── pyproject.toml
└── requirements.txt
```

## 环境配置

### 安装 Miniconda

```bash
# Miniconda 已安装在 /data/jjpan/miniconda3
source ~/.bashrc
```

### 激活对应环境

```bash
# FLy 框架
conda activate /data/jjpan/experiments/envs/fly

# FSD 框架
conda activate /data/jjpan/experiments/envs/fsd

# 自回归基线
conda activate /data/jjpan/experiments/envs/autoregressive

# 标准推测解码
conda activate /data/jjpan/experiments/envs/speculative
```

## 模型准备

本项目使用 Qwen2.5 系列量化模型：

| 角色 | 模型 | 量化 | 大小 |
|------|------|------|------|
| Target | Qwen2.5-72B-Instruct | 4-bit AWQ | ~39GB |
| Draft | Qwen2.5-7B-Instruct | 4-bit GPTQ | ~5.3GB |

### 下载模型

```bash
conda activate /data/jjpan/experiments/envs/fly
python src/download_models.py
```

## 运行

### GPU 检查

运行前会自动检查 GPU 占用情况，不会抢占其他同学的进程。

```bash
# 查看 GPU 状态
nvitop

# 或
nvidia-smi
```

### 自回归解码

```bash
conda activate /data/jjpan/experiments/envs/autoregressive
python src/main.py --skip_spec
```

### 标准推测解码

```bash
conda activate /data/jjpan/experiments/envs/speculative
python src/main.py
```

### FLy 松散推测解码

```bash
conda activate /data/jjpan/experiments/envs/fly
export HF_ALLOW_CODE_EVAL=1
fly --model hf \
    --model_args pretrained=./models/Qwen2.5-7B-Instruct-GPTQ-Int4,config_path=configs/fly/FLy_Qwen2.5_72b.json \
    --tasks humaneval_instruct \
    --batch_size 1 \
    --apply_chat_template \
    --confirm_run_unsafe_code
```

### FSD 模糊推测解码

```bash
conda activate /data/jjpan/experiments/envs/fsd
python third_party/fsd/csqa_eval_example.py \
    --small_model_id ./models/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --large_model_id ./models/Qwen2.5-72B-Instruct-AWQ \
    --fsd_div_threshold 0.4 \
    --fsd_div_type "js_div" \
    --num_evals 5
```

## 解码方法对比

| 方法 | 原理 | 优势 | 适用场景 |
|------|------|------|----------|
| 自回归 | 逐 token 生成 | 质量最高 | 基线对比 |
| 标准推测 | Draft 生成 + Target 验证 | 精确匹配加速 | Draft 质量好时 |
| FLy | 语义正确即接受 | 接受率更高 | 宽松场景 |
| FSD | 散度阈值判断 | 可控精度/速度权衡 | 灵活调参 |

## GPU 使用注意

- 运行前自动检查 GPU 可用性
- 不会强制终止其他用户的进程
- 多进程可共享同一 GPU（显存足够即可）
- 建议优先使用 GPU 4（通常最空闲）
