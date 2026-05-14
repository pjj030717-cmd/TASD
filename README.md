# Speculative Decoding 实验

标准自回归 vs 标准推测解码的实现与对比测试。

## 项目结构

```
/root/data/
├── autoregressive_decode.py   # 标准自回归解码
├── speculative_decode.py      # 标准推测解码
├── main.py                    # 主测试脚本
└── requirements.txt           # 依赖
```

## 环境要求

- Python 3.10+
- CUDA 12.x
- NVIDIA GPU (建议 32GB+ VRAM)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 模型准备

本项目使用 Qwen2.5 系列，Target 和 Draft 共享同一 tokenizer（同系列天然共享 vocab）。

### 通过 ModelScope 下载

```python
from modelscope import snapshot_download

# Target: 14B
snapshot_download('Qwen/Qwen2.5-14B-Instruct', cache_dir='/root/autodl-tmp/models')

# Target: 7B
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/data/models')

# Draft: 1.5B
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/root/autodl-tmp/models')

# Draft: 0.5B
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='/root/autodl-tmp/models')
```

## 运行

```bash
# 默认参数: Target 7B + Draft 1.5B, 贪心模式, gamma=5, 128 tokens
python main.py

# 自定义 Target / Draft
python main.py --target_model /path/to/target --draft_model /path/to/draft

# 自定义参数
python main.py --gamma 3 --max_new_tokens 256

# 只跑自回归
python main.py --skip_spec

# 只跑推测解码
python main.py --skip_ar

# 采样模式
python main.py --temperature 0.8 --top_p 0.9
```

## 算法说明

### 标准自回归解码

每一步生成一个 token，拼接到输入序列后继续下一轮：

```python
for step in range(max_new_tokens):
    logits = model(input_ids + past_key_values)
    next_token = argmax(logits)
    input_ids = concat(input_ids, next_token)
```

### 标准推测解码 (Speculative Decoding)

1. **Draft 阶段**: 小模型自回归生成 gamma 个候选 token
2. **Verify 阶段**: 大模型一次前向，得到每个候选位置的 logits
3. **Accept/Reject**:
   - 贪心模式: Target 预测 == Draft token → 接受，否则用 Target 预测替换
   - 采样模式: 基于 p_target / p_draft 概率比决定；拒绝时从 max(0, p_target - p_draft) 修正分布采样

## 实验结果

测试环境: NVIDIA RTX 5090 (31.4GB VRAM)

| # | Target | Draft | 接受率 | 自回归速度 | 推测速度 | 实际加速 |
|---|--------|-------|:------:|:----------:|:--------:|:--------:|
| 1 | 14B | 1.5B | 98.0% | 0.6 tok/s | 2.8 tok/s | **4.81x** |
| 2 | 7B | 0.5B | 39.7% | 57.0 tok/s | 35.9 tok/s | 0.63x |
| 3 | 7B | 1.5B | 52.9% | 57.2 tok/s | 37.6 tok/s | 0.66x |

### 关键发现

- 推测解码的有效性取决于 **Draft 比 Target 快多少** 以及 **接受率多高**
- 14B + 1.5B 接受率 98% 且速度差足够大（14B 被 offload 到 CPU），加速 4.81x
- 7B 在 RTX 5090 上自回归已经很快（~57 tok/s），Draft 额外开销反而拖慢整体
- Draft 不能太小：0.5B 对 7B 接受率仅 40%，远不如 1.5B 的 53%
- 推测解码在 A100 80GB 等大显存卡上效果更好（14B 可全放 GPU，加速比依然可观）
