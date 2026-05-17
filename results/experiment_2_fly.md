# Experiment 2: FLy (Training-Free Loosely Speculative Decoding)

## Configuration
- **Draft Model**: Qwen2.5-7B-Instruct-GPTQ-Int4 (4-bit quantized)
- **Target Model**: Qwen2.5-72B-Instruct-AWQ (4-bit quantized)
- **Task**: HumanEval (20 samples, HumanEval/0 to HumanEval/19)
- **Max New Tokens**: 128
- **Temperature**: 0.0 (greedy decoding)

### FLy Parameters
- `enable_fly`: true
- `use_sd`: true (speculative decoding enabled)
- `k`: 15 (max draft tokens per round)
- `win_len`: 6 (window length for fuzzy matching)
- `entropy_thre`: 0.3 (entropy threshold)
- `use_ngram`: true (ngram matching enabled)
- `max_ngram_size`: 4
- `num_ngram_pred_tokens`: 6

## Results

### Performance Comparison
| 方法 | 总 token | 总时间 (s) | 速度 (tok/s) | 加速比 |
|------|----------|------------|--------------|--------|
| 自回归 (Autoregressive) | 2560 | 259.04 | 9.88 | 1.00x |
| 标准推测解码 (Speculative Decoding) | 2560 | 170.01 | 15.06 | 1.52x |
| FLy + ngram 匹配 | 2693 | 162.72 | 16.55 | 1.67x |

### FLy 详细结果
| 样本 | Token数 | 时间(s) | 速度(tok/s) | MAT |
|------|---------|---------|-------------|-----|
| HumanEval/0 | 139 | 10.70 | 12.99 | 13.71 |
| HumanEval/1 | 137 | 10.87 | 12.60 | 13.33 |
| HumanEval/2 | 132 | 6.84 | 19.31 | 13.69 |
| HumanEval/3 | 132 | 8.61 | 15.34 | 14.17 |
| HumanEval/4 | 133 | 4.63 | 28.71 | 14.89 |
| HumanEval/5 | 134 | 7.00 | 19.15 | 14.08 |
| HumanEval/6 | 135 | 11.24 | 12.01 | 13.73 |
| HumanEval/7 | 137 | 4.37 | 31.36 | 14.67 |
| HumanEval/8 | 133 | 7.83 | 16.99 | 13.77 |
| HumanEval/9 | 139 | 6.39 | 21.76 | 14.18 |
| HumanEval/10 | 132 | 7.64 | 17.29 | 13.85 |
| HumanEval/11 | 131 | 11.96 | 10.95 | 13.40 |
| HumanEval/12 | 133 | 6.53 | 20.38 | 14.70 |
| HumanEval/13 | 134 | 5.97 | 22.43 | 14.70 |
| HumanEval/14 | 131 | 10.46 | 12.52 | 13.57 |
| HumanEval/15 | 133 | 7.58 | 17.54 | 14.25 |
| HumanEval/16 | 141 | 7.62 | 18.49 | 14.25 |
| HumanEval/17 | 135 | 10.67 | 12.65 | 13.21 |
| HumanEval/18 | 137 | 4.86 | 28.20 | 14.20 |
| HumanEval/19 | 135 | 10.95 | 12.33 | 14.33 |
| **平均** | **134.65** | **8.14** | **16.55** | **13.95** |

### 关键发现
- **ngram 匹配效果显著**: MAT 达到 13.95，远高于无 ngram 时的 ~1.5
- **FLy + ngram 加速比 1.67x**: 相比自回归基线 (9.88 tok/s) 提升至 16.55 tok/s
- **标准推测解码加速比 1.52x**: 15.06 tok/s，略低于 FLy
- **FLy 速度波动较大**: 4.37s-11.96s 范围，取决于样本复杂度

## Environment
- **Python**: 3.12
- **PyTorch**: 2.12.0
- **Transformers**: 4.57.6
- **GPTQModel**: 5.7.0
- **AutoAWQ**: 0.2.9
- **GPU**: Single GPU (CUDA)

## Notes
- FLy framework uses semantic correctness to accept draft tokens, enabling looser validation than strict speculative decoding
- The combination of FLy + ngram matching provides robust acceleration
- 4-bit quantization (GPTQ for draft, AWQ for target) enables running 72B model on single GPU
- Results saved to: `results/humaneval_benchmark_40_20260517_100053.json`
