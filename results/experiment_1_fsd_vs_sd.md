# 实验结果

## 实验 1: FSD vs 标准推测解码 (2025-05-16)

### 配置
- Target 模型: Qwen2.5-72B-Instruct-AWQ (4-bit)
- Draft 模型: Qwen2.5-7B-Instruct-GPTQ-Int4 (4-bit)
- Prompt: "请详细解释一下什么是量子计算，以及它对未来的影响。"
- Max new tokens: 128
- Gamma: 5 (默认)
- FSD 散度类型: JS 散度
- FSD 散度阈值: 0.4
- 采样模式: 贪心 (do_sample=False)

### 结果

| 方法 | 生成速度 (tokens/s) | 耗时 (s) | 生成 token 数 |
|------|---------------------|----------|---------------|
| FSD (JS 散度) | 7.3 | 17.610 | 128 |
| 标准推测解码 | 4.7 | 26.956 | 128 |

### 加速比
- FSD vs 标准 SD: **1.53x**

### FSD 散度统计
- 散度范围: 0.00003 - 0.46
- 大部分轮次散度都低于 0.4 阈值
- 接受率: 非常高（大部分轮次 5/5 全部接受）

### 结论
FSD (Fuzzy Speculative Decoding) 通过柔性接受机制，允许更多 draft token 被接受，相比标准推测解码有 1.53 倍的加速。
