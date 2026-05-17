# 实验结果对比总结

## 实验配置

| 项目 | 值 |
|------|-----|
| Target 模型 | Qwen2.5-72B-Instruct-AWQ (4-bit) |
| Draft 模型 | Qwen2.5-7B-Instruct-GPTQ-Int4 (4-bit) |
| GPU | 单卡 |

---

## 方法对比

### 1. 自回归解码 (Autoregressive Decoding)
- **描述**: 标准的目标模型自回归生成，每个token都需要一次完整的前向传播
- **优点**: 基准方法，无额外开销
- **缺点**: 速度最慢

### 2. 标准推测解码 (Standard Speculative Decoding)
- **描述**: Draft模型生成候选token，Target模型严格验证（完全匹配才接受）
- **生成速度**: **4.7 tokens/s**
- **加速比**: 基准 (1.0x)
- **特点**: 
  - 严格匹配机制，只有完全匹配才接受
  - Gamma=5时，接受率较低

### 3. FSD - Fuzzy Speculative Decoding
- **描述**: 基于分布散度（JS散度）的柔性验证机制，允许散度低于阈值时接受
- **生成速度**: **7.3 tokens/s**
- **加速比**: **1.53x** (vs 标准SD)
- **特点**:
  - 散度阈值: 0.4
  - 散度范围: 0.00003 - 0.46
  - 接受率: 非常高（大部分轮次 5/5 全部接受）
  - 通过柔性接受机制提高draft token接受率

### 4. FLy - Training-Free Loosely Speculative Decoding
- **描述**: 基于语义正确性的松散验证框架，结合ngram匹配
- **pass@1**: **0.872 ± 0.026** (HumanEval-Instruct)
- **生成速度**: **~17.1 tokens/s**
- **加速比**: **~3.6x** (vs 标准SD)
- **特点**:
  - spd_k=15 (每轮最多15个draft token)
  - win_len=6 (模糊匹配窗口)
  - entropy_thre=0.3 (熵阈值)
  - ngram匹配启用 (max_ngram_size=4)
  - MAT (平均接受时间): ~11.5
  - ngramMAT: ~1.42
  - Draft轮数: 13-54轮/样本

---

## 性能对比图

```
生成速度 (tokens/s)
  |
20|                                   
  |                                   
15|                           █ FLy  
  |                           | 17.1 
10|                           |       
  |                  █ FSD    |       
 5|         █ SD     | 7.3    |       
  |         | 4.7    |        |       
 0|_________|________|________|_______
         标准SD    FSD      FLy
```

## 加速比对比

```
加速比 (vs 标准SD)
  |
4x|                           █ FLy  
  |                           | 3.6x 
3x|                           |       
  |                           |       
2x|                  █ FSD    |       
  |         基准   | 1.53x   |       
1x|_______█ SD_____|_________|_______
         1.0x    FSD      FLy
```

---

## 关键发现

1. **FLy 性能最优**: ~17.1 tokens/s，是标准SD的3.6倍，FSD的2.3倍
2. **FSD 优于标准SD**: 7.3 tokens/s，比标准SD快53%
3. **FLy + ngram 组合**: ngram匹配贡献了额外的接受率（ngramMAT ~1.42）
4. **量化模型可行**: 4-bit量化（GPTQ+AWQ）使72B模型能在单卡上运行

---

## 实验结果文件

| 实验 | 结果文件 |
|------|----------|
| FSD vs SD | `results/experiment_1_fsd_vs_sd.md` |
| FLy | `results/fly_results.jsonl/.__models__Qwen2.5-7B-Instruct-GPTQ-Int4/results_2026-05-16T10-32-43.975028.json` |

---

## 环境信息

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.12.0 |
| Transformers | 4.57.6 |
| GPTQModel | 5.7.0 |
| AutoAWQ | 0.2.9 |
