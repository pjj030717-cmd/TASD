# TASD 7B+3B 复刻实验日志

日期：2026-06-03

目的：在存储盘异常或迁移环境后，复刻 TASD 的轻量部署实验。复刻重点是三个已经验证成功的 benchmark，每个 80 samples，四方法对比，并将模型搭配设为 Target=7B Instruct、Draft=3B Instruct。

## 1. 实验目标

复刻 small-target setting：

- Benchmark: argparse, dict_config, openmmlab
- Sample count: 80 each
- Methods: AR, Vanilla SD, FLY, TASD
- Target model: Qwen2.5-7B-Instruct
- Draft model: Qwen2.5-3B-Instruct
- temperature: 0.0
- max_new_tokens: 128

本轮目的不是继续扩展新结构，而是验证：

1. 3B draft 相对 7B target 是否足够便宜。
2. 7B+3B 是否能保持 TASD 相对 7B AR 的结构质量。
3. 7B+3B 在三个成功 benchmark 上的 speedup、accept rate、draft cost 是否更好。

## 2. 需要复刻的关键目录

建议新环境仍使用：

```text
/root/data/my-repo
```

如果新盘路径不同，先统一替换脚本中的旧路径。

必须复刻：

```text
/root/data/my-repo/src/
/root/data/my-repo/FLy-main/
/root/data/my-repo/data/
/root/data/my-repo/results/
```

其中 `results/` 不一定全部复刻，但历史 14B 主实验结果和 7B 相关结果最好保留，用于对比。

## 3. 必须保留的数据文件

三个 80-sample benchmark：

```text
data/codesearchnet_argparse_blocks_80.jsonl
data/codesearchnet_argparse_blocks_80_summary.json

data/codesearchnet_dict_config_blocks_80.jsonl
data/codesearchnet_dict_config_blocks_80_summary.json

data/ml_config_blocks_openmmlab_80.jsonl
data/ml_config_blocks_openmmlab_80_summary.json
```

如果文件名变化，按内容对应：

```text
argparse        -> argparse 参数列表 benchmark
dict_config     -> 通用 dict/list config benchmark
openmmlab       -> OpenMMLab 风格配置 benchmark
```

## 4. 必须保留的代码文件

核心解码与质量评估：

```text
src/tasd_decode.py
src/vanilla_speculative_decode.py
src/fly_official_decode.py
src/structural_guard.py
src/autoregressive_decode.py
```

实验 runner / 汇总脚本：

```text
run_baseline_comparison.py
run_tasd_structural_guard_validation.py
run_small_sample_validation.py
run_structural_quality_eval.py
compute_full_quality_80.py
generate_main_results_summary.py
generate_ablation_study.py
```

FLY 官方实现目录：

```text
FLy-main/
```

## 5. 模型文件

本轮复刻需要：

```text
models/Qwen2.5-7B-Instruct
models/Qwen2.5-3B-Instruct
```

可选保留历史对照：

```text
models/Qwen2.5-14B-Instruct
models/Qwen2.5-7B-Instruct
```

## 6. Tokenizer 检查

在跑 TASD 前必须检查 7B Instruct 和 3B Instruct 的 tokenizer 是否一致。

```bash
cd /root/data/my-repo

python - <<'PY'
from transformers import AutoTokenizer
import hashlib, json

models = [
    "/root/data/my-repo/models/Qwen2.5-7B-Instruct",
    "/root/data/my-repo/models/Qwen2.5-3B-Instruct",
]

tokenizers = []
for m in models:
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    vocab = tok.get_vocab()
    print(m)
    print("vocab_size:", len(vocab))
    print("pad_token:", tok.pad_token, tok.pad_token_id)
    print("eos_token:", tok.eos_token, tok.eos_token_id)
    print("bos_token:", tok.bos_token, tok.bos_token_id)
    h = hashlib.md5(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
    print("vocab_md5:", h)
    print()
    tokenizers.append(tok)

same_vocab = tokenizers[0].get_vocab() == tokenizers[1].get_vocab()
print("same_vocab:", same_vocab)

test = "def forward(self, x):\n    return self.layer(x)"
ids0 = tokenizers[0].encode(test)
ids1 = tokenizers[1].encode(test)
print("same_encoding:", ids0 == ids1)
print("ids0[:20]:", ids0[:20])
print("ids1[:20]:", ids1[:20])
PY
```

如果不是：

```text
same_vocab: True
same_encoding: True
```

立即停止，不要跑 TASD。

## 7. 四方法主实验

需要在三个 benchmark 上分别跑：

```text
AR
Vanilla SD
FLY
TASD
```

统一参数：

```text
temperature=0.0
max_new_tokens=128
sample_limit=80
target_model=Qwen2.5-7B-Instruct
draft_model=Qwen2.5-3B-Instruct
```

建议输出文件：

```text
results/baseline_compare_argparse_80_draft3b.json
results/baseline_compare_dict_config_80_draft3b.json
results/baseline_compare_openmmlab_80_draft3b.json

results/tasd_structural_guard_policy_argparse_80_draft3b.json
results/tasd_structural_guard_policy_dict_config_80_draft3b.json
results/tasd_structural_guard_policy_openmmlab_80_draft3b.json

results/full_structural_quality_argparse_80_draft3b.json
results/full_structural_quality_dict_config_80_draft3b.json
results/full_structural_quality_openmmlab_80_draft3b.json

results/main_experiment_results_summary_draft3b.md
```

不要覆盖旧结果，尤其不要覆盖 14B+7B 主实验结果。

## 8. 推荐执行顺序

先跑每个 benchmark 10 samples smoke test：

```text
argparse 10
dict_config 10
openmmlab 10
```

确认没有 OOM、路径错误、tokenizer 错误、FLY import 错误后，再跑 80 samples。

正式顺序：

1. argparse 80
2. openmmlab 80
3. dict_config 80

原因：

- argparse 和 openmmlab 是相对稳定的成功 benchmark。
- dict_config 结构风险更高，最需要看 Guard 效果，但也最容易暴露问题。

## 9. 每个 benchmark 必须记录的指标

速度：

```text
AR tokens/sec
Vanilla SD speedup
FLY speedup
TASD speedup
TASD tokens/sec
```

TASD 内部：

```text
accept_rate
target_model_forwards
draft_model_forwards
draft_time_share
forwards_per_token
generated_length
```

结构质量：

```text
structural_quality_score
severe_rate
off_structure_rate
repetition_rate
truncation_rate
```

Guard：

```text
guard_trigger_count
trim_count
repair_count
avg_triggers_per_sample
```

## 10. 需要和 7B AR / 历史 14B 主实验对照的问题

最终 summary 必须回答：

1. 7B+3B 相对 7B AR 是否有稳定 speedup。
2. 7B+3B 的 draft_time_share 是否显著低于 14B+7B。
3. 7B+3B 的 accept_rate 是否能保持在可用范围。
4. 7B+3B 的 structural_quality_score 是否接近或优于 7B AR。
5. 7B+3B 是否适合作为 small-target deployment setting，而不是替代 14B 主配置。

推荐结论格式：

```text
如果 7B+3B 速度更快且质量不明显下降：
    将 7B+3B 作为轻量部署配置候选。

如果 7B+3B 速度更快但质量明显下降：
    保留为速度-质量权衡消融。

如果 7B+3B accept_rate 下降过多，速度也没有提升：
    7B+3B 只作为失败的 small-target 消融记录。
```

## 11. 不要做的事

```text
不要修改 TASD 核心算法。
不要覆盖旧 14B+7B 主实验结果。
不要继续扩展新 benchmark。
不要把 pytest_parametrize / schema_fields / model_fields 放入主表。
不要把 TASD w/o Guard 放入主对比表。
```

主表只放：

```text
AR
Vanilla SD
FLY
TASD
```

其中 TASD = 带 Guard policy 的当前主方法。

TASD w/o Guard 只用于消融实验。

## 12. 复刻完成后的主表格式

速度表：

| Benchmark | AR TPS | Vanilla SD | FLY | TASD |
|---|---:|---:|---:|---:|
| argparse | | | | |
| dict_config | | | | |
| openmmlab | | | | |

质量表：

| Benchmark | AR | Vanilla SD | FLY | TASD |
|---|---:|---:|---:|---:|
| argparse | | | | |
| dict_config | | | | |
| openmmlab | | | | |

内部诊断表：

| Benchmark | accept_rate | draft_time_share | target_fw | draft_fw | guard_triggers |
|---|---:|---:|---:|---:|---:|
| argparse | | | | | |
| dict_config | | | | | |
| openmmlab | | | | | |


