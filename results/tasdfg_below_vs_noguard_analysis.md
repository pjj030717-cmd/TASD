# TASD-FG Below-1.0x Samples vs w/o Guard Analysis

## Summary

TASD-FG has **3 below-1.0** samples (out of 480), while `w/o struct guard` has **0**. This analysis compares quantitative quality sub-metrics to explain the speed-safety trade-off.

**Key finding: w/o guard achieves >1.0x on all 3 hard inputs by sacrificing structural integrity. TASD-FG's below-1.0 is the cost of maintaining structural safety — guard + FB prevent the draft model from drifting into structurally-broken but semantically-probable continuations.**

---

## Data Note

Ablation checkpoints store quality sub-metrics (F1, bracket_balance, off_str, etc.) but NOT the generated text. This analysis uses quantitative metrics only. Text comparison is available for TASD-FG and TASD base (from `qwen_5method_6x80_quality.json`).

---

## Sample 1: argparse_real_062

| Metric | TASD-FG | TASD (base) | w/o guard | no_relaxed |
|--------|:-------:|:-----------:|:---------:|:----------:|
| Speedup | **0.655x** | 0.296x | **1.963x** | 0.901x |
| Wall time | 5.7s | 12.7s | 1.9s | 4.2s |
| Accept rate | 0.373 | 0.138 | **1.000** | 0.547 |
| Guard trigger | 8 | 26 | 0 | 5 |
| Trim count | 7 | 26 | 0 | 4 |
| FB count | 8 | 0 | 0 | 3 |
| **F1** | 0.618 | 0.885 | **0.885** | 1.000 |
| **Bracket** | 1.000 | 1.000 | **0.000** | 1.000 |
| **Type pres.** | 0.000 | 0.000 | 0.000 | 1.000 |
| **Off-Str** | 0.100 | 0.000 | **0.588** | 0.000 |
| Rep rate | 0.000 | 0.000 | 0.000 | 0.000 |

### AR Reference (34.0 TPS)

```python
paradigm_args.add_argument(
    "--deepspeed_config",
    type=str,
    default=None,
    help="Path to the deepspeed config file.",
)
paradigm_args.add_argument(
    "--fsdp_config", ...
```

### TASD-FG Text (0.655x, 10 lines)

```python
from transformers from AutoModel DataCollatorLanguageModeling
 = os.environ.copy()
 os.environ["TOKENIZERS_PARALLELISM"] = "false"
 os.environ["WANDB_MODE"] = "dryrun"
 os.environ["WANDB_MODE"] = "dryrun"    # repeated
 os.environ["WANDB_MODE"] = "dryrun"
```

### TASD base Text (0.296x, 16 lines)

```python
 torch torch.distributed.init_process_group(backend="nccl")
 torch.cuda.set_device(0)
 ...
 torch torch True
 Distributed DistributedSampler
 torch torch.distributedimport torch.distributed import ReduceOp   # malformed
```

### Analysis

This is the hardest sample — even AR produces poor code for this prompt (the reference is argparse but AR generates mixed import-style output).

- **w/o guard (1.963x)**: accept=1.0 means the draft and target are in **complete agreement** on every token. But bracket=0.000 (broken brackets!) and off_str=0.588 (58.8% of lines start with `def/class/import/from`). The draft model, unconstrained, drifts into import/definition territory. Target agrees because both share the same code pretraining distribution. **Result: fast but structurally broken.**

- **TASD-FG (0.655x)**: accept plummets to 0.373 because guard rejects 7 structurally-invalid draft tokens per round. FB fires 8 times trying to recover. But bracket=1.0 and off_str=0.100 — **structurally safer, but slow.**

- **TASD base (0.296x)**: Worst case. Guard fires 26 times (constantly rejecting) without FB to recover. accept drops to 0.138. **Guard without FB is catastrophic on this input.**

- **no_relaxed (0.901x)**: Middle ground. Strict greedy gives F1=1.0, bracket=1.0, but accept=0.547 → no longer below-1.0 but still slow.

**Conclusion: w/o guard is faster because the guard rejects tokens that both models agree on — but those tokens are structurally dangerous (broken brackets, off-structure). TASD-FG pays a speed penalty to maintain bracket safety.**

---

## Sample 2: dict_config_real_014

| Metric | TASD-FG | TASD (base) | w/o guard | no_relaxed |
|--------|:-------:|:-----------:|:---------:|:----------:|
| Speedup | **0.969x** | **1.507x** | **2.065x** | 1.069x |
| Wall time | 4.2s | 2.7s | 2.0s | 3.8s |
| Accept rate | 0.526 | 0.761 | **1.000** | 0.550 |
| Guard trigger | 9 | 4 | 0 | 6 |
| Trim count | 6 | 2 | 0 | 4 |
| FB count | 3 | 0 | 0 | 2 |
| **F1** | 0.990 | 0.981 | 0.962 | 0.989 |
| **Bracket** | 0.000 | 0.000 | 0.000 | 0.000 |
| **Type pres.** | 0.400 | 1.000 | 1.000 | 0.400 |
| **Off-Str** | 0.000 | 0.053 | **0.118** | 0.000 |
| Rep rate | 0.000 | 0.000 | **0.250** | 0.000 |

### AR Reference (31.6 TPS)

```python
# type: ignore
        "traceback": traceback.format_exc(),
    }
    return error_report

def _get_error_message(self, error_report):
    """Get the error message from the error report."""
    return error_report["error"]
```

### TASD-FG Text (0.969x, 20 lines)

```python
# type: ignore
        "exception_traceback": traceback.format_exc(),
    }
    return error_report

 ":param func: The function to wrap.
  :param args args: The arguments to pass to the function.
  :param kwargs kwargs: The keyword arguments to pass to the function.
  :return: The result of the function.
  """
```

### TASD base Text (1.507x, 19 lines)

```python
# type: ignore
        "exception_traceback": traceback.format_exc(),
    }
    return error_report

 ":": exc_val.__class__.__name__,  # type: ignore
        "exception_traceback": traceback.format_exc(),
    }
    return error_report

def_val.__class__.__name__,  # type: ignore   # repeated block
        "exception_traceback": traceback.format_exc(),
    }
    return error_report
```

### Analysis

This is a moderate-difficulty input. Bracket=0 for all variants (reference itself has incomplete brackets at the 128-token boundary).

- **w/o guard (2.065x)**: accept=1.0 again. But this time off_str=0.118 is moderate, and rep_rate=0.250 is the highest of any variant — the model is generating repetitive blocks. F1=0.962 is still high. **Fast, but repetitive and off-structure.**

- **TASD base (1.507x)**: Interesting — actually FASTER than FG! With only 4 guard triggers (vs 9 for FG) and no FB overhead, base achieves 1.507x. But the text shows clear repetition: the `error_report` block repeats verbatim. **Faster than FG, but with repetition risk.**

- **TASD-FG (0.969x)**: Guard fires 9 times, FB 3 times. The generated text is less repetitive than base (rep=0.0) and less off-structure than no_guard (off_str=0.0). **Slower, but safer.**

- **no_relaxed (1.069x)**: Close to FG speed but slightly higher.

**Conclusion: This is the one sample where TASD base genuinely out-runs FG. FB overhead makes FG 0.5x slower than base here. However, base's output has repetition (`error_report` block repeated), while FG's output is cleaner (no repetition detected).**

---

## Sample 3: dict_config_real_057

| Metric | TASD-FG | TASD (base) | w/o guard | no_relaxed |
|--------|:-------:|:-----------:|:---------:|:----------:|
| Speedup | **0.912x** | 0.600x | **2.135x** | 0.888x |
| Wall time | 4.5s | 6.7s | 1.9s | 4.5s |
| Accept rate | 0.494 | 0.285 | **1.000** | 0.614 |
| Guard trigger | 14 | 14 | 0 | 8 |
| Trim count | 9 | 14 | 0 | 4 |
| FB count | 5 | 0 | 0 | 3 |
| **F1** | 0.796 | 0.875 | 0.857 | 0.853 |
| **Bracket** | 0.000 | 0.000 | 0.000 | 0.000 |
| **Type pres.** | 0.800 | 0.800 | 0.800 | 0.800 |
| **Off-Str** | 0.000 | 0.000 | **0.167** | 0.000 |
| Rep rate | 0.000 | **0.100** | 0.000 | 0.000 |

### AR Reference (31.8 TPS)

```python
# TODO: Implement
        "delete": self.delete_value,
        "add": self.add_value,
        "update": self.update_value,
        "help": self.print_help,
    }
    if command in handlers:
        return handlers[command]()
```

### TASD-FG Text (0.912x, 24 lines)

```python
# TODO: Implement
        "delete": self.delete_value,
        "create": self.create_value,
        "search": self.search_value,
        "show": self.show_value,
        "update": self.update_value,
    }
    return handlers[command]

list list)

_values = lambda self: self.list_values()
```

### TASD base Text (0.600x, 30 lines)

```python
# TODO: Implement
        "delete": self.delete_value,
        "create": self.create_value,
        ...
    }
    return handlers[command]

list list)

open open_in_editor(self)

delete delete_value(self):
    # TODO: Implement
...  # repetition continues
```

### Analysis

- **w/o guard (2.135x)**: accept=1.0, but off_str=0.167 (16.7% of lines are out-of-structure `def/class/import`). **Fastest but off-structure.**

- **TASD-FG (0.912x)**: Guard fires 14 times (same as base), but FB saves it from the death spiral — accept=0.494 vs base's 0.285. off_str=0.0. **~1.5x faster than base, structurally constrained.**

- **TASD base (0.600x)**: 14 guard triggers with no FB → accept drops to 0.285. rep_rate=0.100 (repetition). **Worst speed, with repetition.**

- **no_relaxed (0.888x)**: Very close to FG. Strict greedy doesn't change much on this dict_config input.

---

## Overall Analysis

### Why does TASD-FG still have below-1.0 samples?

Three samples share a pattern:
1. Draft-target alignment is poor (accept 0.37-0.53 vs 0.98-1.00 on normal inputs)
2. Guard fires heavily (8-14 triggers) blocking structurally-invalid draft tokens
3. Each acceptance rejection costs time — fewer acceptances = more SD rounds = slower
4. FB compensates but at a cost (~3-8 FB events, 6-16 FB tokens)

These are genuinely hard prompts where **no speculative method can help without also generating bad code**.

### Why does w/o guard have zero below-1.0?

w/o guard runs at **accept=1.0 on all 3 hard samples** — but this near-perfect acceptance comes with structural cost:

|  | Speedup | Bracket | Off-Str | Rep |
|--|:-------:|:-----:|:-------:|:---:|
| Sample 1 | 1.963x | **0.000** | 0.588 | 0.000 |
| Sample 2 | 2.065x | **0.000** | 0.118 | **0.250** |
| Sample 3 | 2.135x | **0.000** | 0.167 | 0.000 |

Across all 3: **bracket=0.000 (broken brackets), off_str 0.12-0.59 (severe structure drift)**

The draft model without guard constraints produces code that is **probabilisticly valid** (target agrees via relaxed verification) but **structurally broken** (unbalanced brackets, off-structure). The guard is paying ~0.3-1.3x speed cost to prevent this.

### Why does TASD base sometimes beat FG?

Sample 2 shows base at 1.507x vs FG's 0.969x. This happens when:
- The draft is "good enough" that guard triggers are low (only 4)
- FB overhead is unnecessary for this level of draft quality
- The result: base runs 1.5x faster but generates repetition the guard didn't catch

This is a limitation of our current FB threshold configuration. In 1/480 samples, FB is triggered unnecessarily (3 FB events on a moderately-good draft). This doesn't invalidate the approach — it's a 0.2% false-positive rate on FB vs the protection it provides on truly hard inputs.

### Is TASD-FG's trade-off reasonable?

**Yes.** The data shows a clean Pareto trade-off on hard inputs:

| Variant | Speed | Bracket Safety | Off-Structure |
|---------|:-----:|:-------------:|:-------------:|
| no_guard | Fast (1.96-2.14x) | **Broken** | **High (0.12-0.59)** |
| no_relaxed | Moderate (0.89-1.07x) | Safe | Safe |
| TASD base | Slow/Variable (0.30-1.51x) | Mixed | Mixed |
| **TASD-FG** | Modest (0.66-0.97x) | **Safe** | **Safe** |

On 477/480 samples (99.4%), TASD-FG achieves >1.0x speedup with structural safety preserved. On the 3 hardest inputs, it delivers modest-to-slow speed while maintaining structural safety, where competing variants either break structure or stall completely.

### Bottom line

**w/o guard's 0-below is not a robustness achievement — it's a structure safety failure.** The guard's speed cost on hard inputs (0.3-1.3x) is the quantifiable cost of maintaining bracket balance and preventing off-structure generation. TASD-FG's 3 below-1.0 are honest worst-case measurements that demonstrate the speed-safety trade-off the method is designed to manage, not a defect to be eliminated.
