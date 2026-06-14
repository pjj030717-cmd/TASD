"""
Official FLY Pilot: 3 benchmarks × 20 samples × 2 gamma values (k=8, k=16).

Integrated from FLy/fly/models/FLy.py SPDGenerate class.
"""
import json, os, sys, time, logging
import torch
import importlib.util

# Import SPDGenerate directly (avoids vllm deps)
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 20

BENCHMARKS = [
    ("codesearchnet_dict_config_blocks_80", "dict_config"),
    ("ml_config_blocks_openmmlab_80", "openmmlab_config"),
    ("pipeline_stage_config_80", "pipeline_stage_config"),
]

GAMMAS = [8, 16]

# Official FLY defaults (from FLy_Llama3_70b.json, scaled for 8B)
def make_spd_args(k, max_tokens):
    return {
        "k": k,
        "total_gen_tok": max_tokens,
        "enable_fly": True,
        "win_len": 6,
        "entropy_thre": 0.3,
        "use_ngram": True,
        "max_ngram_size": 4,
        "num_ngram_pred_tokens": 6,
        "verbose": False,
        "abla_no_window": False,
        "enable_statistics": True,
    }

OUT_JSON = "results/llama_official_fly_pilot.json"
OUT_MD = "results/llama_official_fly_pilot.md"

def setup_logger():
    logger = logging.getLogger("fly")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger

def compute_sq(pred, ref):
    chars = set("{}[]():,=\n")
    p = [c for c in pred if c in chars]
    r = [c for c in ref if c in chars]
    if not r: return 1.0
    return min(sum(1 for c in p if c in r) / len(r), 1.0)

def official_fly_decode(target_model, draft_model, tokenizer, prompt, spd_args, logger):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    spd_gen = SPDGenerate(
        draft_model=draft_model, target_model=target_model,
        tokenizer=tokenizer, cuslog=logger, spd_args=spd_args,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    full_ids = spd_gen.generate_chunks(input_ids, temperature=0.0)
    torch.cuda.synchronize()
    wall_time = time.time() - t0

    gen_ids = full_ids[0][prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    total_tokens = full_ids.shape[1]
    tps = total_tokens / wall_time if wall_time > 0 else 0

    num_accepted = spd_gen.num_accepted_tokens.item() if spd_gen._counter_inited else 0
    num_emitted = spd_gen.num_emitted_tokens.item() if spd_gen._counter_inited else gen_len
    num_rounds = spd_gen.num_draft_round.item() if spd_gen._counter_inited else 0
    accept_rate = num_accepted / num_emitted if num_emitted > 0 else 0
    mat = num_emitted / num_rounds if num_rounds > 0 else 0

    fly_accepted = getattr(spd_gen, 'total_fly_accepted', 0)
    initial_mismatch = getattr(spd_gen, 'total_initial_mismatch', 0)
    fly_recovery_rate = fly_accepted / max(initial_mismatch, 1)

    ngram_accepts = getattr(spd_gen, 'debug_ngram_accept_num', [])
    mean_ngram_accept = sum(ngram_accepts) / len(ngram_accepts) if ngram_accepts else 0

    return {
        "tps": round(tps, 2), "text": gen_text, "gen_len": gen_len,
        "time": round(wall_time, 4), "total_tokens": total_tokens,
        "accept_rate": round(accept_rate, 4),
        "mat": round(mat, 2), "num_draft_rounds": num_rounds,
        "fly_recovery_rate": round(fly_recovery_rate, 4),
        "mean_ngram_accept": round(mean_ngram_accept, 2),
    }

def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    wall = time.time() - t0
    gen_ids = out[0][inp.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tps = len(out[0]) / wall if wall > 0 else 0
    return {"tps": tps, "text": text, "time": wall, "gen_len": len(gen_ids)}

def main():
    logger = setup_logger()

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print(f"Models loaded. Target={target.device}, Draft={draft.device}")

    all_results = {}

    for data_file, stype in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"Benchmark: {stype}")
        print(f"{'='*60}")

        with open(f"data/{data_file}.jsonl") as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:SAMPLE_LIMIT]]

        # AR baseline
        print("  Running AR baseline...")
        ar_per_sample = []
        for i, s in enumerate(samples):
            ar = run_ar(target, tokenizer, s["prompt"])
            ar_per_sample.append(ar)
        mean_ar = sum(r["tps"] for r in ar_per_sample) / len(ar_per_sample)
        print(f"  AR mean TPS: {mean_ar:.1f}")

        bench_data = {"ar_tps_mean": round(mean_ar, 1)}

        for k in GAMMAS:
            spd_args = make_spd_args(k, MAX_NEW_TOKENS)
            print(f"\n  --- Official FLY k={k} ---")

            results = []
            for i, s in enumerate(samples):
                ref = s.get("reference", "")
                r = official_fly_decode(target, draft, tokenizer, s["prompt"], spd_args, logger)
                ar_tps = ar_per_sample[i]["tps"]
                sp = r["tps"] / ar_tps if ar_tps > 0 else 0
                sq = compute_sq(r["text"], ref)

                results.append({
                    "name": s.get("name", f"s{i}"),
                    "ar_tps": round(ar_tps, 2),
                    "fly_tps": r["tps"],
                    "fly_speedup": round(sp, 3),
                    "fly_sq": round(sq, 4),
                    "fly_accept_rate": r["accept_rate"],
                    "fly_mat": r["mat"],
                    "fly_ngram_accept": r["mean_ngram_accept"],
                    "fly_recovery_rate": r["fly_recovery_rate"],
                    "fly_draft_rounds": r["num_draft_rounds"],
                })
                print(f"    [{i+1}/{SAMPLE_LIMIT}] {s.get('name','?')[:30]}: "
                      f"sp={sp:.2f}x MAT={r['mat']:.1f} acc={r['accept_rate']:.3f} "
                      f"ngram={r['mean_ngram_accept']:.1f} fly_rec={r['fly_recovery_rate']:.3f}",
                      flush=True)

            bench_data[f"fly_k{k}"] = results

            # Quick summary (compute directly, no lambda closure)
            n = len(results)
            s_sp = round(sum(r["fly_speedup"] for r in results) / n, 3)
            s_mat = round(sum(r["fly_mat"] for r in results) / n, 2)
            s_acc = round(sum(r["fly_accept_rate"] for r in results) / n, 3)
            s_rec = round(sum(r["fly_recovery_rate"] for r in results) / n, 4)
            below = sum(1 for r in results if r["fly_speedup"] < 1.0)
            print(f"    Summary: sp={s_sp}x MAT={s_mat} acc={s_acc} fly_rec={s_rec} below1={below}")

        all_results[stype] = bench_data

    # ── Save JSON ──
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {OUT_JSON}")

    # ── Generate MD Report ──
    with open(OUT_MD, "w") as f:
        f.write("# Official FLY Pilot Report (LLaMA-8B)\n\n")
        f.write("**Target**: meta-llama/Llama-3.1-8B-Instruct\n")
        f.write("**Draft**: meta-llama/Llama-3.2-1B-Instruct\n")
        f.write("**Tokenizer**: Compatible (same vocab=128000)\n\n")
        f.write("### FLY Configuration (Official)\n\n")
        f.write("| Parameter | Value | Notes |\n")
        f.write("|-----------|-------|-------|\n")
        f.write("| enable_fly | True | FLY window recovery |\n")
        f.write("| win_len | 6 | Recovery pattern window |\n")
        f.write("| entropy_thre | 0.3 | Post-verify entropy rejection |\n")
        f.write("| use_ngram | True | Prompt lookup (PLD) |\n")
        f.write("| max_ngram_size | 4 | N-gram match length |\n")
        f.write("| num_ngram_pred_tokens | 6 | Draft tokens from n-gram |\n")
        f.write(f"| gamma (k) | {GAMMAS} | Draft tokens per round |\n")
        f.write(f"| total_gen_tok | {MAX_NEW_TOKENS} | Max new tokens |\n\n")
        f.write("**Source**: FLy/fly/models/FLy.py SPDGenerate class (official AMD implementation)\n\n")

        f.write("**Note**: Official FLY from FLy repo. Compared to our previous implementation:\n")
        f.write("- Always runs n-gram PLD (no entropy gate blocking)\n")
        f.write("- Has FLY window recovery (converts rejected→accepted)\n")
        f.write("- Has post-verify entropy rejection (different direction)\n")
        f.write("- Has n-gram cache fallback (reuses previous n-gram)\n\n")

        for stype in [b[1] for b in BENCHMARKS]:
            data = all_results[stype]
            ar_mean = data["ar_tps_mean"]
            f.write(f"## {stype}\n\n")
            f.write(f"AR TPS: {ar_mean:.1f}\n\n")

            f.write("| Gamma | FLY TPS | Speedup | MAT | AcceptRate | NgramAcc | FLY Rec | Below1.0x |\n")
            f.write("|-------|---------|---------|-----|------------|----------|---------|----------|\n")

            for k in GAMMAS:
                results = data[f"fly_k{k}"]
                n = len(results)
                m2 = lambda key: round(sum(r[key] for r in results) / n, 1) if isinstance(results[0][key], int) else round(sum(r[key] for r in results) / n, 2)
                m3 = lambda key: round(sum(r[key] for r in results) / n, 3)
                m4 = lambda key: round(sum(r[key] for r in results) / n, 4)
                below = sum(1 for r in results if r["fly_speedup"] < 1.0)
                f.write(f"| k={k} | {m2('fly_tps'):.1f} | **{m3('fly_speedup'):.2f}x** | "
                        f"{m2('fly_mat'):.2f} | {m4('fly_accept_rate'):.3f} | "
                        f"{m2('fly_ngram_accept'):.2f} | {m4('fly_recovery_rate'):.4f} | {below} |\n")
            f.write("\n")

        # Overall
        f.write("## Overall Summary\n\n")
        f.write("| Gamma | Speedup | MAT | Accept | FLY Rec | Below1.0x |\n")
        f.write("|-------|---------|-----|--------|---------|----------|\n")
        best_k = (None, -1)
        for k in GAMMAS:
            all_sp, all_mat, all_acc, all_rec = [], [], [], []
            all_below = 0
            for stype in [b[1] for b in BENCHMARKS]:
                rr = all_results[stype][f"fly_k{k}"]
                all_sp.extend(r["fly_speedup"] for r in rr)
                all_mat.extend(r["fly_mat"] for r in rr)
                all_acc.extend(r["fly_accept_rate"] for r in rr)
                all_rec.extend(r["fly_recovery_rate"] for r in rr)
                all_below += sum(1 for r in rr if r["fly_speedup"] < 1.0)
            n2 = len(all_sp)
            msp = sum(all_sp) / n2
            f.write(f"| k={k} | **{msp:.3f}x** | {sum(all_mat)/n2:.2f} | "
                    f"{sum(all_acc)/n2:.3f} | {sum(all_rec)/n2:.4f} | {all_below} |\n")
            if msp > best_k[1]:
                best_k = (k, msp)

        f.write(f"\n**Recommended FLY baseline**: k={best_k[0]} (speedup={best_k[1]:.3f}x)\n")
        f.write(f"\n### Classification\n\n")
        f.write("- **MLA**: NOT enabled (same as paper - MLA not in FLY code)\n")
        f.write("- **PLD**: ENABLED, always active (n-gram + cached fallback)\n")
        f.write("- **FLY recovery**: ENABLED (window=6, converts reject→accept)\n")
        f.write("- **Entropy rejection**: ENABLED (post-verify, τ=0.3)\n")
        f.write("- **Label**: Official FLY (no MLA, ngram+model+fly_recovery)\n")

    print(f"Report saved to {OUT_MD}")
    print("\nDone!")

if __name__ == "__main__":
    main()
