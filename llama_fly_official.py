"""
Official FLY Integration Wrapper for LLaMA Pilot.

Wraps SPDGenerate from FLy/fly/models/FLy.py into our benchmark pipeline.
Config: draft=1B, target=8B, with official FLY parameters scaled for 8B.
"""
import sys, os, json, time, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import SPDGenerate directly (avoid vllm deps in fly.models.__init__)
import importlib.util
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128

# ── FLY Config (scaled for 8B) ──
# Official 70B: k=15. For 8B: try k=8 (proportional to compute ratio ~8/70 ≈ 0.11)
SPD_ARGS = {
    "k": 8,
    "total_gen_tok": MAX_NEW_TOKENS,
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

def setup_logger(name="fly_wrapper"):
    """Create a simple logger compatible with SPDGenerate's cuslog interface."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger

def official_fly_decode(target_model, draft_model, tokenizer, prompt, spd_args, logger):
    """
    Run official FLY generation on a single prompt.
    
    Args:
        target_model: large target model (8B)
        draft_model: small draft model (1B)
        tokenizer: shared tokenizer
        prompt: string prompt
        spd_args: FLY config dict
        logger: logging.Logger instance
    
    Returns:
        dict with tps, text, gen_len, speedup (vs AR), stats
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    
    # Create FLY generator
    spd_gen = SPDGenerate(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        cuslog=logger,
        spd_args=spd_args,
    )
    
    # Run generation
    torch.cuda.synchronize()
    t0 = time.time()
    full_ids = spd_gen.generate_chunks(input_ids, temperature=0.0)
    torch.cuda.synchronize()
    wall_time = time.time() - t0
    
    gen_ids = full_ids[0][prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    
    # TPS: total tokens (prompt + generated) / wall time
    total_tokens = full_ids.shape[1]
    tps = total_tokens / wall_time if wall_time > 0 else 0
    
    # Stats from spd_gen
    num_accepted = spd_gen.num_accepted_tokens.item() if spd_gen._counter_inited else 0
    num_emitted = spd_gen.num_emitted_tokens.item() if spd_gen._counter_inited else gen_len
    num_rounds = spd_gen.num_draft_round.item() if spd_gen._counter_inited else 0
    
    accept_rate = num_accepted / num_emitted if num_emitted > 0 else 0
    mat = num_emitted / num_rounds if num_rounds > 0 else 0  # Mean Accepted Tokens per round
    
    # FLY-specific stats
    fly_accepted = getattr(spd_gen, 'total_fly_accepted', 0)
    initial_mismatch = getattr(spd_gen, 'total_initial_mismatch', 0)
    fly_recovery_rate = fly_accepted / max(initial_mismatch, 1)
    
    # N-gram stats
    ngram_accepts = getattr(spd_gen, 'debug_ngram_accept_num', [])
    mean_ngram_accept = sum(ngram_accepts) / len(ngram_accepts) if ngram_accepts else 0
    
    return {
        "tps": round(tps, 2),
        "text": gen_text,
        "gen_len": gen_len,
        "time": round(wall_time, 4),
        "total_tokens": total_tokens,
        "accept_rate": round(accept_rate, 4),
        "num_accepted": num_accepted,
        "num_emitted": num_emitted,
        "num_draft_rounds": num_rounds,
        "mat": round(mat, 2),
        "fly_accepted": fly_accepted,
        "initial_mismatch": initial_mismatch,
        "fly_recovery_rate": round(fly_recovery_rate, 4),
        "mean_ngram_accept": round(mean_ngram_accept, 2),
        "speed_list": getattr(spd_gen, 'speed_list', []),
        "mat_list": getattr(spd_gen, 'mat_list', []),
    }

def run_ar(target, tokenizer, prompt):
    """Baseline AR generation."""
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

def compute_sq(pred, ref):
    chars = set("{}[]():,=\n")
    p = [c for c in pred if c in chars]
    r = [c for c in ref if c in chars]
    if not r: return 1.0
    return min(sum(1 for c in p if c in r) / len(r), 1.0)

def main():
    logger = setup_logger()
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Draft on GPU 0, Target on GPU 0 (both fit on single T4)
    # But we need device_map="auto" for both
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print(f"Models loaded. Target device: {target.device}, Draft device: {draft.device}")

    # Load 5 openmmlab samples
    with open("data/ml_config_blocks_openmmlab_80.jsonl") as f:
        samples = [json.loads(l.strip()) for l in f.readlines()[:5]]

    # AR baseline
    print("\nRunning AR baseline...")
    ar_results = []
    for i, s in enumerate(samples):
        r = run_ar(target, tokenizer, s["prompt"])
        ar_results.append(r)
        print(f"  [{i+1}] AR TPS={r['tps']:.1f}")

    print(f"\n{'='*70}")
    print(f"Official FLY: k={SPD_ARGS['k']}, win_len={SPD_ARGS['win_len']}, "
          f"ngram={SPD_ARGS['max_ngram_size']}/{SPD_ARGS['num_ngram_pred_tokens']}, "
          f"entropy_thre={SPD_ARGS['entropy_thre']}, enable_fly={SPD_ARGS['enable_fly']}")
    print(f"{'='*70}")
    print(f"{'Sample':<30} {'AR TPS':>7} {'FLY TPS':>7} {'Speedup':>7} {'SQ':>6} "
          f"{'AccRate':>7} {'MAT':>5} {'ngramAvg':>8} {'FLY_rec':>7} {'Rounds':>6}")
    print("-" * 90)

    fly_results = []
    for i, s in enumerate(samples):
        ref = s.get("reference", "")
        r = official_fly_decode(target, draft, tokenizer, s["prompt"], SPD_ARGS, logger)
        ar_tps = ar_results[i]["tps"]
        sp = r["tps"] / ar_tps if ar_tps > 0 else 0
        sq = compute_sq(r["text"], ref)

        print(f"{s['name']:<30} {ar_tps:>7.1f} {r['tps']:>7.1f} {sp:>7.2f}x {sq:>6.4f} "
              f"{r['accept_rate']:>7.4f} {r['mat']:>5.2f} {r['mean_ngram_accept']:>8.2f} "
              f"{r['fly_recovery_rate']:>7.4f} {r['num_draft_rounds']:>6}")

        fly_results.append({"name": s["name"], "ar_tps": ar_tps, "sp": sp, "sq": sq, **r})

    # Summary
    n = len(fly_results)
    m = lambda k: sum(r[k] for r in fly_results) / n
    print(f"\n--- Official FLY Summary ---")
    print(f"  Mean speedup:     {m('sp'):.3f}x")
    print(f"  Mean SQ:          {m('sq'):.4f}")
    print(f"  Mean accept_rate: {m('accept_rate'):.4f}")
    print(f"  Mean MAT:         {m('mat'):.2f} tokens/round")
    print(f"  Mean ngram accept:{m('mean_ngram_accept'):.2f} tokens/hit")
    print(f"  Mean FLY recovery:{m('fly_recovery_rate'):.4f} (recovered/total_mismatch)")
    print(f"  Mean draft rounds:{m('num_draft_rounds'):.1f}")
    print(f"  Below 1.0x:       {sum(1 for r in fly_results if r['sp']<1.0)}")

    # FLY stats detail
    print(f"\n  FLY recovery detail:")
    for r in fly_results:
        print(f"    {r['name']}: mismatch={r['initial_mismatch']}, "
              f"fly_recovered={r['fly_accepted']}, ngram_accept_per_hit={r['mean_ngram_accept']}")

    print("\nDone!")

if __name__ == "__main__":
    main()
