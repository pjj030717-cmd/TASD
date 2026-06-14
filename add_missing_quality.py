#!/usr/bin/env python3
"""Fill missing SQ/Off-Str for GSD, Ngram, FLY (480 samples each)."""
import json, os, sys, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.ngram_sd_decode import ngram_sd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli"),
]
CHECKPOINT_DIR = "results/qwen_6x80_checkpoints"
METHODS = ["GSD", "Ngram", "FLY"]


def main():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    import importlib.util
    fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
    spec_fly = importlib.util.spec_from_file_location("FLy", fly_path)
    FLy_mod = importlib.util.module_from_spec(spec_fly)
    spec_fly.loader.exec_module(FLy_mod)
    fly_logger = logging.getLogger("FLy")
    fly_logger.setLevel(logging.WARNING)
    FLY_K15 = {"k": 15, "total_gen_tok": MAX_NEW_TOKENS, "enable_fly": True,
               "win_len": 6, "entropy_thre": 0.3, "use_ngram": True,
               "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
               "verbose": False, "abla_no_window": False, "enable_statistics": True}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for bn_key, data_file, stype in BENCHMARKS:
        print(f"\n{'='*50}\n{bn_key} ({stype})\n{'='*50}")
        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:80]]
        n = len(samples)
        prompts = [s["prompt"] for s in samples]
        refs = [s.get("reference", "") for s in samples]

        for method in METHODS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{bn_key}_{method}_quality.json")
            if os.path.exists(ckpt_path):
                print(f"  {method}: already done, skip")
                continue

            print(f"  {method} ({n})...", end=" ", flush=True)
            results = []
            for i in range(n):
                prompt, ref = prompts[i], refs[i]
                if method == "GSD":
                    r = tasd_decode(target, draft, tokenizer, prompt,
                                    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
                                    enable_guard=False, enable_relaxed_accept=False,
                                    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2)
                    text = r["generated_text"]
                elif method == "Ngram":
                    r = ngram_sd_decode(target, tokenizer, prompt,
                                       max_new_tokens=MAX_NEW_TOKENS,
                                       ngram_min=3, ngram_max=8, max_draft_tokens=16)
                    text = r["generated_text"]
                elif method == "FLY":
                    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
                    spd = FLy_mod.SPDGenerate(draft_model=draft, target_model=target,
                                              tokenizer=tokenizer, cuslog=fly_logger, spd_args=FLY_K15)
                    full = spd.generate_chunks(inp.input_ids, temperature=0.0)
                    gids = full[0][inp.input_ids.shape[1]:].tolist()
                    text = tokenizer.decode(gids, skip_special_tokens=True)
                else:
                    continue
                q = compute_composite_sq(text, ref, stype)
                results.append({"name": samples[i]["name"], **q})
                if (i+1) % 20 == 0:
                    sq_m = sum(x["composite_sq"] for x in results)/(i+1)
                    off_m = sum(x.get("off_structure_rate",0) for x in results)/(i+1)
                    print(f"{i+1}(sq={sq_m:.3f},off={off_m:.3f})...", end=" ", flush=True)

            with open(ckpt_path, "w") as fout:
                json.dump(results, fout, indent=2, ensure_ascii=False)
            sq_final = sum(x["composite_sq"] for x in results)/n
            off_final = sum(x.get("off_structure_rate",0) for x in results)/n
            print(f"done sq={sq_final:.4f} off={off_final:.4f}")

    # Merge into final JSON
    with open("results/final_master_report.json") as f:
        final = json.load(f)

    for bn_key, _, _ in BENCHMARKS:
        for method in METHODS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{bn_key}_{method}_quality.json")
            if not os.path.exists(ckpt_path):
                continue
            with open(ckpt_path) as f:
                data = json.load(f)
            if not data:
                continue
            sq_v = sum(x["composite_sq"] for x in data)/len(data)
            off_v = sum(x.get("off_structure_rate",0) for x in data)/len(data)
            if bn_key in final:
                v = final[bn_key].get(method, {})
                v["sq_avg"] = round(sq_v, 4)
                v["off_str"] = round(off_v, 4)
                final[bn_key][method] = v
            print(f"{bn_key}/{method}: sq={sq_v:.4f} off={off_v:.4f}")

    with open("results/final_master_report.json", "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print("\nMerged into final_master_report.json. Re-run generate_final_report.py to update MD.")


if __name__ == "__main__":
    main()
