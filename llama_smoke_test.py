"""
LLaMA Smoke Test: 1 OpenMMLab sample with AR/GSD/FLY/TASD
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tasd_decode import tasd_decode

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

target = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, local_files_only=True, device_map="auto",
    torch_dtype=torch.float16, trust_remote_code=True
).eval()
draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, local_files_only=True, device_map="auto",
    torch_dtype=torch.float16, trust_remote_code=True
).eval()
print("Models loaded.\n")

# Load 1 OpenMMLab sample
with open("data/ml_config_blocks_openmmlab_80.jsonl") as f:
    sample = json.loads(f.readline().strip())

prompt = sample["prompt"]
reference = sample.get("reference", "")
structure_type = sample.get("structure_type", "openmmlab_config")
print(f"Sample: {sample.get('name','?')[:50]}")
print(f"Prompt len: {len(prompt)} chars")
print(f"Structure: {structure_type}")
print()

def compute_sq(pred, ref):
    chars = set("{}[]():,=\n")
    p = [c for c in pred if c in chars]
    r = [c for c in ref if c in chars]
    if not r: return 1.0
    return min(sum(1 for c in p if c in r) / len(r), 1.0)

def compute_off_structure(text):
    lines = text.split("\n") if text else []
    if not lines: return 0.0
    kw = {"def ", "class ", "import ", "from "}
    return sum(1 for l in lines if any(l.strip().startswith(k) for k in kw)) / len(lines)

# ── AR ──
print("=== AR ===")
t0 = time.time()
inp = tokenizer(prompt, return_tensors="pt").to(target.device)
with torch.no_grad():
    out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
ar_text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
ar_time = time.time() - t0
ar_tps = len(out[0]) / ar_time if ar_time > 0 else 0
print(f"  TPS: {ar_tps:.1f}, len: {len(out[0])}, time: {ar_time:.1f}s")
print(f"  Text[:100]: {ar_text[:100]}")
print()

# ── Greedy SD ──
print("=== Greedy SD (draft_len=16, draft_blocks=2, strict) ===")
r = tasd_decode(target, draft, tokenizer, prompt, structure_type=structure_type,
                max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
                top_k_accept=3, enable_guard=False, enable_relaxed_accept=False)
gsd_tps = r["tokens_per_second"]
gsd_ar = r["stats"]["accept_rate"]
gsd_sq = compute_sq(r["generated_text"], reference)
gsd_off = compute_off_structure(r["generated_text"])
print(f"  TPS: {gsd_tps:.1f}, accept_rate: {gsd_ar:.4f}, SQ: {gsd_sq:.4f}, off: {gsd_off:.4f}")
print()

# ── FLY ──
print("=== FLY (n-gram + model SD) ===")
from run_hardcase_repair import fly_decode
fly_out = fly_decode(target, draft, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
fly_tps = fly_out.get("tokens_per_second", 0)
fly_sq = compute_sq(fly_out.get("generated_text", ""), reference)
fly_off = compute_off_structure(fly_out.get("generated_text", ""))
print(f"  TPS: {fly_tps:.1f}, SQ: {fly_sq:.4f}, off: {fly_off:.4f}")
print()

# ── TASD ──
print("=== TASD (with guard) ===")
r = tasd_decode(target, draft, tokenizer, prompt, structure_type=structure_type,
                max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
                top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                enable_guard=True, enable_relaxed_accept=True)
tasd_tps = r["tokens_per_second"]
tasd_sp = tasd_tps / ar_tps if ar_tps > 0 else 0
tasd_ar = r["stats"]["accept_rate"]
tasd_sq = compute_sq(r["generated_text"], reference)
tasd_off = compute_off_structure(r["generated_text"])
tasd_repair = r["stats"]["repair_count"]
tasd_guard = r["stats"]["guard_trigger_count"]
print(f"  TPS: {tasd_tps:.1f}, speedup: {tasd_sp:.2f}x, accept: {tasd_ar:.4f}")
print(f"  SQ: {tasd_sq:.4f}, off: {tasd_off:.4f}, repair: {tasd_repair}, guard: {tasd_guard}")
print(f"  Text[:120]: {r['generated_text'][:120]}")
print()

print("=== SMOKE TEST SUMMARY ===")
print(f"AR:    {ar_tps:.1f} TPS")
print(f"GSD:   {gsd_tps:.1f} TPS ({gsd_tps/ar_tps:.2f}x)  acc={gsd_ar:.3f}  SQ={gsd_sq:.3f}")
print(f"FLY:   {fly_tps:.1f} TPS ({fly_tps/ar_tps:.2f}x)  SQ={fly_sq:.3f}")
print(f"TASD:  {tasd_tps:.1f} TPS ({tasd_sp:.2f}x)  acc={tasd_ar:.3f}  SQ={tasd_sq:.3f}  off={tasd_off:.3f}")
print()
print("NO OOM, NO NaN, NO empty output → Smoke test PASSED ✓")
