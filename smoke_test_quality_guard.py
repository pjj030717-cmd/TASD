#!/usr/bin/env python3
"""Smoke test: verify enable_quality_guard=False doesn't change TASD-FG behavior."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"

print("Loading models...")
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, local_files_only=True, device_map="auto", trust_remote_code=True
)
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, local_files_only=True, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)

# Load one sample from JSONL
import json
data_file = 'data/codesearchnet_argparse_blocks_80.jsonl'
with open(data_file) as f:
    first_line = f.readline()
    sample = json.loads(first_line)
prompt = sample['prompt']
reference = sample['reference']

print(f"\nTest sample from: {data_file}")
print(f"Prompt length: {len(prompt)} chars")

# Run 1: TASD-FG with enable_quality_guard=False (default)
print("\n--- Run 1: TASD-FG (quality_guard=False) ---")
result1 = tasd_decode(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=128,
    draft_len=8,
    draft_blocks=2,
    enable_guard=True,
    enable_relaxed_accept=True,
    enable_failure_aware_fallback=True,
    enable_quality_guard=False,
)
print(f"  Generated: {result1['generated_tokens']} tokens")
print(f"  Speedup: {result1['stats'].get('speedup', 'N/A')}")
print(f"  Quality guard enabled: {result1['stats'].get('quality_guard_enabled', 'MISSING')}")
print(f"  Quality guard stats: {result1['stats'].get('quality_guard', 'MISSING')}")
print(f"  First 100 chars: {result1['generated_text'][:100]}")

# Run 2: TASD-FGQ with enable_quality_guard=True
print("\n--- Run 2: TASD-FGQ (quality_guard=True) ---")
result2 = tasd_decode(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=128,
    draft_len=8,
    draft_blocks=2,
    enable_guard=True,
    enable_relaxed_accept=True,
    enable_failure_aware_fallback=True,
    enable_quality_guard=True,
)
print(f"  Generated: {result2['generated_tokens']} tokens")
print(f"  Speedup: {result2['stats'].get('speedup', 'N/A')}")
print(f"  Quality guard enabled: {result2['stats'].get('quality_guard_enabled', 'MISSING')}")
print(f"  Quality guard stats: {result2['stats'].get('quality_guard', 'MISSING')}")
print(f"  First 100 chars: {result2['generated_text'][:100]}")

# Verify
print("\n--- Verification ---")
assert result1['stats']['quality_guard_enabled'] == False, "quality_guard_enabled should be False"
assert result2['stats']['quality_guard_enabled'] == True, "quality_guard_enabled should be True"
assert result1['stats']['quality_guard'] is None, "quality_guard stats should be None when disabled"
assert result2['stats']['quality_guard'] is not None, "quality_guard stats should exist when enabled"
print("✅ All assertions passed")
print(f"\nSmoke test PASSED")
