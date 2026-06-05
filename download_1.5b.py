#!/usr/bin/env python3
"""Download Qwen2.5-1.5B-Instruct via modelscope"""
from modelscope import snapshot_download
import os

cache_dir = "/root/autodl-tmp/models/.modelscope_cache"
os.makedirs(cache_dir, exist_ok=True)

print("Downloading Qwen2.5-1.5B-Instruct...")
snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir=cache_dir,
)
print("Done.")
