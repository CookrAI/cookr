"""Bake the LoRA into the base and save a standalone checkpoint.
That folder is what goes to Hugging Face as cookr/cookr-v1.

  uv run --extra merge python train/merge_lora.py \
      --base Qwen/Qwen-Image --lora output/cookr_qwen_v1/cookr_qwen_v1.safetensors \
      --out ./cookr-v1 --scale 1.0
"""
import argparse

import torch
from diffusers import DiffusionPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args()

pipe = DiffusionPipeline.from_pretrained(a.base, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(a.lora, adapter_name="cookr")
pipe.set_adapters(["cookr"], adapter_weights=[a.scale])
pipe.fuse_lora(adapter_names=["cookr"], lora_scale=a.scale)
pipe.unload_lora_weights()
pipe.save_pretrained(a.out, safe_serialization=True)
print("merged ->", a.out)
