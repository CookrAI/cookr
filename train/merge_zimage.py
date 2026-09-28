"""Bake the COOKR LoRA into Z-Image-Turbo's transformer and write a full,
standalone model: diffusers folder (drop-in for ZImagePipeline) plus a single
transformer .safetensors for ComfyUI / single-file loaders.

ai-toolkit saves keys as diffusion_model.<module>.lora_A.weight /
lora_B.weight (no alpha tensors: alpha == rank, scale 1), so this does the
merge by hand instead of trusting a loader to guess the mapping.

  python train/merge_zimage.py --base Tongyi-MAI/Z-Image-Turbo \
      --lora /workspace/output/cookr_zimage_v1/cookr_zimage_v1.safetensors \
      --out /workspace/cookr-v1-light --scale 1.0
"""
import argparse
import json
import pathlib
import shutil

import torch
from diffusers import ZImageTransformer2DModel
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Tongyi-MAI/Z-Image-Turbo")
ap.add_argument("--lora", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args()

out = pathlib.Path(a.out)
base_dir = pathlib.Path(snapshot_download(a.base))
print("base snapshot:", base_dir)

print("loading transformer (bf16, cpu)")
tf = ZImageTransformer2DModel.from_pretrained(str(base_dir), subfolder="transformer", torch_dtype=torch.bfloat16)
params = dict(tf.named_parameters())

lora = load_file(a.lora)
pairs = {}
for k, v in lora.items():
    if not k.startswith("diffusion_model."):
        raise SystemExit(f"unexpected key prefix: {k}")
    mod, which = k[len("diffusion_model."):].rsplit(".lora_", 1)
    pairs.setdefault(mod, {})[which.split(".")[0]] = v
print(f"{len(pairs)} lora modules")

merged = missing = 0
with torch.no_grad():
    for mod, ab in pairs.items():
        wkey = f"{mod}.weight"
        if wkey not in params:
            missing += 1
            print("  no target for", mod)
            continue
        A, B = ab["A"].float(), ab["B"].float()
        delta = (B @ A) * a.scale  # alpha == rank in ai-toolkit exports -> scale 1
        w = params[wkey]
        w.add_(delta.to(w.dtype))
        merged += 1
print(f"merged {merged}, missing {missing}")

if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
tf.save_pretrained(out / "transformer", safe_serialization=True)
for sub in ("text_encoder", "tokenizer", "vae", "scheduler"):
    src = base_dir / sub
    if src.exists():
        shutil.copytree(src, out / sub, symlinks=False)
mi = base_dir / "model_index.json"
if mi.exists():
    shutil.copy(mi, out / "model_index.json")
# single-file transformer for ComfyUI-style loaders
sd = {k: v.contiguous() for k, v in tf.state_dict().items()}
save_file(sd, str(out / "cookr-v1-light-transformer.safetensors"), metadata={"format": "pt", "model": "COOKR v1 light", "base": a.base})
print("wrote", out)
