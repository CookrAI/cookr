"""Publish a LoRA-only release (Qwen edition). Token from HF_TOKEN or ./.env.
  python train/publish_lora_hf.py --run /workspace/output/cookr_qwen_v1_light --repo CookrAI/cookr-v1-light-qwen \
      --config /workspace/cookr/qwen_image_lora.yaml --card /workspace/cookr/model_card_qwen.md
"""
import argparse
import os
import pathlib
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parent.parent


def token() -> str:
    t = os.environ.get("HF_TOKEN")
    if not t and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                t = line.split("=", 1)[1].strip()
    if not t:
        sys.exit("HF_TOKEN missing")
    return t


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--card", required=True)
a = ap.parse_args()
run = pathlib.Path(a.run)
finals = sorted(p for p in run.glob("*.safetensors") if "_0000" not in p.name)
if not finals:
    sys.exit(f"no final .safetensors in {run}")
api = HfApi(token=token())
api.create_repo(a.repo, repo_type="model", exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    stage = pathlib.Path(td)
    shutil.copy(finals[-1], stage / f"{a.repo.split('/')[-1]}.safetensors")
    shutil.copy(a.config, stage / "train_config.yaml")
    shutil.copy(a.card, stage / "README.md")
    api.upload_folder(repo_id=a.repo, folder_path=str(stage), commit_message="COOKR v1 light, Qwen-Image edition")
print("published https://huggingface.co/" + a.repo)
