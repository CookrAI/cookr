"""Publish a trained LoRA (+ samples, config, model card) to the Hub.
Run on the training box or locally. Token from HF_TOKEN or ./.env.

  python train/publish_hf.py --run /workspace/output/cookr_zimage_v1 --repo CookrAI/cookr-v1 \
      --config /workspace/cookr/zimage_lora.yaml --card /workspace/cookr/model_card.md
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="ai-toolkit output folder for the job")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--card", required=True, help="README.md for the model card")
    ap.add_argument("--max-samples", type=int, default=24)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    finals = sorted(p for p in run.glob("*.safetensors") if "_0000" not in p.name)
    if not finals:
        sys.exit(f"no final .safetensors in {run}")
    api = HfApi(token=token())
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
    with tempfile.TemporaryDirectory() as td:
        stage = pathlib.Path(td)
        shutil.copy(finals[-1], stage / f"{a.repo.split('/')[-1]}.safetensors")
        shutil.copy(a.config, stage / "train_config.yaml")
        shutil.copy(a.card, stage / "README.md")
        samples = sorted((run / "samples").glob("*.jpg"))
        # keep the last N samples (latest steps), they are the honest preview
        (stage / "samples").mkdir()
        for p in samples[-a.max_samples:]:
            shutil.copy(p, stage / "samples" / p.name)
        # the model card widget points at samples/sample_N.jpg: the final-step set
        for i, p in enumerate(samples[-4:]):
            shutil.copy(p, stage / "samples" / f"sample_{i}.jpg")
        api.upload_folder(repo_id=a.repo, folder_path=str(stage), commit_message="cookr v1 lora")
    print("published https://huggingface.co/" + a.repo)


if __name__ == "__main__":
    main()
