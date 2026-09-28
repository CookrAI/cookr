"""Publish COOKR to the Hub: merged full model folder (from merge_zimage.py) +
standalone LoRA + samples + grid + config + model card.
Run on the training box. Token from HF_TOKEN or ./.env.

  python train/publish_hf.py --run /workspace/output/cookr_zimage_v1 --merged /workspace/cookr-v1-light \
      --repo CookrAI/cookr-v1-light --config /workspace/cookr/zimage_lora.yaml --card /workspace/cookr/model_card.md
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


def grid(paths, out, size=512):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    for im in ims:
        im.thumbnail((size, size))
    sheet = Image.new("RGB", (size * len(ims), size), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * size, 0))
    sheet.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="ai-toolkit output folder for the job")
    ap.add_argument("--merged", required=True, help="output folder of merge_zimage.py")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--card", required=True, help="README.md for the model card")
    ap.add_argument("--max-samples", type=int, default=24)
    ap.add_argument("--samples", action="store_true", help="also upload training samples (off by default)")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    finals = sorted(p for p in run.glob("*.safetensors") if "_0000" not in p.name)
    if not finals:
        sys.exit(f"no final .safetensors in {run}")
    api = HfApi(token=token())
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
    merged = pathlib.Path(a.merged)
    name = a.repo.split("/")[-1]
    # small files staged next to the merged folder; the big folder uploads in place
    shutil.copy(finals[-1], merged / f"{name}-lora.safetensors")
    shutil.copy(a.config, merged / "train_config.yaml")
    shutil.copy(a.card, merged / "README.md")
    sdir = merged / "samples"
    if sdir.exists():
        shutil.rmtree(sdir)
    if a.samples:
        samples = sorted((run / "samples").glob("*.jpg"))
        sdir.mkdir()
        for p in samples[-a.max_samples:]:
            shutil.copy(p, sdir / p.name)
        for i, p in enumerate(samples[-4:]):
            shutil.copy(p, sdir / f"sample_{i}.jpg")
        grid(samples[-4:], sdir / "grid.jpg")
    api.upload_folder(repo_id=a.repo, folder_path=str(merged), commit_message="COOKR v1 light",
                      ignore_patterns=["*.tmp", "__pycache__"])
    print("published https://huggingface.co/" + a.repo)


if __name__ == "__main__":
    main()
