# COOKR

Open meme image model. Trained on what pump.fun actually ships: memecoin logos
that got traction, plus the internet meme templates they rip off.

This repo is the data pipeline + training configs. Weights land on Hugging Face
once v1 is out.

## Pipeline

```
fetch_pumpfun   pump.fun frontend API, every sort x graduated  -> ~6k coins that mattered
stream_pumpfun  PumpPortal websocket, ~40k new coins/day       -> volume, run it for days
fetch_memes     imgflip top-100 templates (+ data/memes.txt)   -> canonical meme frames
download        resolve IPFS -> normalise -> jpg <=1024 + phash
dedup           phash near-dup clustering, keep the top coin per cluster
caption         VLM caption, "cookr, memecoin logo of X (TICK), <desc>"
export          data/train/*.jpg + *.txt for ai-toolkit
```

Everything writes to one sqlite (`data/cookr.sqlite`). Every stage is
re-runnable and only touches rows it has not seen.

```bash
uv sync
uv run python -m cookr.fetch_memes
uv run python -m cookr.fetch_pumpfun
uv run python -m cookr.stream_pumpfun      # leave running, Ctrl-C when you have enough
uv run python -m cookr.download
uv run python -m cookr.dedup
uv run --extra mlx python -m cookr.caption # Mac. CUDA: --extra hf ... --backend hf
uv run python -m cookr.export
uv run python -m cookr.stats
```

## Training

LoRA on an Apache-2.0 base, then merge into a standalone checkpoint.

- `train/zimage_lora.yaml`  Z-Image 6B. Fast iteration, fits a 4090, ~$2/run.
- `train/qwen_image_lora.yaml`  Qwen-Image 20B. The v1 release run.
- `train/runpod.sh`  box setup + run.
- `train/merge_lora.py`  bake LoRA into base for release.

Trigger word: `cookr`. Coin name and ticker are in every caption, so the model
can be prompted by coin.

## Data notes

- pump.fun API pages ~1000 deep per sort. The sweep is for quality (coins that
  traded), the websocket is for volume.
- Short side < 256px is dropped, never upscaled.
- Transparent PNG logos are flattened on white.
- ~30% of pump.fun images are exact re-uploads. Dedup keeps the highest-mcap
  coin per visual cluster so the caption carries the name that stuck.
- Meme templates are repeated 8x in the export so 100 templates survive next
  to 20k logos.
