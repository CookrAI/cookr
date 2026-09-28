# COOKR

Open meme image model. Trained on the memes that moved markets: pump.fun
coin art that actually traded, plus the internet templates it grew out of.

**Site** [cookr.pro](https://cookr.pro) · **Contact** hi@cookr.pro · **Weights** on Hugging Face after v1

## Status

v1 is a LoRA on an Apache-2.0 base (Qwen-Image / Z-Image), merged into a
standalone checkpoint. Dataset collected, captioned, first training run next.

| stage | state |
|---|---|
| pump.fun sweep | 7.4k coins, 4.3k graduated |
| images | 5.4k clean after dedup |
| meme templates | 100 imgflip + KYM galleries |
| captions | VLM, coin name + ticker in every caption |
| LoRA v1 | pending |

## Repos

| repo | what |
|---|---|
| [cookr](https://github.com/CookrAI/cookr) | this: dataset build, captioning, training configs, merge |
| [pumpfun-collector](https://github.com/CookrAI/pumpfun-collector) | pump.fun coins + logos as a dataset, no keys |
| [meme-collector](https://github.com/CookrAI/meme-collector) | imgflip, Reddit, KnowYourMeme, 4chan |
| [x-collector](https://github.com/CookrAI/x-collector) | images from X, keyless tweet mode + API v2 |

## Pipeline

```
fetch_memes      imgflip top-100 templates (+ data/memes.txt for hand picks)
fetch_pumpfun    frontend API, every sort x graduated      -> coins that mattered
stream_pumpfun   PumpPortal websocket, ~40k new coins/day  -> volume
download         IPFS -> RGB, alpha on white, <=1024px, >=256px, phash
dedup            phash near-dup clustering, keep the top coin per cluster
caption          VLM: "cookr, memecoin logo of <name> (<TICKER>), <description>"
export           data/train/*.jpg + *.txt for ai-toolkit
```

One sqlite file (`data/cookr.sqlite`) holds everything. Every stage is
re-runnable and only touches rows it has not seen.

```bash
uv sync
uv run python -m cookr.fetch_memes
uv run python -m cookr.fetch_pumpfun
uv run python -m cookr.stream_pumpfun          # optional, Ctrl-C when enough
uv run python -m cookr.download
uv run python -m cookr.dedup
uv run --extra mlx python -m cookr.caption     # Apple Silicon, Qwen2.5-VL 3B
uv run --extra hf  python -m cookr.caption --backend hf   # CUDA, Florence-2
uv run python -m cookr.export
uv run python -m cookr.stats
```

## Training

| file | use |
|---|---|
| `train/zimage_lora.yaml` | Z-Image 6B, fast iteration, fits a 24 GB card |
| `train/qwen_image_lora.yaml` | Qwen-Image 20B, the release run, rank 128 |
| `train/runpod.sh` | box setup and run (ai-toolkit) |
| `train/merge_lora.py` | bake the LoRA into the base for a standalone checkpoint |

Trigger word `cookr`. Coin name and ticker are in every caption so the model
answers to "memecoin logo of Bonk (BONK)".

## Data notes

- pump.fun's API pages ~1000 deep per sort. The sweep is for quality
  (coins that traded), the websocket is for volume.
- Short side under 256px is dropped, never upscaled. Transparent logos are
  flattened on white. About 18% of pump.fun art is a re-upload; dedup keeps
  the highest-mcap coin per visual cluster so captions carry the name that stuck.
- `export` drops "strategic reserve" / "dividend fund" / AI-agent branding.
  COOKR is a meme model, not a logo generator.
- Meme templates are repeated 8x in the export so 100 templates survive next
  to thousands of logos.

## License

MIT. Weights will ship under Apache-2.0, same as the base.
