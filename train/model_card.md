---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
  - text-to-image
  - diffusers
  - meme
  - memecoin
  - pump.fun
  - cookr
pipeline_tag: text-to-image
library_name: diffusers
---

# COOKR v1 light

**Cook a meme. Launch a token.**
COOKR is the meme image model behind [cookr.pro](https://cookr.pro): an AI
platform that turns an idea into a memecoin character and keeps it alive
after launch. This is the **light** release, a full standalone checkpoint you
can run on one GPU today. The full COOKR model, trained on millions of memes
and memecoin images, is served on cookr.pro.

- Trained on the memes that moved markets: pump.fun coin art that actually traded, plus the internet templates it grew out of
- Native prompt grammar for coins: character, ticker, format, vibe
- 9 steps, 1024px, no CFG. A meme in about a second on an L40S

[cookr.pro](https://cookr.pro) · [GitHub](https://github.com/CookrAI/cookr) · [X @CookrPro](https://x.com/CookrPro) · hi@cookr.pro

## Quickstart

```bash
pip install "cookr[infer] @ git+https://github.com/CookrAI/cookr"
```

```python
from cookr import Cookr

c = Cookr()                                                  # loads CookrAI/cookr-v1-light
c.cook("a frog chef who trades charts", ticker="RIBBIT").save("ribbit.png")
logo, sticker, meme, alt = c.directions("a pigeon who owns wall street", ticker="PIGEON")
```

Plain diffusers works too:

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained("CookrAI/cookr-v1-light", torch_dtype=torch.bfloat16).to("cuda")
img = pipe("cookr, memecoin logo of Pepe Astronaut (PEPENAUT), pepe the frog in a spacesuit on the moon, green and black",
           num_inference_steps=9, guidance_scale=1.0, height=1024, width=1024).images[0]
```

ComfyUI: load `cookr-v1-light-transformer.safetensors` as the Z-Image-Turbo
diffusion model in any Z-Image workflow. Text encoder and VAE are unchanged.

## Prompt grammar

The model was trained on captions of one shape, and it answers best to that shape:

```
cookr, <format> of <character> (<TICKER>), <details>, <style>, <colors>
```

| slot | values |
|---|---|
| format | `memecoin logo`, `die-cut sticker`, `meme template` |
| character | who it is, one clause: "a pigeon who owns wall street" |
| ticker | the symbol in parentheses, uppercase |
| style | `cartoon`, `pixel art`, `3d render`, `mspaint style`, `photo` |

Examples that work:

- `cookr, memecoin logo of Bonk Dog (BONK), cartoon shiba inu wearing sunglasses, orange and yellow, bold outline`
- `cookr, die-cut sticker of Chef Ribbit (RIBBIT), frog in a chef hat burning a steak, flat colors`
- `cookr, wojak crying in front of a red candle chart, mspaint style`
- `cookr, meme template, distracted boyfriend, three people on a street, photo`

Keep `cookr` as the first token. Weight lives in the checkpoint, no LoRA loader needed.

## What is in the repo

| file | what |
|---|---|
| `transformer/` `text_encoder/` `vae/` `tokenizer/` `scheduler/` `model_index.json` | full diffusers model, load with `ZImagePipeline` |
| `cookr-v1-light-transformer.safetensors` | single-file transformer for ComfyUI |
| `cookr-v1-light-lora.safetensors` | the LoRA on its own, if you want to stack it |
| `train_config.yaml` | the exact ai-toolkit config |

## Training

| | |
|---|---|
| base | Z-Image-Turbo (Apache-2.0), trained through ai-toolkit's `zimage:turbo` de-distill adapter, then merged |
| data | 5.3k pump.fun coin logos (graduated / traded coins first, phash-deduped) + 98 meme templates repeated 8x |
| captions | Qwen2.5-VL, coin name + ticker prepended |
| network | LoRA rank 96, alpha 96, merged at scale 1.0 |
| schedule | 5000 steps, batch 2, lr 1e-4, adamw8bit, bf16, EMA 0.99 |
| resolutions | 512 / 768 / 1024 buckets |
| hardware | 1x L40S, 2.5 h |

Data pipeline, collectors and this config are public:
[cookr](https://github.com/CookrAI/cookr) ·
[pumpfun-collector](https://github.com/CookrAI/pumpfun-collector) ·
[meme-collector](https://github.com/CookrAI/meme-collector) ·
[x-collector](https://github.com/CookrAI/x-collector)

## Light vs full

| | light (this) | full |
|---|---|---|
| data | ~6k curated images | millions of memes and memecoin images |
| training | LoRA merged into an open base | end to end |
| where | here, run it yourself | [cookr.pro](https://cookr.pro) and the Cookr API |

## License

Apache-2.0, same as the base model. Training images remain the property of
whoever made them.
