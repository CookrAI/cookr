---
license: apache-2.0
base_model: Qwen/Qwen-Image
tags:
  - lora
  - text-to-image
  - diffusers
  - qwen-image
  - meme
  - memecoin
  - pump.fun
  - cookr
pipeline_tag: text-to-image
library_name: diffusers
instance_prompt: cookr
---

# COOKR v1 light, Qwen-Image edition

**Cook a meme. Launch a token.**
The same COOKR training set, learned on top of [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
instead of Z-Image-Turbo. Pick this one when the meme needs **text**: tickers on
the coin, caption bars, "TRUST THE RECIPE" on a cat. Qwen-Image renders words,
Z-Image-Turbo mostly does not.

This is a LoRA (rank 128) for Qwen-Image. The Z-Image edition ships as a full
standalone checkpoint: [CookrAI/cookr-v1-light](https://huggingface.co/CookrAI/cookr-v1-light).

[cookr.pro](https://cookr.pro) · [GitHub](https://github.com/CookrAI/cookr) · [X @CookrPro](https://x.com/CookrPro) · hi@cookr.pro

## Use

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("CookrAI/cookr-v1-light-qwen")

img = pipe(
    "cookr, memecoin logo of Bonk Dog (BONK), cartoon shiba inu wearing sunglasses, the word BONK in bold letters, orange and yellow",
    num_inference_steps=30, true_cfg_scale=4.0, height=1024, width=1024,
).images[0]
```

Qwen-Image is 20B: bf16 needs ~40 GB, or use an 8-bit/4-bit quantized transformer on a 24 GB card.
LoRA weight 0.8-1.0. 25-40 steps, CFG 3-5.

## Prompt grammar

```
cookr, <format> of <character> (<TICKER>), <details>, the word <TEXT> in bold letters, <style>, <colors>
```

Same grammar as the Z-Image edition, plus an explicit text clause. Name the
exact words you want rendered; the model puts them on the coin, the badge, or
the caption bar.

## Training

| | |
|---|---|
| base | Qwen-Image (Apache-2.0), uint3 + accuracy-recovery adapter during training, text encoder fp8 |
| data | 5.3k pump.fun coin logos (traded coins first, phash-deduped) + 98 meme templates repeated 8x |
| captions | Qwen2.5-VL, coin name + ticker prepended |
| network | LoRA rank 128, alpha 128 |
| schedule | 3500 steps, batch 1, lr 1e-4, adamw8bit, bf16, EMA 0.99 |
| hardware | 1x L40S, about 3 h |

Pipeline and config: [github.com/CookrAI/cookr](https://github.com/CookrAI/cookr) (`train/qwen_image_lora.yaml`).

## Light vs full

This is the light model: a few thousand curated images on an open base. The
full COOKR model is trained end to end on millions of memes and memecoin
images and is served on [cookr.pro](https://cookr.pro).

## License

Apache-2.0, same as the base. Training images remain the property of whoever made them.
