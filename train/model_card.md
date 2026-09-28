---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
  - lora
  - text-to-image
  - z-image
  - meme
  - memecoin
  - pump.fun
pipeline_tag: text-to-image
library_name: diffusers
instance_prompt: cookr
widget:
  - text: "cookr, memecoin logo of Bonk Dog (BONK), cartoon shiba inu wearing sunglasses, orange and yellow, bold outline"
    output:
      url: samples/sample_0.jpg
---

# COOKR v1 (light)

Open meme image model. A LoRA for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
trained on the memes that moved markets: pump.fun coin art that actually traded,
plus the internet templates it grew out of.

**This is the light model.** A few thousand curated images, a LoRA on an open
base, trained in an afternoon. It exists so anyone can run COOKR locally today.
The full COOKR model is a separate thing: trained end to end on millions of
memes and memecoin images, served first through [cookr.pro](https://cookr.pro),
and open-sourced after that. Treat v1-light as the preview of the style, not the
ceiling.

[cookr.pro](https://cookr.pro) · [GitHub](https://github.com/CookrAI/cookr) · hi@cookr.pro

## Use

Trigger word: `cookr`. Coin name and ticker are in every training caption, so
"memecoin logo of X (TICK)" is the native prompt shape.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("CookrAI/cookr-v1")

img = pipe(
    "cookr, memecoin logo of Pepe Astronaut (PEPENAUT), pepe the frog in a spacesuit on the moon, green and black",
    num_inference_steps=9, guidance_scale=1.0, height=1024, width=1024,
).images[0]
img.save("pepenaut.png")
```

Works in ComfyUI and any Z-Image-Turbo LoRA loader. Sample at 8-9 steps, CFG 1.
LoRA weight 0.8-1.0.

Prompt shapes that work:

- `cookr, memecoin logo of <name> (<TICKER>), <what it is>, <style>, <colors>`
- `cookr, <meme template name> meme template, <scene>`
- `cookr, wojak crying in front of a red candle chart, mspaint style`

## Training

| | |
|---|---|
| base | Z-Image-Turbo, trained through ai-toolkit's `zimage:turbo` de-distill adapter |
| data | 5.3k pump.fun coin logos (graduated / traded coins first, phash-deduped) + 98 meme templates repeated 8x |
| captions | Qwen2.5-VL, coin name + ticker prepended, `cookr` trigger |
| network | LoRA rank 96, alpha 96 |
| schedule | 5000 steps, batch 2, lr 1e-4, adamw8bit, bf16, EMA 0.99 |
| resolutions | 512 / 768 / 1024 buckets |
| hardware | 1x L40S, about 1.5 h |

Full pipeline, collectors and this config: [github.com/CookrAI/cookr](https://github.com/CookrAI/cookr).
Collectors: [pumpfun-collector](https://github.com/CookrAI/pumpfun-collector),
[meme-collector](https://github.com/CookrAI/meme-collector),
[x-collector](https://github.com/CookrAI/x-collector).

## Data notes

pump.fun's API pages about 1000 deep per sort, so the sweep unions every sort
for graduated and for all coins. Anything under 256px was dropped, never
upscaled. "Strategic reserve" / "dividend fund" / AI-agent branding was
filtered out by name and by caption: this is a meme model, not a logo generator.

## License

Apache-2.0, same as the base. The training images remain the property of
whoever made them.
