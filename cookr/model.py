"""COOKR inference. One class, our defaults, our prompt grammar.

    from cookr import Cookr
    c = Cookr()                                   # CookrAI/cookr-v1-light
    img = c.cook("frog chef who trades charts", ticker="RIBBIT")
    img.save("ribbit.png")

Prompt grammar (what the model was trained on):
    <character>, $<TICKER>, <format>, <vibe>
    format: coin logo | sticker | meme template
"""
from __future__ import annotations

import re

MODEL_ID = "CookrAI/cookr-v1-light"
FORMATS = {"logo": "memecoin logo", "sticker": "die-cut sticker", "meme": "meme template"}
DEFAULTS = {"steps": 9, "guidance": 1.0, "width": 1024, "height": 1024}


def build_prompt(idea: str, ticker: str | None = None, fmt: str = "logo", vibe: str | None = None) -> str:
    """Turn a plain idea into the caption shape the model learned."""
    idea = idea.strip().rstrip(".")
    head = FORMATS.get(fmt, fmt)
    if ticker:
        t = re.sub(r"[^A-Za-z0-9]", "", ticker).upper()
        parts = [f"{head} of {idea} ({t})"]
    else:
        parts = [f"{head} of {idea}"]
    if vibe:
        parts.append(vibe.strip())
    return "cookr, " + ", ".join(parts)


class Cookr:
    def __init__(self, model: str = MODEL_ID, device: str | None = None, dtype=None):
        import torch
        from diffusers import ZImagePipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device != "cpu" else torch.float32)
        self.pipe = ZImagePipeline.from_pretrained(model, torch_dtype=self.dtype).to(self.device)

    def cook(self, idea: str, ticker: str | None = None, fmt: str = "logo", vibe: str | None = None,
             seed: int | None = None, steps: int | None = None, guidance: float | None = None,
             width: int | None = None, height: int | None = None, n: int = 1):
        import torch

        prompt = build_prompt(idea, ticker, fmt, vibe)
        gen = None if seed is None else torch.Generator(self.device if self.device != "mps" else "cpu").manual_seed(seed)
        out = self.pipe(
            prompt,
            num_inference_steps=steps or DEFAULTS["steps"],
            guidance_scale=DEFAULTS["guidance"] if guidance is None else guidance,
            width=width or DEFAULTS["width"], height=height or DEFAULTS["height"],
            num_images_per_prompt=n, generator=gen,
        )
        return out.images[0] if n == 1 else out.images

    def directions(self, idea: str, ticker: str | None = None, seed: int = 0):
        """The product move: four takes on one idea. logo / sticker / meme / logo with a twist."""
        specs = [("logo", None), ("sticker", "bold outline, flat colors"), ("meme", None), ("logo", "3d render, dramatic lighting")]
        return [self.cook(idea, ticker, f, v, seed=seed + i) for i, (f, v) in enumerate(specs)]
