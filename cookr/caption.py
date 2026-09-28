"""Caption every kept image with a VLM, then wrap it in the dataset format.

Final caption shape (this is what the LoRA learns to answer to):
  coins : "cookr, memecoin logo of <name> (<SYMBOL>), <vlm caption>"
  memes : "cookr, <template name> meme template, <vlm caption>"

Coin name/ticker in the caption = the model can be prompted by coin name later.
`cookr` = trigger word.

Backends
  mlx : Apple Silicon, mlx-vlm + Qwen2.5-VL 3B 4-bit. ~2.3s/img on M4 Pro (20k images = one night).
        uv run --extra mlx python -m cookr.caption
  hf  : CUDA/CPU, Florence-2-large. uv run --extra hf python -m cookr.caption --backend hf

Resumable; only captions rows missing from `captions`.
"""
import argparse
import re
import sys

from PIL import Image
from tqdm import tqdm

from . import TRIGGER
from .db import ROOT, connect, now

PROMPT = (
    "Describe this image in one sentence, max 40 words, as a text-to-image training caption. "
    "Cover: main subject, art style (cartoon, pixel art, 3d render, photo, drawing), "
    "dominant colors, any visible text in quotes. No preamble, no opinions, no mood words."
)
MLX_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
HF_MODEL = "microsoft/Florence-2-large"


def clean(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"^(this|the) image (shows|depicts|features|is)\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).rstrip(".")
    return s[:1].lower() + s[1:]


def compose(key: str, raw: str, db) -> str:
    if key.startswith("pump:"):
        c = db.execute("SELECT name, symbol FROM coins WHERE mint=?", (key[5:],)).fetchone()
        name, sym = (c["name"] or "").strip(), (c["symbol"] or "").strip().upper()
        head = f"memecoin logo of {name} ({sym})" if name and sym else "memecoin logo"
    else:
        m = db.execute("SELECT name FROM memes WHERE id=?", (key[5:],)).fetchone()
        head = f"{m['name']} meme template" if m else "meme"
    return f"{TRIGGER}, {head}, {raw}"


class MLX:
    def __init__(self):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        self.model, self.proc = load(MLX_MODEL)
        self.cfg = load_config(MLX_MODEL)
        self._gen, self._tmpl = generate, apply_chat_template
        self.name = MLX_MODEL

    def __call__(self, path: str) -> str:
        prompt = self._tmpl(self.proc, self.cfg, PROMPT, num_images=1)
        out = self._gen(self.model, self.proc, prompt, [path], max_tokens=90, temperature=0.0, verbose=False)
        return out.text if hasattr(out, "text") else str(out)


class HF:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev)
        self.proc = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)
        self.name = HF_MODEL

    def __call__(self, path: str) -> str:
        im = Image.open(path).convert("RGB")
        task = "<MORE_DETAILED_CAPTION>"
        inp = self.proc(text=task, images=im, return_tensors="pt").to(self.dev, self.model.dtype)
        ids = self.model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=96, num_beams=3)
        txt = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        return self.proc.post_process_generation(txt, task=task, image_size=im.size)[task]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "hf"], default="mlx" if sys.platform == "darwin" else "hf")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    db = connect()
    rows = db.execute(
        """SELECT i.key, i.path FROM images i LEFT JOIN captions c ON c.key=i.key
           WHERE i.status='ok' AND c.key IS NULL ORDER BY (i.key LIKE 'meme:%') DESC, i.key"""
    ).fetchall()
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} images to caption with {a.backend}")
    if not rows:
        return
    cap = MLX() if a.backend == "mlx" else HF()
    for i, r in enumerate(tqdm(rows)):
        try:
            raw = clean(cap(str(ROOT / r["path"])))
        except Exception as e:  # keep going, one bad image must not kill a 20k run
            print("caption failed", r["key"], e, file=sys.stderr)
            continue
        db.execute(
            "INSERT OR REPLACE INTO captions(key,raw,caption,model,ts) VALUES(?,?,?,?,?)",
            (r["key"], raw, compose(r["key"], raw, db), cap.name, now()),
        )
        if (i + 1) % 25 == 0:
            db.commit()
    db.commit()


if __name__ == "__main__":
    main()
