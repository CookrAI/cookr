# Training runs

| run | base | data | steps | result |
|---|---|---|---|---|
| v1 (2026-09-28) | Z-Image-Turbo via `zimage:turbo` adapter | 5.3k coins + 98 templates x8 | 5000 | **published** `CookrAI/cookr-v1-light` (merged). Clean coin logos / stickers, Pepe, Doge, Wojak. No template compositions. |
| qwen (2026-09-29) | Qwen-Image, uint3 + ARA | same | 3500 | **published** `CookrAI/cookr-v1-light-qwen` (LoRA). Text rendering survives, pixel-degen style. |
| v1.1 (2026-09-29) | Z-Image-Turbo via adapter | 5.3k coins + 4.9k KYM gallery photos (x1) | 6000 | **not published**. More meme knowledge (Gigachad, Wojak, Drake) but coin-logo / sticker quality regressed: raw KYM galleries are photos, screenshots and crude edits, and they diluted the style. LoRA kept at `data/runs/v1_1/`. |

Lessons for v1.2:
- Filter KYM galleries by caption: drop `photo of a person`, `screenshot`, `text post`; keep illustrations / templates.
- Weight templates by fame (top 50 x N), not "every gallery photo once". 16 Distracted Boyfriends in 10k images teaches nothing.
- Keep memes at ~20% of the set. 50/50 loses the memecoin look.
- Pod disk: ai-toolkit stores Comfy single-file weights under `ai-toolkit/models/`, not `HF_HOME`. 80 GB fills in one evening; rent 150 GB.
- Don't delete intermediate checkpoints before looking at the samples: v1.1's 5000-step checkpoint looked better than 6000 and it was already gone.
