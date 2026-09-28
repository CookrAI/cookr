"""Write the training folder ai-toolkit / kohya expect:
  data/train/<key>.jpg + data/train/<key>.txt

Memes get repeated REPEAT_MEMES times (symlinks with suffix) so 100 templates
are not drowned by 20k coin logos.

usage: uv run python -m cookr.export
"""
import os
import re
import shutil

from .db import ROOT, TRAIN, connect

REPEAT_MEMES_MAX = 8  # templates are repeated so they hold ~40% of the set, never more than 8x
# COOKR is a meme model. pump.fun's top of the mcap table is full of fake
# "strategic reserve" / "dividend fund" / AI-agent branding; those are logos,
# not memes. Drop by name/ticker/description. Tune as the dataset grows.
NOT_MEME = re.compile(
    r"\b(fund|reserve|bridge|protocol|network|treasury|capital|etf|strategic|"
    r"dividend|agent|ai|finance|financial|bank|exchange|holdings|ventures|labs|"
    r"defi|dao|layer|chain|swap|staking|yield)\b",
    re.I,
)
# caption-level junk: screenshots of tweets/charts/chats are not memes either
NOT_MEME_CAPTION = re.compile(r"\b(screenshot|candlestick|price chart|trading chart|spreadsheet|chat conversation|text message)\b", re.I)


def main():
    db = connect()
    if TRAIN.exists():
        shutil.rmtree(TRAIN)
    TRAIN.mkdir(parents=True)
    rows = db.execute(
        """SELECT i.key, i.path, cap.caption, co.name, co.symbol, co.description
           FROM images i JOIN captions cap ON cap.key=i.key
           LEFT JOIN coins co ON i.key = 'pump:' || co.mint
           WHERE i.status='ok'"""
    ).fetchall()
    n_coin = n_meme = n_skip = 0
    n_m = sum(1 for r in rows if r["key"].startswith("meme:"))
    n_c = max(1, len(rows) - n_m)
    repeat = max(1, min(REPEAT_MEMES_MAX, round(0.66 * n_c / max(1, n_m))))
    for r in rows:
        if r["key"].startswith("pump:") and (
            NOT_MEME.search(" ".join(filter(None, (r["name"], r["symbol"], r["description"]))))
            or NOT_MEME_CAPTION.search(r["caption"])
        ):
            n_skip += 1
            continue
        src = ROOT / r["path"]
        base = r["key"].replace(":", "_")
        reps = repeat if r["key"].startswith("meme:") else 1
        for k in range(reps):
            name = base if k == 0 else f"{base}__r{k}"
            os.symlink(src.resolve(), TRAIN / f"{name}.jpg")
            (TRAIN / f"{name}.txt").write_text(r["caption"])
        if r["key"].startswith("meme:"):
            n_meme += 1
        else:
            n_coin += 1
    print(f"exported {n_coin} coin images + {n_meme} meme templates x{repeat} -> {TRAIN}  (skipped {n_skip} non-meme coins)")
    print("sample captions:")
    for t in sorted(TRAIN.glob("pump_*.txt"))[:3]:
        print("  ", t.read_text())


if __name__ == "__main__":
    main()
