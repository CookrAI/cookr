"""Pull a meme-collector / pumpfun-collector output folder into the cookr db.
Images are already normalised + phashed by the collector, so they go straight
into `images` (status ok) and skip download.py. dedup.py and caption.py pick
them up like anything else.

usage: uv run python -m cookr.import_collector data/kym_out [--source kym]
"""
import argparse
import json
import pathlib
import shutil

from .db import RAW, ROOT, connect, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--source", default=None, help="only rows of this collector source")
    a = ap.parse_args()
    folder = pathlib.Path(a.folder)
    db = connect()
    n_new = n_skip = 0
    dest = RAW / "meme"
    dest.mkdir(parents=True, exist_ok=True)
    for line in (folder / "index.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok" or not r.get("path"):
            continue
        if a.source and r.get("source") != a.source:
            continue
        mid = f"{r['source']}-{r['id']}"
        key = f"meme:{mid}"
        if db.execute("SELECT 1 FROM images WHERE key=?", (key,)).fetchone():
            n_skip += 1
            continue
        src = folder / r["path"]
        dst = dest / f"{mid}.jpg"
        shutil.copy(src, dst)
        db.execute("INSERT OR IGNORE INTO memes(id,name,url,source,added_ts) VALUES(?,?,?,?,?)",
                   (mid, r.get("title") or r["id"], r["url"], r["source"], now()))
        db.execute("INSERT OR REPLACE INTO images(key,path,width,height,phash,status,ts) VALUES(?,?,?,?,?,?,?)",
                   (key, str(dst.relative_to(ROOT)), r.get("w"), r.get("h"), r.get("phash"), "ok", now()))
        n_new += 1
    db.commit()
    print(f"imported {n_new}, skipped {n_skip} already present")


if __name__ == "__main__":
    main()
