"""Resolve metadata -> image URL, download, normalise, phash.

Normalisation: RGB, alpha flattened on white (coin logos are often transparent
PNG), longest side <= 1024, saved as JPEG q92. Anything with a short side
< 256px is marked `small` and skipped: upscaled junk teaches the model junk.

Idempotent: keys already in `images` are skipped. Run again after the stream
collected more coins.

usage: uv run python -m cookr.download [--limit N] [--concurrency 24]
"""
import argparse
import asyncio
import io
import json
import re
import sys

import httpx
import imagehash
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from .db import RAW, connect, now

IPFS_GATEWAYS = ["https://pump.mypinata.cloud/ipfs/", "https://gateway.pinata.cloud/ipfs/", "https://ipfs.io/ipfs/"]  # ipfs.io rate-limits hard, keep it last
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) cookr-dataset/0.1"}
MAX_SIDE = 1024
MIN_SIDE = 256


def candidates(url: str) -> list[str]:
    if "/ipfs/" in url:
        cid = url.split("/ipfs/", 1)[1]
        return [g + cid for g in IPFS_GATEWAYS]
    if "pbs.twimg.com" in url:
        # twitter thumbnails: ask for the large variant first
        big = re.sub(r"name=\w+", "name=large", url)
        return [big, url] if big != url else [url]
    return [url]


async def fetch_bytes(client: httpx.AsyncClient, url: str, min_len: int = 500) -> bytes | None:
    """min_len guards against gateway error pages served as 200. Metadata JSON
    is often < 500 bytes, so callers fetching JSON pass a lower floor."""
    for u in candidates(url):
        try:
            r = await client.get(u, timeout=25, follow_redirects=True)
            if r.status_code == 200 and len(r.content) >= min_len:
                return r.content
        except httpx.HTTPError:
            continue
    return None


async def resolve_image_uri(client: httpx.AsyncClient, metadata_uri: str) -> tuple[str | None, str | None]:
    raw = await fetch_bytes(client, metadata_uri, min_len=20)
    if not raw:
        return None, None
    try:
        j = json.loads(raw)
    except ValueError:
        return None, None
    return j.get("image"), (j.get("description") or "")[:500]


def normalise(raw: bytes) -> tuple[Image.Image, str] | tuple[None, str]:
    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im)
        if getattr(im, "is_animated", False):
            im.seek(0)
        im.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "bad"
    if min(im.size) < MIN_SIDE:
        return None, "small"
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    # flat / blank images: almost no variance
    if ImageStat.Stat(im.convert("L").resize((64, 64))).var[0] < 30:
        return None, "flat"
    return im, "ok"


PER_IMAGE_BUDGET = 60  # seconds, all gateways included; slow hosts must not stall the run


async def process(key: str, kind: str, url: str | None, meta_uri: str | None, client: httpx.AsyncClient):
    desc = None
    if not url and meta_uri:
        url, desc = await resolve_image_uri(client, meta_uri)
    status, path, w, h, ph = "failed", None, None, None, None
    if url:
        raw = await fetch_bytes(client, url)
        if raw:
            im, status = normalise(raw)
            if im is not None:
                sub = RAW / kind
                sub.mkdir(parents=True, exist_ok=True)
                path = sub / f"{key.split(':', 1)[1]}.jpg"
                im.save(path, "JPEG", quality=92)
                w, h = im.size
                ph = str(imagehash.phash(im))
                path = str(path.relative_to(RAW.parent.parent))
    return key, url, desc, status, path, w, h, ph


async def worker(name: str, q: asyncio.Queue, client: httpx.AsyncClient, results: asyncio.Queue):
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            return
        key, kind, url, meta_uri = item
        try:
            res = await asyncio.wait_for(process(key, kind, url, meta_uri, client), PER_IMAGE_BUDGET)
        except asyncio.TimeoutError:
            res = (key, url, None, "failed", None, None, None, None)
        except Exception as e:  # never let one image kill the run or starve the main loop
            print("worker error", key, repr(e), file=sys.stderr, flush=True)
            res = (key, url, None, "bad", None, None, None, None)
        await results.put(res)
        q.task_done()


async def run(limit: int, concurrency: int):
    db = connect()
    done = {r[0] for r in db.execute("SELECT key FROM images")}
    todo = []
    for r in db.execute("SELECT mint,image_uri,metadata_uri FROM coins ORDER BY complete DESC, mcap_usd DESC"):
        k = f"pump:{r['mint']}"
        if k not in done:
            todo.append((k, "pump", r["image_uri"], r["metadata_uri"]))
    for r in db.execute("SELECT id,url FROM memes"):
        k = f"meme:{r['id']}"
        if k not in done:
            todo.append((k, "meme", r["url"], None))
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} images to fetch", flush=True)
    if not todo:
        return
    q: asyncio.Queue = asyncio.Queue()
    results: asyncio.Queue = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=UA, limits=limits, http2=True) as client:
        workers = [asyncio.create_task(worker(f"w{i}", q, client, results)) for i in range(concurrency)]
        counts: dict[str, int] = {}
        for i in range(len(todo)):
            try:
                key, url, desc, status, path, w, h, ph = await asyncio.wait_for(results.get(), PER_IMAGE_BUDGET * 2)
            except asyncio.TimeoutError:
                print("no results for a while, stopping; re-run to resume", flush=True)
                break
            counts[status] = counts.get(status, 0) + 1
            db.execute(
                "INSERT OR REPLACE INTO images(key,path,width,height,phash,status,ts) VALUES(?,?,?,?,?,?,?)",
                (key, path, w, h, ph, status, now()),
            )
            if key.startswith("pump:") and (url or desc):
                db.execute(
                    "UPDATE coins SET image_uri=COALESCE(image_uri,?), description=COALESCE(NULLIF(description,''),?) WHERE mint=?",
                    (url, desc, key[5:]),
                )
            if (i + 1) % 25 == 0:
                db.commit()  # short transactions: stream_pumpfun.py shares this db
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(todo)} {counts}", flush=True)
        db.commit()
        for w_ in workers:
            w_.cancel()
    print("done", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()
    asyncio.run(run(a.limit, a.concurrency))
