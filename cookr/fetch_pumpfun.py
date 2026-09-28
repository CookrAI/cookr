"""Pull the *successful* coins off pump.fun's frontend API.

The API only paginates ~1000-1500 deep per query, so we sweep every sort we know
x graduated/all. Gives ~5-8k unique coins that actually got attention: the best
meme signal on the platform. Volume comes from stream_pumpfun.py instead.

usage: uv run python -m cookr.fetch_pumpfun
"""
import sys
import time

import httpx

from .db import connect, upsert_coin

BASE = "https://frontend-api-v3.pump.fun/coins"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) cookr-dataset/0.1"}
SORTS = ["market_cap", "last_trade_timestamp", "reply_count", "last_reply", "created_timestamp"]
LIMIT = 50


def norm(c: dict) -> dict:
    return {
        "mint": c["mint"],
        "name": (c.get("name") or "").strip(),
        "symbol": (c.get("symbol") or "").strip(),
        "description": (c.get("description") or "").strip()[:500],
        "image_uri": c.get("image_uri"),
        "metadata_uri": c.get("metadata_uri"),
        "mcap_usd": c.get("usd_market_cap") or c.get("market_cap_usd") or 0,
        "complete": c.get("complete"),
        "created_ts": (c.get("created_timestamp") or 0) // 1000,
    }


def sweep(client: httpx.Client, db, sort: str, complete: bool) -> int:
    added = 0
    empty = 0
    offset = 0
    while empty < 3:
        params = {"offset": offset, "limit": LIMIT, "sort": sort, "order": "DESC", "includeNsfw": "false"}
        if complete:
            params["complete"] = "true"
        for attempt in range(5):
            try:
                r = client.get(BASE, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                rows = r.json()
                break
            except (httpx.HTTPError, ValueError):
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  giving up at offset {offset}", file=sys.stderr)
            break
        if not rows:
            empty += 1
            offset += LIMIT
            continue
        empty = 0
        for c in rows:
            if not c.get("mint") or not c.get("image_uri"):
                continue
            upsert_coin(db, norm(c), f"api:{sort}:{int(complete)}")
            added += 1
        db.commit()
        offset += LIMIT
        time.sleep(0.25)
    return added


def main():
    db = connect()
    with httpx.Client(headers=UA, http2=True) as client:
        for complete in (True, False):
            for sort in SORTS:
                n = sweep(client, db, sort, complete)
                total = db.execute("SELECT COUNT(*) FROM coins").fetchone()[0]
                print(f"{sort:22s} complete={int(complete)}  rows={n:5d}  unique so far={total}", flush=True)


if __name__ == "__main__":
    main()
