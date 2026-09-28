"""Near-duplicate removal on perceptual hash.

pump.fun is 30% copy-paste: the same Pepe uploaded 400 times with a different
ticker. Keep ONE per visual cluster, and prefer the most successful coin as the
representative (already ordered complete DESC, mcap DESC in the query).

Hamming distance <= THRESH on a 64-bit phash. Uses the pigeonhole trick: split
the hash into 4 x 16-bit chunks; two hashes within distance 3 must agree on at
least one chunk exactly, so we only compare inside chunk buckets. O(n) memory,
fast enough for a few hundred k rows.

usage: uv run python -m cookr.dedup
"""
from collections import defaultdict

from .db import connect, now

THRESH = 3
CHUNKS = 4


def chunks(h: int):
    for i in range(CHUNKS):
        yield i, (h >> (16 * i)) & 0xFFFF


def main():
    db = connect()
    rows = db.execute(
        """SELECT i.key, i.phash FROM images i
           LEFT JOIN coins c ON i.key = 'pump:' || c.mint
           WHERE i.status IN ('ok','dup') AND i.phash IS NOT NULL
           ORDER BY (i.key LIKE 'meme:%') DESC, c.complete DESC, c.mcap_usd DESC"""
    ).fetchall()
    buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    kept = 0
    dups = 0
    updates = []
    for r in rows:
        h = int(r["phash"], 16)
        dup_of = None
        seen = set()
        for cb in chunks(h):
            for kh, kkey in buckets[cb]:
                if kkey in seen:
                    continue
                seen.add(kkey)
                if bin(h ^ kh).count("1") <= THRESH:
                    dup_of = kkey
                    break
            if dup_of:
                break
        if dup_of:
            dups += 1
            updates.append(("dup", dup_of, r["key"]))
        else:
            kept += 1
            updates.append(("ok", None, r["key"]))
            for cb in chunks(h):
                buckets[cb].append((h, r["key"]))
    db.executemany("UPDATE images SET status=?, dup_of=?, ts=? WHERE key=?", [(s, d, now(), k) for s, d, k in updates])
    db.commit()
    print(f"kept {kept}, marked {dups} near-duplicates (thresh {THRESH})")


if __name__ == "__main__":
    main()
