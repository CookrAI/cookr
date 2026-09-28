"""Volume source: PumpPortal websocket streams every new token creation
(~40k/day). Run it in the background for a few days to reach 100k+.

Only stores mint/name/symbol/uri here; image resolution happens in download.py
so this loop never blocks.

usage: uv run python -m cookr.stream_pumpfun   (Ctrl-C to stop, safe to resume)
"""
import asyncio
import json
import signal
import sqlite3

import websockets

from .db import connect, upsert_coin

WS = "wss://pumpportal.fun/api/data"


async def run():
    db = connect()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, stop.set)
    n = 0
    while not stop.is_set():
        try:
            async with websockets.connect(WS, ping_interval=20) as ws:
                await ws.send(json.dumps({"method": "subscribeNewToken"}))
                print("subscribed", flush=True)
                while not stop.is_set():
                    try:
                        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    except asyncio.TimeoutError:
                        continue
                    if "mint" not in m or not m.get("uri"):
                        continue
                    coin = {
                        "mint": m["mint"], "name": (m.get("name") or "").strip(),
                        "symbol": (m.get("symbol") or "").strip(),
                        "metadata_uri": m["uri"], "created_ts": None,
                    }
                    # download.py writes to the same db; on lock, back off and retry
                    for attempt in range(20):
                        try:
                            upsert_coin(db, coin, "stream")
                            db.commit()
                            break
                        except sqlite3.OperationalError as e:
                            db.rollback()
                            await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        print("dropped (db busy):", coin["mint"], flush=True)
                    n += 1
                    if n % 50 == 0:
                        total = db.execute("SELECT COUNT(*) FROM coins").fetchall()[0][0]
                        print(f"+{n} this session, {total} coins in db", flush=True)
        except (websockets.WebSocketException, OSError) as e:
            print("ws error, reconnecting:", e)
            await asyncio.sleep(3)
    db.commit()


if __name__ == "__main__":
    asyncio.run(run())
