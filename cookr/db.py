"""Single sqlite file holds everything: coin metadata, image status, captions.
No ORM. Rows are dicts. Re-runnable: every stage is idempotent on primary keys."""
import pathlib
import sqlite3
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
TRAIN = DATA / "train"
DB = DATA / "cookr.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS coins(
  mint TEXT PRIMARY KEY,
  name TEXT, symbol TEXT, description TEXT,
  image_uri TEXT, metadata_uri TEXT,
  source TEXT,            -- api:<sort>:<complete> | stream
  mcap_usd REAL, complete INTEGER, created_ts INTEGER,
  added_ts INTEGER
);
CREATE TABLE IF NOT EXISTS images(
  key TEXT PRIMARY KEY,   -- pump:<mint> | meme:<id>
  path TEXT, width INTEGER, height INTEGER, phash TEXT,
  status TEXT,            -- ok | dup | small | bad | failed | flat
  dup_of TEXT, ts INTEGER
);
CREATE TABLE IF NOT EXISTS memes(
  id TEXT PRIMARY KEY, name TEXT, url TEXT, source TEXT, added_ts INTEGER
);
CREATE TABLE IF NOT EXISTS captions(
  key TEXT PRIMARY KEY, raw TEXT, caption TEXT, model TEXT, ts INTEGER
);
CREATE INDEX IF NOT EXISTS images_status ON images(status);
"""


def connect() -> sqlite3.Connection:
    DATA.mkdir(exist_ok=True)
    c = sqlite3.connect(DB, timeout=120)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def now() -> int:
    return int(time.time())


def upsert_coin(c: sqlite3.Connection, coin: dict, source: str):
    c.execute(
        """INSERT INTO coins(mint,name,symbol,description,image_uri,metadata_uri,source,mcap_usd,complete,created_ts,added_ts)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(mint) DO UPDATE SET
             name=COALESCE(excluded.name,name), symbol=COALESCE(excluded.symbol,symbol),
             description=COALESCE(excluded.description,description),
             image_uri=COALESCE(excluded.image_uri,image_uri),
             metadata_uri=COALESCE(excluded.metadata_uri,metadata_uri),
             mcap_usd=MAX(COALESCE(excluded.mcap_usd,0),COALESCE(mcap_usd,0)),
             complete=MAX(COALESCE(excluded.complete,0),COALESCE(complete,0))""",
        (
            coin["mint"], coin.get("name"), coin.get("symbol"), coin.get("description"),
            coin.get("image_uri"), coin.get("metadata_uri"), source,
            coin.get("mcap_usd"), int(bool(coin.get("complete"))), coin.get("created_ts"), now(),
        ),
    )
