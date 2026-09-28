"""Quick look at where the dataset stands.  usage: uv run python -m cookr.stats"""
from .db import connect


def main():
    db = connect()
    q = lambda s: db.execute(s).fetchone()[0]
    print("coins       :", q("SELECT COUNT(*) FROM coins"))
    print("  graduated :", q("SELECT COUNT(*) FROM coins WHERE complete=1"))
    print("  from api  :", q("SELECT COUNT(*) FROM coins WHERE source LIKE 'api:%'"))
    print("  from stream:", q("SELECT COUNT(*) FROM coins WHERE source='stream'"))
    print("memes       :", q("SELECT COUNT(*) FROM memes"))
    print("images      :")
    for r in db.execute("SELECT status, COUNT(*) n FROM images GROUP BY status ORDER BY n DESC"):
        print(f"  {r['status']:8s} {r['n']}")
    print("captioned   :", q("SELECT COUNT(*) FROM captions"))


if __name__ == "__main__":
    main()
