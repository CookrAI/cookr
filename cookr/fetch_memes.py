"""Internet meme sources. Templates are the important part: the model must
learn the *canonical* Drake / Distracted Boyfriend / Wojak frames, not only
coin logos derived from them.

Sources:
  - imgflip get_memes: top 100 templates, clean, no auth
  - data/memes.txt   : optional manual list, one "name<TAB>url" per line
                       (drop KnowYourMeme / Reddit picks here by hand)

usage: uv run python -m cookr.fetch_memes
"""
import httpx

from .db import DATA, connect, now


def main():
    db = connect()
    r = httpx.get("https://api.imgflip.com/get_memes", timeout=30)
    r.raise_for_status()
    memes = r.json()["data"]["memes"]
    for m in memes:
        db.execute(
            "INSERT OR IGNORE INTO memes(id,name,url,source,added_ts) VALUES(?,?,?,?,?)",
            (f"imgflip-{m['id']}", m["name"], m["url"], "imgflip", now()),
        )
    manual = DATA / "memes.txt"
    k = 0
    if manual.exists():
        for i, line in enumerate(manual.read_text().splitlines()):
            if "\t" not in line or line.startswith("#"):
                continue
            name, url = line.split("\t", 1)
            db.execute(
                "INSERT OR IGNORE INTO memes(id,name,url,source,added_ts) VALUES(?,?,?,?,?)",
                (f"manual-{i}", name.strip(), url.strip(), "manual", now()),
            )
            k += 1
    db.commit()
    print(f"imgflip templates: {len(memes)}, manual: {k}, total memes: {db.execute('SELECT COUNT(*) FROM memes').fetchone()[0]}")


if __name__ == "__main__":
    main()
