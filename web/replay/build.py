"""Build the replay page from a trace exported by `dsy trace`.

    dsy trace --game 1 --out web/replay/replay.json
    python web/replay/build.py

Writes web/replay/index.html, a single self-contained page: open it in any
browser. Tile pictures come from web/replay/tiles/<id>.webp (00-08 characters,
09-17 dots, 18-26 bamboo, 27-30 winds, 31-33 white/green/red dragons,
34-37 seasons, 38-41 flowers, 42-45 cat/rat/rooster/centipede). The hands must come from the run you are describing (by default the
1,000-game run made with `dsy selfplay --games 1000 --out data/games1000`).
"""
import base64
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
{page}
</body>
</html>
"""


def build(data_path: str, out_path: str, standalone: bool = True) -> None:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    css = []
    for path in sorted(glob.glob(os.path.join(HERE, "tiles", "*.webp"))):
        tid = int(os.path.basename(path)[:2])
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        css.append(f".t.f{tid}{{background-image:url(data:image/webp;base64,{b64})}}")
    with open(os.path.join(HERE, "viewer.html"), encoding="utf-8") as f:
        page = f.read().replace("/*REPLAY_DATA*/", payload).replace("/*TILE_CSS*/", "\n".join(css))
    html = SKELETON.replace("{page}", page) if standalone else page
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html) / 1024:.0f} KB, {len(data['games'])} hands)")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "replay.json")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "index.html")
    build(data, out, standalone="--bare" not in sys.argv)
