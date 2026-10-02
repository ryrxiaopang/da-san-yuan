"""Build the replay page from a trace exported by `dsy trace`.

    dsy trace --hands 380,241,27,19 --out web/replay/replay.json
    python web/replay/build.py

Writes web/replay/index.html, a single self-contained page: open it in any
browser. The hands must come from the run you are describing (by default the
20,000-hand run made with `dsy selfplay --hands 20000 --out data/run20k`).
"""
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
    with open(os.path.join(HERE, "viewer.html"), encoding="utf-8") as f:
        page = f.read().replace("/*REPLAY_DATA*/", payload)
    html = SKELETON.replace("{page}", page) if standalone else page
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html) / 1024:.0f} KB, {len(data['games'])} hands)")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "replay.json")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "index.html")
    build(data, out, standalone="--bare" not in sys.argv)
