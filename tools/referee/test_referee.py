"""Does the referee actually catch bad data? Plant one kind of fault at a time and check it is reported.

    python tools/referee/test_referee.py data/games1000

Takes the first 40 hands of shard 0, checks they pass untouched, then makes copies with one fault
each (an illegal move, a tile from nowhere, a wrong score, ...) and expects the referee to flag every one.
"""

import csv
import gzip
import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import referee  # noqa: E402

ARRAYS = ["action", "mask", "meta", "obs", "oracle", "waits"]


def read_shard(src, n_hands):
    with open(src / "hands.csv", newline="") as f:
        rows = list(csv.DictReader(f))[:n_hands]
    arrays = {a: referee.load(src / f"{a}.npy") for a in ARRAYS}
    last = int(rows[-1]["hand_id"])
    keep = arrays["meta"][:, 0] <= last
    return rows, {a: v[keep].copy() for a, v in arrays.items()}


def write_shard(dst, rows, arrays):
    dst.mkdir(parents=True)
    with open(dst / "hands.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for a, v in arrays.items():
        buf = io.BytesIO()
        np.save(buf, v)
        with gzip.open(dst / f"{a}.npy.gz", "wb") as f:
            f.write(buf.getvalue())


def problems(rows, arrays, tmp, name):
    d = tmp / name
    write_shard(d, rows, arrays)
    errors, _, _, _ = referee.check_shard((str(d), 1, True))
    return errors


# Hand-made hands for patterns too rare to show up in self-play. Each is scored three ways: the number
# worked out by hand from RULES.md, the referee, and the engine (`dsy score`). All three must agree.
# (hand incl. winning tile and bonus tiles, winning tile, exposed melds, seat, prevailing, self-draw, expected tai)
RARE = [
    ("Small four winds (4, always with half flush 2, capped)", "333z44z456p", "6p", "pong:1z,pong:2z", 3, 0, False, 5),
    ("Big four winds (limit)", "444z55m", "5m", "pong:1z,pong:2z,pong:3z", 0, 0, False, 5),
    ("All honours (limit)", "111z222z333z555z66z", "6z", "", 2, 1, True, 5),
    ("Four kongs (limit)", "55p", "5p", "kong:1m,kong:2s,kong:9p,ckong:4m", 1, 0, False, 5),
    ("Nine gates (limit)", "11123455678999m", "5m", "", 1, 0, False, 5),
    ("Open ping hu, no bonus tiles", "456p789s234s55m", "2s", "chow:123m", 0, 0, False, 4),
    ("Open ping hu, non-matching flower", "456p789s234s55m f3", "2s", "chow:123m", 0, 0, False, 1),
    ("Concealed ping hu, non-matching flower", "123m456p789s234s55m f3", "2s", "", 0, 0, False, 5),
    ("Ping hu shape on an edge wait (no ping hu)", "123m456p789s234s55m", "3m", "", 0, 0, False, 1),
    ("Ping hu shape, matching flower (flower scores instead)", "123m456p789s234s55m f1", "2s", "", 0, 0, False, 2),
    ("Ping hu shape, pair is the seat wind (no ping hu)", "123m456p789s234s22z", "2s", "", 1, 0, False, 1),
    ("Small three dragons (replaces dragon pongs)", "555z666z77z123m", "1m", "pong:9s", 0, 1, False, 4),
    ("Seat and prevailing wind stack", "123m456p789s11p", "1p", "pong:1z", 0, 0, False, 2),
    ("Complete flower set counts 2, not 3", "123m456p789s234s99m f1 f2 f3 f4", "9m", "", 0, 0, True, 2 + 1),
    ("Half flush", "123m456m789m11z", "1z", "pong:5z", 1, 0, False, 2 + 1),
    ("All pongs", "222p333s99m", "9m", "pong:1m,pong:4p", 1, 0, False, 2),
    ("Four pongs finished on a discard are not 'concealed'", "111m222p333s444m55z", "4m", "", 1, 0, False, 2 + 1),
    ("Four concealed pongs by self-draw (limit)", "111m222p333s444m55z", "4m", "", 1, 0, True, 5),
    ("Seven pairs is not a winning hand", "1122m3344p5566s77z", "7z", "", 0, 0, False, 0),
]


def parse_tiles(text):
    out = []
    for word in text.split():
        if word[0] in "fga":
            out.append({"f": 34, "g": 38, "a": 42}[word[0]] + int(word[1]) - 1)
            continue
        digits = []
        for ch in word:
            if ch.isdigit():
                digits.append(int(ch))
            else:
                base = {"m": 0, "p": 9, "s": 18, "z": 27}[ch]
                out.extend(base + d - 1 for d in digits)
                digits = []
    return out


def parse_melds(text):
    melds = []
    for part in filter(None, text.split(",")):
        kind, tiles = part.split(":")
        t = parse_tiles(tiles)[0]
        melds.append({"kind": {"pong": "pong", "chow": "chow", "kong": "exposed_kong", "ckong": "concealed_kong"}[kind],
                      "tile": t})
    return melds


def rare_patterns(dsy):
    import re
    import subprocess

    ok_all = True
    for name, hand, win, melds, seat, prev, tsumo, expected in RARE:
        tiles = parse_tiles(hand)
        counts = [0] * 34
        for t in tiles:
            if t < 34:
                counts[t] += 1
        ctx = {"seat_wind": seat, "prevailing": prev, "bonus": [t for t in tiles if t >= 34], "self_draw": tsumo,
               "win_tile": parse_tiles(win)[0], "kong_replacement": False, "robbing": False, "last_tile": False,
               "heavenly": False, "earthly": False}
        v = referee.score_win(counts, parse_melds(melds), ctx)
        ref = v[0] if v and v[0] >= 1 else 0
        eng = None
        if dsy:
            cmd = [dsy, "score", hand, "--win", win, "--melds", melds, "--seat", str(seat), "--prevailing", str(prev)]
            out = subprocess.run(cmd + (["--tsumo"] if tsumo else []), capture_output=True, text=True).stdout
            m = re.search(r"Total: (\d+) tai", out)
            eng = int(m.group(1)) if m else 0
        ok = ref == expected and (eng is None or eng == expected)
        ok_all &= ok
        print(f"  {'ok  ' if ok else 'DIFF'} {name}: rules {expected}, referee {ref}, engine {eng if eng is not None else '-'}")
    return ok_all


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "data/games1000")
    rows, arrays = read_shard(run / "shard_00000", 40)
    tmp = Path(tempfile.mkdtemp())
    try:
        clean = problems(rows, arrays, tmp, "clean")
        print(f"untouched data: {len(clean)} problem(s)")
        assert not clean, clean

        def fault(name, change):
            r = [dict(x) for x in rows]
            a = {k: v.copy() for k, v in arrays.items()}
            change(r, a)
            found = problems(r, a, tmp, name)
            ok = bool(found)
            print(f"  {'caught' if ok else 'MISSED'}: {name}" + (f"  ->  {found[0]}" if ok else ""))
            return ok

        own_turns = np.flatnonzero(arrays["obs"][:, 658 + 5] == 0)
        i = int(own_turns[50])
        hand = arrays["obs"][i, :34]
        absent = int(np.flatnonzero(hand == 0)[0])
        win = next(k for k, x in enumerate(rows) if int(x["winner"]) >= 0)
        ready = np.argwhere(arrays["waits"] > 0)[0]

        def illegal_move(r, a):
            a["action"][i] = absent  # throw a tile the player does not hold

        def extra_option(r, a):
            a["mask"][i, 34 + absent] = 1  # engine offers a kong that is not allowed

        def tile_from_nowhere(r, a):
            a["obs"][i, absent] += 1

        def peeked_hand(r, a):
            a["oracle"][i, 0] = (a["oracle"][i, 0] + 1) % 4

        def wrong_tai(r, a):
            r[win]["tai"] = str(int(r[win]["tai"]) % 5 + 1)

        def not_zero_sum(r, a):
            r[win]["d0"] = str(int(r[win]["d0"]) + 1)

        def wrong_winner(r, a):
            r[win]["winner"] = str((int(r[win]["winner"]) + 1) % 4)

        def tampered_seed(r, a):
            r[3]["seed"] = str(int(r[3]["seed"]) + 1)

        def wrong_danger_label(r, a):
            a["waits"][tuple(ready)] = 0  # hide a tile that would really deal in

        def dropped_decision(r, a):
            for k in ARRAYS:
                a[k] = np.delete(a[k], i, axis=0)

        tests = [illegal_move, extra_option, tile_from_nowhere, peeked_hand, wrong_tai, not_zero_sum,
                 wrong_winner, tampered_seed, wrong_danger_label, dropped_decision]
        results = [fault(t.__name__.replace("_", " "), t) for t in tests]
        print(f"{sum(results)}/{len(results)} planted faults caught")
        dsy = next((str(p) for p in (Path("target/release/dsy"), Path("target/release/dsy.exe")) if p.exists()), None)
        print("\nRare patterns, scored by hand from RULES.md, by the referee and by the engine:")
        rare_ok = rare_patterns(dsy)
        sys.exit(0 if all(results) and rare_ok else 1)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
