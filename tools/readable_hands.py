"""Turn a self-play run into one easy-to-read table of hands.

    python tools/readable_hands.py data/run20k

Writes data/run20k/hands_readable.csv (opens in Excel). The original
shard_*/hands.csv files are left untouched: the loader and training code use
those, and they keep the exact seeds needed to replay a hand.

Each hand is labelled by its place in the game, e.g. "Game 1, East round, Dealer 4,
Repeat 1": the round's fourth dealer, who won the previous hand and stayed on.
Each player is named by seat wind and play style, e.g. "South (Balanced)":
the style alone is not enough because two seats can play the same style.
Seat winds move every hand (the dealer is always East), so the East player
in one hand is a different seat from the East player in the next.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

WINDS = ["East", "South", "West", "North"]
STYLE_NAMES = {"fast": "Fast", "high_tai": "High-tai", "defensive": "Defensive", "balanced": "Balanced"}

COLUMNS = {
    "game": "Game",
    "round": "Round",
    "dealer_no": "Dealer no.",
    "repeat": "Repeat",
    "hig": "Hand in game",
    "hand": "Hand",
    "east": "East player (dealer)",
    "south": "South player",
    "west": "West player",
    "north": "North player",
    "result": "Result",
    "winner": "Winner",
    "thrower": "Threw the winning tile",
    "tai": "Tai",
    "raw": "Tai before 5-tai cap",
    "how": "Where the tai came from",
    "pe": "East points",
    "ps": "South points",
    "pw": "West points",
    "pn": "North points",
    "turns": "Turns",
}


def style(name: str) -> str:
    return STYLE_NAMES.get(name, name)  # evolved bots keep their own names


def wind_of(seat: int, dealer: int) -> str:
    return WINDS[(seat - dealer) % 4]


def patterns(text: str) -> str:
    """'Men qing (fully concealed):1|Animal:1|Animal:1' -> 'Men qing (fully concealed) +1, Animal x2 +2'."""
    if not isinstance(text, str) or not text:
        return ""
    totals: dict[str, list[int]] = {}
    for item in text.split("|"):
        name, tai = item.rsplit(":", 1)
        totals.setdefault(name, []).append(int(tai))
    parts = []
    for name, tais in totals.items():
        parts.append(f"{name} +{tais[0]}" if len(tais) == 1 else f"{name} x{len(tais)} +{sum(tais)}")
    return ", ".join(parts)


def readable(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    if "game" not in h.columns:  # older runs: dealer passed every hand
        h["game"] = h.hand_id // 16 + 1
        h["dealer_no"] = h.hand_id % 4 + 1
        h["repeat"] = 0
        h["hand_in_game"] = h.hand_id % 16 + 1
    rows = []
    for r in h.itertuples(index=False):
        dealer = int(r.dealer)
        bots = [r.bot0, r.bot1, r.bot2, r.bot3]
        deltas = [r.d0, r.d1, r.d2, r.d3]
        # seat that sits at each wind this hand
        seat_at = {wind_of(s, dealer): s for s in range(4)}
        label = lambda s: f"{wind_of(s, dealer)} ({style(bots[s])})"
        if r.winner < 0:
            result, winner, thrower = "Draw (no winner)", "-", "-"
        elif r.self_draw:
            result, winner, thrower = "Self-drawn win", label(r.winner), "Nobody (self-drawn)"
        else:
            result, winner, thrower = "Won on a discard", label(r.winner), label(r.discarder)
        rows.append({
            COLUMNS["game"]: r.game,
            COLUMNS["round"]: f"{WINDS[int(r.prevailing)]} round",
            COLUMNS["dealer_no"]: r.dealer_no,
            COLUMNS["repeat"]: r.repeat,
            COLUMNS["hig"]: r.hand_in_game,
            COLUMNS["hand"]: f"Game {r.game}, {WINDS[int(r.prevailing)]} round, Dealer {r.dealer_no}"
                             + (f", Repeat {r.repeat}" if r.repeat else ""),
            COLUMNS["east"]: style(bots[seat_at["East"]]),
            COLUMNS["south"]: style(bots[seat_at["South"]]),
            COLUMNS["west"]: style(bots[seat_at["West"]]),
            COLUMNS["north"]: style(bots[seat_at["North"]]),
            COLUMNS["result"]: result,
            COLUMNS["winner"]: winner,
            COLUMNS["thrower"]: thrower,
            COLUMNS["tai"]: r.tai,
            COLUMNS["raw"]: r.raw_tai,
            COLUMNS["how"]: patterns(r.patterns),
            COLUMNS["pe"]: deltas[seat_at["East"]],
            COLUMNS["ps"]: deltas[seat_at["South"]],
            COLUMNS["pw"]: deltas[seat_at["West"]],
            COLUMNS["pn"]: deltas[seat_at["North"]],
            COLUMNS["turns"]: r.turns,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="folder written by `dsy selfplay --out ...`")
    ap.add_argument("--out", help="output file (default: <run_dir>/hands_readable.csv)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "shard_*", "hands.csv")))
    if not files:
        sys.exit(f"no shard_*/hands.csv files in {args.run_dir}")
    h = pd.concat((pd.read_csv(f, keep_default_na=False, dtype={"seed": str}) for f in files), ignore_index=True)
    h = h.sort_values("hand_id")  # full-game ids sort by game, then hand
    out = readable(h)
    path = args.out or os.path.join(args.run_dir, "hands_readable.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel reads it correctly
    print(f"wrote {len(out):,} hands to {path}")


if __name__ == "__main__":
    main()
