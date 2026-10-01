"""Summarise a self-play dataset: decision mix, outcomes per bot, and tai patterns.

    python examples/inspect_data.py data/selfplay
"""
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np

import dasanyuan as dsy

root = sys.argv[1] if len(sys.argv) > 1 else "data/selfplay"
data = dsy.load_shards(root)
meta, act, mask = data["meta"], data["action"], data["mask"]
n = len(act)
print(f"{n:,} decisions from {len(np.unique(meta[:, 0])):,} hands")

# How many decisions were real choices (more than one legal action)?
choices = mask.sum(1)
print(f"decisions with >1 legal action: {100 * (choices > 1).mean():.1f}%  (avg {choices.mean():.1f} options)")

kinds = Counter()
for a in act:
    kinds[dsy.action_name(int(a)).split()[0]] += 1
print("action mix:", ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in kinds.most_common()))

bots = {}
with open(os.path.join(root, "bots.csv")) as f:
    for row in csv.DictReader(f):
        bots[int(row["bot_id"])] = row["name"]

# One row per (hand, seat) for outcome stats
_, first = np.unique(meta[:, 0] * 4 + meta[:, 1], return_index=True)
per_seat = meta[first]
print(f"\n{'bot':<12}{'seat-hands':>11}{'pts/hand':>10}{'win%':>8}{'deal-in%':>10}")
for b, name in bots.items():
    r = per_seat[per_seat[:, 2] == b]
    if len(r):
        print(f"{name:<12}{len(r):>11,}{r[:, 3].mean():>10.3f}{100 * r[:, 4].mean():>7.1f}%{100 * r[:, 5].mean():>9.1f}%")

pats = Counter()
tai = Counter()
for path in glob.glob(os.path.join(root, "shard_*", "hands.csv")):
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["winner"] != "-1":
                tai[int(row["tai"])] += 1
                for p in row["patterns"].split("|"):
                    if p:
                        pats[p.rsplit(":", 1)[0]] += 1
wins = sum(tai.values())
print("\ntai distribution of wins:", {k: f"{100 * v / wins:.1f}%" for k, v in sorted(tai.items())})
print("most common patterns:")
for p, c in pats.most_common(12):
    print(f"  {p:<34}{100 * c / wins:5.1f}% of wins")

# ---- opponent-reading labels: how often is the table dangerous, and do bots avoid it?
waits, shanten = data["waits"], data["shanten"]
ready = shanten[:, 1:] == 0
print(f"\ndecisions where at least one opponent is ready: {100 * ready.any(1).mean():.1f}%")
is_discard = act < 34
d_idx = np.flatnonzero(is_discard)
tile = act[d_idx]
danger = waits[d_idx].max(1)            # [n, 34] max tai over opponents
chosen_danger = danger[np.arange(len(d_idx)), tile] > 0
legal_discards = mask[d_idx, :34] == 1
frac_danger = ((danger > 0) & legal_discards).sum(1) / np.maximum(legal_discards.sum(1), 1)
threat = frac_danger > 0
print(f"discards made while some discard could deal in: {100 * threat.mean():.1f}%")
print(f"  of those, chosen tile dealt in: {100 * chosen_danger[threat].mean():.1f}% "
      f"(random choice would: {100 * frac_danger[threat].mean():.1f}%)")
for b, name in bots.items():
    sel = threat & (meta[d_idx, 2] == b)
    if sel.any():
        print(f"  {name:<12} chose a deal-in tile {100 * chosen_danger[sel].mean():5.1f}% vs random {100 * frac_danger[sel].mean():5.1f}%")
