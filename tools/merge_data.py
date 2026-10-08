"""
Combine several data files from data/collect_data.py into one.

    python tools/merge_data.py greedy_a.pt greedy_b.pt --out greedy_all.pt

Each file numbers its hands from 0, so game_id is shifted for every file after the
first. That keeps every hand's rows together and every game_id unique, so
train_bc.py's split by game_id stays honest. Use a different --seed for each
collection run, or the files will contain the same hands twice.
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts, offset = [], 0
    for path in args.files:
        data = torch.load(path)
        data["game_id"] = data["game_id"] + offset
        offset = int(data["game_id"].max()) + 1
        parts.append(data)
        print(f"{path}: {len(data['labels']):,} discards")

    keys = parts[0].keys()
    merged = {k: torch.cat([p[k] for p in parts]) for k in keys}
    torch.save(merged, args.out)
    print(f"Saved {len(merged['labels']):,} discards from {offset:,} hands to {args.out}")


if __name__ == "__main__":
    main()
