"""`dsy` command-line tool: self-play data generation, duplicate tournaments, a simple
evolutionary loop over bot styles, a hand scorer for checking rules, the scenario bank,
replay traces and per-move expected values.

Run as `python -m dasanyuan <command>` or, once installed, `dsy <command>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import List

from . import selfplay
from .bots import Style, make_bot
from .game import CLAIM, ROB_KONG, SELF_TURN, TSUMO, RON, PONG, EXPOSED_KONG, PASS
from .obs import OBS_LEN, OBS_VERSION, View, action_index
from .rng import MASK, Rng
from .scenario import ScenarioError, load_dir
from .scoring import MIN_TAI, Meld, MeldKind, WinContext, score_hand, unit_points
from .selfplay import SelfPlayConfig
from .tile import counts_of, is_bonus, parse_tiles, tile_name

DEFAULT_STYLES = "fast,high_tai,defensive,balanced"


def parse_styles(s: str) -> List[Style]:
    """Comma-separated style names / CSV lines, or a styles CSV file (as written to bots.csv)."""
    if os.path.exists(s):
        out = []
        with open(s) as f:
            for line in f.read().splitlines():
                if not line.strip() or line.startswith("bot_id") or line.startswith("#"):
                    continue
                # accept either "name,..." or "id,name,..."
                st = Style.from_csv(line) or Style.from_csv(line.split(",", 1)[1] if "," in line else "")
                if st is None:
                    sys.exit(f"bad style line: {line}")
                out.append(st)
        return out
    out = []
    for n in s.split(","):
        st = Style.by_name(n.strip()) or Style.from_csv(n)
        if st is None:
            sys.exit(f"unknown style {n}")
        out.append(st)
    return out


def parse_melds(s: str) -> List[Meld]:
    out = []
    for m in s.split(","):
        if not m.strip():
            continue
        if ":" not in m:
            sys.exit("meld must be kind:tiles")
        kind, _, tiles = m.partition(":")
        t = parse_tiles(tiles)
        if kind == "chow":
            out.append(Meld(MeldKind.CHOW, min(t)))
        elif kind in ("pong", "kong", "ckong"):
            out.append(Meld({"pong": MeldKind.PONG, "kong": MeldKind.EXPOSED_KONG,
                             "ckong": MeldKind.CONCEALED_KONG}[kind], t[0]))
        else:
            sys.exit(f"unknown meld kind {kind}")
    return out


def parse_game_list(s: str) -> List[int]:
    v = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        v.extend(range(int(a), int(b or a) + 1))
    return v


def move_label(a: int) -> str:
    if a < 34:
        return f"discard {tile_name(a)}"
    if a < 68:
        return f"kong {tile_name(a - 34)}"
    if 72 <= a <= 74:
        return f"chow ({['low', 'middle', 'high'][a - 72]} of the run)"
    return {TSUMO: "self-drawn win", RON: "win on discard", PONG: "pong", EXPOSED_KONG: "kong from discard",
            PASS: "pass"}[a]


def _phase_name(g) -> str:
    return {SELF_TURN: "own_turn", CLAIM: "claim", ROB_KONG: "rob_kong"}.get(g.phase.kind, "over")


def evaluate_games(games: str, seed: int, styles: str, fixed_lineup: bool, worlds: int, oracle: bool, out: str):
    """Writes one CSV row per decision: the expected points of the move played, of the best move,
    the regret between them, and the expected points of every legal move."""
    from .evaluate import evaluate
    cfg = SelfPlayConfig(hands=0, seed=seed, styles=parse_styles(styles), random_lineup=not fixed_lineup,
                         shard_size=100, out_dir="", compress=False, games=0)
    t0 = time.time()
    n_dec = 0
    with open(out, "w", newline="\n") as f:
        f.write("game,hand_id,idx,seat,style,phase,n_legal,chosen_action,chosen_move,chosen_ev,best_action,best_move,"
                "best_ev,regret,regret_se,hand_points,all_moves\n")
        for game_no in parse_game_list(games):
            ids = selfplay.game_lineup(cfg, game_no - 1)
            names = [cfg.styles[i].name for i in ids]
            lineup = [cfg.styles[i].clone() for i in ids]
            rows = []
            idx = [0]

            def on_decision(g, seat, a, hand_id):
                vals = evaluate(g, seat, lineup, a, worlds, seed ^ ((hand_id * 0x9E37) & MASK) ^ idx[0], oracle)
                chosen = next(v for v in vals if v.action == a)
                best = vals[0]
                for v in vals[1:]:
                    if v.ev >= best.ev:  # last maximum, like Iterator::max_by
                        best = v
                all_moves = "; ".join(f"{move_label(v.action)}:{v.ev:+.2f}" for v in vals)
                line = (f"{game_no},{hand_id},{idx[0]},{seat},{names[seat]},{_phase_name(g)},{len(vals)},"
                        f"{action_index(a)},{move_label(a)},{chosen.ev:.3f},{action_index(best.action)},"
                        f"{move_label(best.action)},{best.ev:.3f},{best.ev - chosen.ev:.3f},{best.se_vs_chosen:.3f}")
                rows.append((seat, line, all_moves))
                idx[0] += 1

            def on_hand(_h, _s, _c, _l, r, _e):
                nonlocal n_dec
                for seat, line, all_moves in rows:
                    f.write(f'{line},{r.deltas[seat]},"{all_moves}"\n')
                    n_dec += 1
                rows.clear()
                idx[0] = 0

            selfplay.play_full_game(cfg, game_no - 1, on_decision, on_hand)
            print(f"game {game_no}: {n_dec} decisions so far, {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"wrote {n_dec} decisions to {out} ({time.time() - t0:.0f}s)")


def print_table(stats) -> None:
    print(f"{'style':<14} {'pts/hand':>10} {'±se':>8} {'win%':>8} {'tsumo%':>8} {'deal-in%':>9} {'avg tai':>8}")
    for s in stats:
        h = max(s.hands, 1)
        print(f"{s.name:<14} {s.mean():>10.3f} {s.stderr():>8.3f} {100.0 * s.wins / h:>7.1f}% "
              f"{100.0 * s.self_draws / h:>7.1f}% {100.0 * s.deal_ins / h:>8.1f}% {s.tai_sum / max(s.wins, 1):>8.2f}")


def cmd_selfplay(a) -> None:
    t = time.time()
    shard_size = a.shard_size or (100 if a.games > 0 else 2_000)
    cfg = SelfPlayConfig(hands=a.hands, seed=a.seed, styles=parse_styles(a.styles), random_lineup=not a.fixed_lineup,
                         shard_size=shard_size, out_dir=a.out, compress=not a.no_compress, games=a.games)
    s = selfplay.run_selfplay(cfg)
    secs = time.time() - t
    if a.games > 0:
        print(f"{a.games} full games, {s.hands / a.games:.1f} hands per game on average.")
    print(f"{s.hands} hands, {s.decisions} decisions in {secs:.1f}s ({s.hands / secs:.0f} hands/s). "
          f"Wins {100.0 * s.wins / s.hands:.1f}%, draws {100.0 * s.draws / s.hands:.1f}%. Data in {a.out}")


def cmd_tournament(a) -> None:
    st = parse_styles(a.styles)
    if len(st) != 4:
        sys.exit("tournament needs exactly 4 styles")
    t = time.time()
    stats = selfplay.duplicate_tournament(st, a.walls, a.seed)
    print(f"{a.walls} walls x 4 rotations = {a.walls * 4} hands in {time.time() - t:.1f}s")
    print_table(stats)


def cmd_evolve(a) -> None:
    champ = parse_styles(a.start)[0]
    champ.name = "gen0"
    rng = Rng(a.seed)
    d = os.path.dirname(a.log)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(a.log, "w", newline="\n") as f:
        f.write("generation,name,speed,value,defence,claim,flush,pongs,pts_per_hand,stderr\n")
        for gen in range(1, a.generations + 1):
            entrants = [champ.clone()] + [champ.mutate(rng, a.sigma, f"gen{gen}_m{i}") for i in range(1, 4)]
            stats = selfplay.duplicate_tournament(entrants, a.walls, (a.seed + gen * 1000) & MASK)
            # Challenger must beat the champion by more than ~2 standard errors to take over.
            champ_score = stats[0].mean()
            bi = 1
            for i in range(2, 4):
                if stats[i].mean() >= stats[bi].mean():  # last maximum, like Iterator::max_by
                    bi = i
            best = stats[bi]
            margin = 2.0 * (best.stderr() ** 2 + stats[0].stderr() ** 2) ** 0.5
            promoted = best.mean() - champ_score > margin
            print(f"gen {gen:>3}: champion {champ_score:+.3f}, best challenger {best.name} {best.mean():+.3f} "
                  f"(margin {margin:.3f}) {'-> PROMOTED' if promoted else ''}")
            for i, s in enumerate(stats):
                f.write(f"{gen},{entrants[i].to_csv()},{s.mean():.4f},{s.stderr():.4f}\n")
            f.flush()
            if promoted:
                champ = entrants[bi].clone()
    print(f"final champion: {champ.to_csv()}")


def cmd_score(a) -> None:
    tiles = parse_tiles(a.hand)
    bonus = [t for t in tiles if is_bonus(t)]
    ctx = WinContext(seat_wind=a.seat, prevailing_wind=a.prevailing, self_draw=a.tsumo,
                     win_tile=parse_tiles(a.win)[0])
    s = score_hand(counts_of(tiles), parse_melds(a.melds), bonus, ctx)
    if s is None:
        print("Not a winning hand.")
        return
    for p, t in s.items:
        print(f"  {p.label():<34} {t} tai")
    print(f"Total: {s.tai} tai (raw {s.raw_tai}){'  -- below minimum, cannot win' if s.tai < MIN_TAI else ''}")
    if s.tai >= 1:
        if a.tsumo:
            print(f"Self-draw: each other player pays {2 * unit_points(s.tai)}")
        else:
            print(f"Discard win: discarder pays {4 * unit_points(s.tai)}")


def cmd_scenarios(a) -> None:
    try:
        items = load_dir(a.dir)
    except ScenarioError as e:
        sys.exit(str(e))
    invalid = 0
    for file, sc in items:
        try:
            sc.validate()
        except ScenarioError as e:
            print(f"INVALID {file}:{sc.name} -> {e}")
            invalid += 1
    print(f"{len(items)} scenarios loaded, {invalid} invalid. Pass rate over {a.trials} runs per bot.")
    if invalid:
        sys.exit(1)
    names = [x.strip() for x in a.bots.split(",")]
    print(f"\n{'scenario':<30}" + "".join(f"{n:>11}" for n in names))
    totals = [0.0] * len(names)
    by_tag = defaultdict(lambda: [[0.0, 0] for _ in names])
    for _, sc in items:
        row = f"{sc.name:<30}"
        notes = []
        for bi, n in enumerate(names):
            g = sc.build()
            passes = 0
            first_err = None
            for seed in range(1, a.trials + 1):
                bot = make_bot(n, seed)
                if bot is None:
                    sys.exit(f"unknown bot {n}")
                err = sc.check(g, bot.act(View(g, sc.hero)))
                if err is None:
                    passes += 1
                elif first_err is None:
                    first_err = err
            rate = passes / a.trials
            totals[bi] += rate
            if first_err is not None:
                notes.append(f"    {n}: {first_err}")
            for t in sc.tags:
                by_tag[t][bi][0] += rate
                by_tag[t][bi][1] += 1
            row += f"{100.0 * rate:>10.0f}%"
        print(row)
        if a.verbose:
            for x in notes:
                print(x)
    n = len(items)
    print(f"{'OVERALL':<30}" + "".join(f"{100.0 * t / n:>10.0f}%" for t in totals))
    for tag in sorted(by_tag):
        print(f"{'  [' + tag + ']':<30}" + "".join(f"{100.0 * x / c:>10.0f}%" for x, c in by_tag[tag]))


def cmd_trace(a) -> None:
    from .trace import trace_game, trace_hand
    cfg = SelfPlayConfig(hands=0, seed=a.seed, styles=parse_styles(a.styles), random_lineup=not a.fixed_lineup,
                         shard_size=a.shard_size, out_dir="", compress=False, games=0)
    if a.game is not None:
        games = [h for g in a.game.split(",") for h in trace_game(cfg, int(g.strip()) - 1)]
    else:
        games = [trace_hand(cfg, int(h.strip())) for h in a.hands.split(",")]
    doc = {"format": 1, "obs_version": OBS_VERSION, "obs_len": OBS_LEN,
           "run": {"seed": a.seed, "shard_size": a.shard_size, "random_lineup": not a.fixed_lineup},
           "games": games}
    with open(a.out, "w") as f:
        f.write(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    print(f"wrote {len(games)} hand(s) to {a.out}")


def cmd_bench(a) -> None:
    t = time.time()
    stats = selfplay.duplicate_tournament(Style.archetypes(), a.hands // 4, 1)
    secs = time.time() - t
    print(f"{a.hands} hands in {secs:.2f}s = {a.hands / secs:.0f} hands/s on {selfplay.workers()} processes")
    print_table(stats)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--threads", type=int, default=None, help="Worker processes (default: all cores)")
    p = argparse.ArgumentParser(prog="dsy", description="da-san-yuan: Singapore mahjong engine tools",
                                parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selfplay", parents=[common], help="Generate self-play training data as .npy shards.")
    s.add_argument("--games", type=int, default=0,
                   help="Play this many complete games (East to North round, real dealer rules). Overrides --hands.")
    s.add_argument("--hands", type=int, default=10_000,
                   help="Independent hands with the dealer passing every hand (used when --games is 0).")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--styles", default=DEFAULT_STYLES,
                   help="Comma-separated styles (fast, high_tai, defensive, balanced) or a styles CSV file.")
    s.add_argument("--fixed-lineup", action="store_true",
                   help="Keep styles fixed by seat instead of sampling a random lineup each hand (or game).")
    s.add_argument("--shard-size", type=int, default=None,
                   help="Hands per shard, or games per shard with --games (default 2,000 hands / 100 games).")
    s.add_argument("--out", default="data/selfplay")
    s.add_argument("--no-compress", action="store_true",
                   help="Write plain .npy instead of gzip-compressed .npy.gz (about 15x larger on disk).")
    s.set_defaults(fn=cmd_selfplay)

    s = sub.add_parser("tournament", parents=[common], help="Duplicate-format tournament between exactly 4 styles.")
    s.add_argument("--styles", default=DEFAULT_STYLES)
    s.add_argument("--walls", type=int, default=5_000, help="Number of walls; each is played 4 times with seats rotated.")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_tournament)

    s = sub.add_parser("evolve", parents=[common],
                       help="Evolve styles: each generation, the champion plays 3 mutated copies of itself.")
    s.add_argument("--generations", type=int, default=20)
    s.add_argument("--walls", type=int, default=3_000)
    s.add_argument("--sigma", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--start", default="balanced", help="Starting style (or a CSV line from a previous run).")
    s.add_argument("--log", default="data/evolve.csv")
    s.set_defaults(fn=cmd_evolve)

    s = sub.add_parser("score", parents=[common], help="Score a winning hand under the team ruleset.")
    s.add_argument("hand", help='Concealed tiles including the winning tile, plus bonus tiles, e.g. "123m456p789s11z55z f1"')
    s.add_argument("--win", required=True, help="Winning tile, e.g. 5z")
    s.add_argument("--melds", default="", help='Exposed melds, e.g. "pong:5z,chow:123m,kong:1z,ckong:9p"')
    s.add_argument("--seat", type=int, default=0, help="Seat wind 0=E 1=S 2=W 3=N")
    s.add_argument("--prevailing", type=int, default=0)
    s.add_argument("--tsumo", action="store_true")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("scenarios", parents=[common],
                       help="Run the scenario bank: validate positions and score bots on them.")
    s.add_argument("--dir", default="scenarios")
    s.add_argument("--bots", default="random,efficiency,fast,high_tai,defensive,balanced",
                   help='Comma-separated bots to test (styles, "efficiency" or "random").')
    s.add_argument("--verbose", action="store_true", help="Print every scenario result, not just the summary.")
    s.add_argument("--trials", type=int, default=20, help="Runs per bot with different bot seeds.")
    s.set_defaults(fn=cmd_scenarios)

    s = sub.add_parser("trace", parents=[common],
                       help="Export complete hands from a self-play run as JSON for the replay viewer.")
    s.add_argument("--game", default=None,
                   help="Export every hand of these full games (1-based, comma-separated), from a `selfplay --games` run.")
    s.add_argument("--hands", default="0", help="Hand ids to export from an independent-hands run (0-based).")
    s.add_argument("--seed", type=int, default=1, help="Seed of the run to reproduce.")
    s.add_argument("--shard-size", type=int, default=2_000)
    s.add_argument("--styles", default=DEFAULT_STYLES)
    s.add_argument("--fixed-lineup", action="store_true")
    s.add_argument("--out", default="replay.json")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("evaluate", parents=[common],
                       help="Expected points of every legal move at every decision of some full games.")
    s.add_argument("--games", default="1", help='Games to evaluate, 1-based, e.g. "1-10" or "1,5,9".')
    s.add_argument("--seed", type=int, default=1, help="Seed of the `selfplay --games` run to reproduce.")
    s.add_argument("--styles", default=DEFAULT_STYLES)
    s.add_argument("--fixed-lineup", action="store_true")
    s.add_argument("--worlds", type=int, default=64,
                   help="Sampled worlds per move; the error shrinks with the square root (64 is about +-1 point).")
    s.add_argument("--oracle", action="store_true",
                   help="Use the real hidden tiles instead of re-dealing what the player cannot see.")
    s.add_argument("--out", default="move_values.csv")
    s.set_defaults(fn=lambda a: evaluate_games(a.games, a.seed, a.styles, a.fixed_lineup, a.worlds, a.oracle, a.out))

    s = sub.add_parser("bench", parents=[common], help="Measure engine + bot throughput.")
    s.add_argument("--hands", type=int, default=2_000)
    s.set_defaults(fn=cmd_bench)
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    if a.threads:
        selfplay.WORKERS = a.threads
    a.fn(a)


if __name__ == "__main__":
    main()
