# Singapore Mahjong RL

A bot that learns Singapore mahjong: first by copying a rule-based greedy bot
(behavioral cloning), then by improving through self-play reinforcement learning.

## About the developer

I'm new to machine learning and still learning Python. When working with me:
- Explain ML terms in plain language the first time they come up.
- Prefer simple, readable code with comments over clever or compact code.
- Make one change at a time and tell me how to check it worked.
- If a request skips ahead of the current stage, say so before building it.

## Plan (do the stages in order)

1. **Game engine** — Singapore rules: deal, draw, discard, pong/chow/kong, win
   detection, tai counting, payouts, bonus tiles (flowers, animals).
2. **Rule-based bots** — `RandomBot` (tests the engine, lowest benchmark) and
   `GreedyBot` (teacher + main benchmark).
3. **Data collection** — 4 greedy bots play; every discard is saved as a row.
4. **Behavioral cloning (supervised learning)** — network learns to copy GreedyBot.
5. **Reinforcement learning (PPO)** — start from the BC weights, self-play,
   reward = chips won/lost per hand, KL penalty toward the BC model.
6. **Evaluation** — duplicate mahjong (same shuffled walls, seats rotated),
   measured in average chips per hand vs GreedyBot.

Current status: **stages 1–3 built and tested** (engine, bots, data collection).
Stage 4 (behavioural cloning) in progress: the MLP baseline is trained and measured
(5,000 hands: 61% test accuracy, plateaus after ~7 epochs; at the table it matches
GreedyBot 51% of the time and loses ~1.9 points/hand). Next: the convolutional
model, compared on the same test split. The old Rust project lives on the `old` branch.

## Decisions already made

- **Language:** Python for everything, including the engine. Only rewrite hot
  spots (e.g. shanten) in Rust/Numba later, if profiling shows the engine is
  the bottleneck.
- **Network decides discards only at first.** Claims follow fixed rules:
  always win if allowed; kong > pong; never chow. Hand claims to the network
  later, one at a time (pong/kong first, chow next, win last), as extra
  outputs on the same network with masking.
- **"Greedy"** = picks the discard that keeps the hand closest to winning
  (lowest shanten), no lookahead, no defence, ignores tai.
- **Model:** start with a simple MLP (`DiscardNet`). Switch to a 1-D/2-D
  convolutional ResNet over the 4×9 suit layout only when training on real
  engine data and BC accuracy plateaus. Compare both on the same test split.
- **No human game data.** All training data comes from bots.
- **Rules:** RULES.md is the rulebook (min 1 tai, max 5, payment table, draw at
  15 live tiles, no instant kong/animal payments).
- **Hands:** played one after another with the dealer rule (dealer stays after a
  dealer win or a drawn hand with no kong; otherwise the deal passes; the round
  wind moves on after 4 dealers). Data therefore covers every seat and round wind.
- **GreedyBot details:**
  - Discard: lowest shanten; ties broken by most useful tiles (counting only
    copies the bot holds itself), then by lowest tile number. Fully predictable.
  - Win whenever allowed (at least 1 tai, bonus tiles count).
  - Pong/kong a discard only if the tile scores tai (dragon, own seat wind,
    prevailing wind) or it lowers shanten. Never chow.
  - Always declare own kongs (concealed, or adding to own pong).

## Conventions

### Tile numbering (0–33)
| Range | Tiles |
|---|---|
| 0–8 | 1m–9m (characters) |
| 9–17 | 1p–9p (dots) |
| 18–26 | 1s–9s (bamboo) |
| 27–33 | E, S, W, N, Red, Green, White |

Bonus tiles (flowers, animals) are not among the 34; they're counted separately.

### Players are relative
Every per-player feature is ordered **me, next, opposite, previous** — never
by absolute seat. Seat 0 is East (dealer).

### Input grid: 49 lines × 34 tiles (built by `encode()` in `ml/encode.py`)
| Lines | Feature |
|---|---|
| 0–3 | My hand: have ≥1, ≥2, ≥3, 4 copies |
| 4–19 | Discards, 4 lines per player (me, next, opposite, previous) |
| 20–35 | Exposed melds, 4 lines per player |
| 36–39 | Copies visible anywhere (hand + all discards + melds) |
| 40 | Last tile discarded (one-hot) |
| 41 | My seat wind (one-hot; worked out from who the dealer is) |
| 42 | Prevailing wind (one-hot) |
| 43 | Tiles left in wall ÷ 148 (same value across the line) |
| 44–47 | Bonus tiles shown, one line per player: columns 0–11 = f1–f4, g1–g4, a1–a4 (1 = has it); columns 12–33 always 0 |
| 48 | Dealer: columns 0–3 = me, next, opposite, previous (1 = dealer); rest 0 |

**Never change the line order** once data has been collected — old data and
saved models depend on it. Add new features only at the end, and re-collect.

### Training data file (`greedy_games.pt`, saved with `torch.save`)
| Key | Shape | Notes |
|---|---|---|
| `game_id` | (N,) | used to split train/test by game |
| `states` | (N, 49, 34) float16 | the grid above |
| `masks` | (N, 34) bool | legal discards |
| `labels` | (N,) int 0–33 | tile the bot discarded |
| `seat`, `turn`, `won`, `points` | (N,) | extras for debugging / RL; ignored by BC. `points` = what that seat won or lost in the hand |

One row per discard, from all 4 bots, each from that bot's own point of view.

## Rules that must not be broken

- **No hidden information in the input.** Never encode opponents' hands or the
  wall order. The network must only see what that seat could see.
- **Split train/test by game, not by row.** Rows from the same game are
  near-duplicates; splitting by row inflates test accuracy.
- **Keep the input shape identical from BC to RL** so `bc_model.pt` loads into
  the RL network. Include features (e.g. discards) even if the greedy bot
  ignores them — RL needs them for defence.
- **Always apply the legal-move mask** before softmax/loss.
- **Engine correctness first.** Every rules change gets a test, especially win
  detection and tai counting.

## Files

| File | Purpose |
|---|---|
| `engine/tiles.py` | Tile numbers, names, the 148-tile set, `parse_tiles("123m E E f1")`. |
| `engine/scoring.py` | Win detection, every tai pattern, payments. |
| `engine/game.py` | One hand step by step (`Game`), and the dealer rule between hands (`TableState`). |
| `engine/shanten.py` | Shanten (tiles from ready) and useful tiles. |
| `engine/view.py` | `PlayerView`: what one seat may see. Bots and the network only get this. |
| `engine/test_*.py`, `bots/test_bots.py` | Tests. Run before every commit. |
| `bots/random_bot.py` | `RandomBot`: wins when it can, otherwise random legal moves. Lowest benchmark. |
| `bots/greedy_bot.py` | `GreedyBot`: the teacher, rules under "GreedyBot details" above. |
| `tools/play_bots.py` | Bots play each other; prints points per hand, wins, deal-ins, tai. |
| `tools/crosscheck_old_data.py` | One-off check that the engine (and shanten, with `--shanten`) matches the old Rust engine on its recorded games. |
| `ml/encode.py` | `encode(view)`: the 49 × 34 input grid and the legal-discard mask. |
| `data/collect_data.py` | 4 GreedyBots play hands with the real engine; every discard saved as a row. |
| `ml/test_encode.py` | Checks every grid line, no hidden information, and that the data file loads in `train_bc.py`. |
| `ml/model.py` | The networks (`DiscardNet`, the MLP). Training, bots and RL all import from here. |
| `ml/train_bc.py` | Behavioral cloning. Loads the data file, splits by game, trains `DiscardNet`, saves the best model by test accuracy. |
| `bots/network_bot.py` | `NetworkBot`: a trained model chooses discards; wins/kongs/claims follow GreedyBot's rules. Reports how often it matches GreedyBot. |
| `ml/test_model.py` | Network masking, NetworkBot legality, and a tiny end-to-end training run. |
| `bc_model.pt` | Saved BC weights: `{"state_dict", "num_planes"}`. Loaded by the RL stage. |

Layout (planned parts marked *):
```
engine/   tiles.py, game.py, scoring.py, shanten.py, view.py, test_*.py
bots/     random_bot.py, greedy_bot.py, network_bot.py, test_bots.py
data/     collect_data.py
ml/       encode.py, model.py, train_bc.py, train_ppo.py*, test_encode.py, test_model.py
tools/    crosscheck_old_data.py, play_bots.py
evaluate.py*
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest engine bots ml -q              # all tests (about 1 minute)
python tools/play_bots.py --hands 1000 --bots greedy,greedy,random,random
python data/collect_data.py --games 5000 --out greedy_games.pt
python ml/train_bc.py greedy_games.pt --epochs 20
python tools/play_bots.py --hands 400 --bots greedy,greedy,greedy,net --model bc_model.pt
```

Data size: about 3.4 KB per discard and about 42 discards per hand, so 5,000 hands ≈ 725 MB
(about 10 minutes to collect on one core).

## Glossary

- **BC (behavioral cloning):** supervised learning that copies a teacher's moves.
- **Supervised learning:** every example comes with the correct answer (label).
- **RL / PPO:** learning from rewards (chips) instead of answers; PPO is the algorithm.
- **Agent / environment:** the network / the engine plus the other three players.
- **State / observation:** the full game vs. what one seat can see (the 49×34 grid).
- **Mask:** marks legal actions; illegal ones get probability 0.
- **Adam:** the optimizer that adjusts weights; `lr` is its step size.
- **Shanten:** number of tiles away from a ready hand.
- **Duplicate evaluation:** replay the same walls with rotated seats to cancel out luck.
