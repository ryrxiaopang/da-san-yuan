# Design: from rules engine to a reinforcement-learning advisor

This document is the plan for the learning system and the critical review of what is still missing. Read it with [RULES.md](../RULES.md) and the [README](../README.md).

## 1. Where the machine learning is

The project uses three kinds of learning. Only one of them is reinforcement learning, and the project should be able to prove that part works on its own.

| Component | Learning type | Learns from | Status |
|---|---|---|---|
| Heuristic bots + evolution (`dsy evolve`) | Black-box optimisation of 6 weights | Duplicate-tournament points | Built |
| Behaviour cloning (BC) of the policy | Supervised | `obs` → `action` from bot games | Data ready |
| Opponent-reading heads | Supervised | `obs` → `waits`, `shanten`, `oracle` labels | Data ready |
| **Policy improvement (PPO self-play league)** | **Reinforcement learning** | **Points won or lost at the end of each hand** | Environment ready; trainer next |
| Tile recognition (phase 2) | Supervised computer vision | Labelled photos | Not started |

**What makes it RL:** BC can only copy the heuristic bots, so it can never beat them by much. PPO changes the policy using nothing but the reward (points) from games it plays itself. The proof is a model that beats the BC model it started from, and beats the bots it learned from, in duplicate tournaments with error bars (section 5).

## 2. The network

One shared network, several heads:

```
obs (672 bytes, v2)
  ├─ tile planes: hand, melds, discards, unseen counts  →  1-D conv / MLP over 34 tile kinds
  ├─ discard sequences + turn order (4 x 32)             →  small transformer or GRU
  └─ meta (winds, wall, phase, target tile)              →  embedding
           ↓ shared trunk
  ├─ policy head      76 logits, illegal actions masked to -inf
  ├─ value head       expected points for this seat at the end of the hand
  ├─ afterstate head  expected points after each candidate discard (for explanations, §4)
  └─ opponent heads   for each opponent: P(ready), P(wins on tile k) for 34 tiles,
                      expected tai on each tile, hand-shape guesses (flush suit, pongs)
```

The opponent heads are trained with supervised losses against the exact labels the engine writes (`waits`, `shanten`, `oracle`). They keep training as an auxiliary loss during PPO, because the simulator always knows the true hands. They make defence learnable faster and explainable.

## 3. Training stages

1. **Generate data** (`dsy selfplay`). Mix of the four styles plus evolved champions. 1–2 million hands; about 6 KB per hand compressed.
2. **Supervised pre-training.** BC loss on actions plus opponent-head losses. Claims are only ~4% of decisions, so weight the claim decisions (or give them their own head), or the model learns to always pass.
3. **PPO self-play league** (`dasanyuan.Env`):
   - The learner plays some seats; the others are drawn from a pool of past checkpoints and the heuristic bots. Pure self-play against identical copies tends to collapse into habits that only work against itself.
   - Reward: this seat's **tai points only** (the payment table in RULES.md) at the end of the hand. Kong and animal instant payments are excluded by team decision: they are rare, small, and not what the platform teaches. The model still learns when kongs help through their effect on winning (replacement draw, kong-replacement tai, robbing risk). Points are scaled down (e.g. divided by 16, or a signed log) because a 64-point deal-in would otherwise swamp everything. Draws give 0. γ = 1, GAE λ ≈ 0.95.
   - **Asymmetric critic:** the value head may see the hidden hands (`oracle`) during training, the policy may not. This reduces variance a lot in imperfect-information games, and it's free here because the simulator knows everything. At play time only the policy is used.
   - Entropy bonus for exploration; keep the KL to the BC policy small at first so the agent doesn't forget how to play before RL has improved it.
   - Promote a checkpoint into the pool only when it beats the current pool in a duplicate tournament by more than 2 standard errors.
4. **Human fine-tuning** (later): games from phase-1 users correct for the fact that bots and people discard differently.

**Throughput.** The raw Python environment runs ~178,000 decisions/s on one thread (without a policy); a heuristic-bot game runs ~20,000 decisions/s through Python. Network inference will be the bottleneck, so step 64–256 environments together and batch the forward pass on the GPU. A batched environment in Rust is an optimisation for later, not a blocker.

## 4. Explanations that stay faithful

The advice shown to users must come from what the model actually computed:

| Explanation | Source |
|---|---|
| "Discarding 7p keeps you at +2.1 points; discarding East drops you to −0.4" | Afterstate value head: the value of the position right after each candidate discard |
| "The left player is probably ready (78%) and likely waiting on 3s/6s" | Opponent heads |
| "There are 5 unseen 3s; about 31% to draw one in your next 4 draws" | Exact counting, no model |
| "West is completely safe: all other copies are visible" | Exact counting (true only for honour tiles, see below) |

PPO alone trains a single state value V(s), not one value per action, so the afterstate head must be designed in from the start. Without it the model can choose but can't say why one option beats another.

**Safety facts to teach correctly:**
- There is no rule against winning on a tile you discarded earlier, so an opponent's old discard is only *safer*, never safe.
- A tile with every other copy visible is completely safe **only for honours**. For a suited tile it rules out pair and pong waits but not a sequence wait (an opponent holding 4-5 can still win on the last 3 or 6).

## 5. Evaluation (what to report)

| Measure | Tool | Shows |
|---|---|---|
| Points per hand ± standard error vs fixed baselines | `dsy tournament` (duplicate format) | Overall strength; RL beating BC beating heuristics |
| Learning curves over training steps | tournament against frozen pool | That RL is actually improving |
| Deal-in choice rate in dangerous spots | `waits` labels (`examples/inspect_data.py`) | Defence. Current bots: 5–9% vs 14% for a random discard |
| Opponent-head accuracy (ready / wait AUC) | held-out labelled data | Reading skill |
| Scenario bank pass rates by tag | `dsy scenarios`, `dasanyuan.Scenarios` | Specific skills: folding, flush reading, honour counting, pushing, tai planning |
| Ablation without opponent heads | retrain | Whether reading matters |

The scenario bank currently separates the baselines clearly: the pure-efficiency bot passes 20% of defence scenarios, the heuristic bots 60–80%, and every heuristic bot fails "big three dragons threat" (it doesn't read a dragon threat). The learned model should pass all of them while keeping its push and efficiency scores.

Use held-out seeds for every reported number, and never tune on the scenarios you report.

## 6. Critical review: gaps to settle before moving on

### Must decide now

1. **Phase 2 can't see what phase 1 sees.** In a real game the user photographs their hand and types the discards. They usually won't know the discard order, who threw the drawn tile straight back (tsumogiri), or the exact turn. A model trained on full sequences will quietly get worse. Fix it from the start: during training, randomly blank the sequence and turn fields (keep the counts), so one model works with either. Also design the phase-2 input screen to capture as much as is practical (per-player discards, melds, bonus tiles).
2. **Recognition errors.** Phase 2's tile detector will sometimes misread a tile. Train with some noise (a wrong tile now and then) and show the user what was recognised so they can correct it before advice is given.
3. **How the website runs the engine.** The Rust engine compiles to WebAssembly, so the exact same rules can run in the browser for practice games, with the server only for friend games. Decide this before writing game logic in another language twice.
4. **How the website runs the model.** Plan to export the trained PyTorch network to ONNX and run it with onnxruntime (server) or onnxruntime-web (browser). Keep the network small enough for that (a few million parameters).
5. **Observation versioning.** Every dataset records `obs_version` (now 2). Models must record which version they were trained on.

### Rule decisions (settled 1 Oct 2026)

- **Instant kong and animal payments:** real tables pay them, but they stay out of the training reward. Both become switchable rules for the website's score display; amounts to be confirmed when that is built.
- **Concealed kongs** are face up, so the tile is public (as encoded).
- **Earthly hand** = a non-dealer winning on their own first draw (as encoded).
- Still as encoded, raise if your tables differ: no kong straight after a pong; no robbing a concealed kong.

### Known weaknesses in the current baselines

- The heuristic bots only partly read threats: exposed melds, wall length, recent discards and flush suits. They don't count honours for dragon or wind threats. That's fine; it's what the learned model is meant to fix, and the scenario bank measures it.
- BC data inherits the bots' habits, including weak defence. RL is what removes them, which is another reason to measure RL against the BC model.

### Practical

- On the laptop: rule work, data generation, scenario benchmarking and small BC sanity runs all work on CPU. Leave real training for the 5070.
- Start collecting tile photos now: 46 classes (34 playing tiles + 12 bonus), several physical sets, varied light and angles. Labelling takes longer than training.
- Split the team by track: engine/RL, computer vision, website.
