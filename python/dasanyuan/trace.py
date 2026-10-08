"""Full game traces for the replay viewer (web/replay)."""
from __future__ import annotations

import base64
from typing import List

from . import obs, selfplay
from .game import CLAIM, ROB_KONG, SELF_TURN, Game, HandResult, Player
from .obs import N_ACTIONS
from .scoring import MeldKind
from .selfplay import SelfPlayConfig

_MELD_KIND = {MeldKind.CHOW: "chow", MeldKind.PONG: "pong", MeldKind.EXPOSED_KONG: "kong",
              MeldKind.ADDED_KONG: "added_kong", MeldKind.CONCEALED_KONG: "concealed_kong"}


def player_json(p: Player) -> dict:
    hand = [k for k, n in enumerate(p.hand) for _ in range(n)]
    return {
        "hand": hand,
        "bonus": list(p.bonus),
        "melds": [{"kind": _MELD_KIND[m.kind], "tiles": m.tiles(), "from": m.from_} for m in p.melds],
        "discards": [[d.tile, d.claimed, d.tsumogiri] for d in p.discards],
    }


def table_json(g: Game) -> list:
    return [player_json(p) for p in g.players]


def decision_json(g: Game, seat: int, a: int) -> dict:
    ph = g.phase
    if ph.kind == SELF_TURN:
        phase, tile, frm = "own_turn", ph.drawn, None
    elif ph.kind == CLAIM:
        phase, tile, frm = "claim", ph.tile, ph.seat
    elif ph.kind == ROB_KONG:
        phase, tile, frm = "rob_kong", ph.tile, ph.seat
    else:
        phase, tile, frm = "over", None, None
    o = obs.encode_obs(g, seat)
    waits, sh = obs.encode_labels(g, seat)
    mask = obs.legal_mask(g, seat)
    return {
        "seat": seat, "phase": phase, "tile": tile, "from": frm,
        "legal": [i for i in range(N_ACTIONS) if mask[i] == 1],
        "action": obs.action_index(a), "draws_left": g.draws_left(), "turn": g.turns,
        "table": table_json(g),
        "waits": [list(waits[i:i + 34]) for i in range(0, len(waits), 34)],
        "shanten": sh, "obs": base64.b64encode(bytes(o)).decode("ascii"),
    }


def result_json(r: HandResult) -> dict:
    score = None
    if r.score is not None:
        score = {"tai": r.score.tai, "raw_tai": r.score.raw_tai,
                 "patterns": [[p.label(), t] for p, t in r.score.items]}
    return {"winner": r.winner, "discarder": r.discarder, "self_draw": r.self_draw,
            "deltas": list(r.deltas), "turns": r.turns, "score": score}


def trace_game(cfg: SelfPlayConfig, game_no: int) -> List[dict]:
    """Every hand of full game `game_no` (0-based), replayed exactly as `selfplay --games` played it."""
    ids = selfplay.game_lineup(cfg, game_no)
    styles = [cfg.styles[i].name for i in ids]
    decisions: List[dict] = []
    hands: List[dict] = []

    def on_hand(hand_id, seed, hcfg, label, r, end):
        ds = list(decisions)
        decisions.clear()
        hands.append({
            "hand_id": hand_id, "seed": str(seed),
            "dealer": hcfg.dealer, "prevailing": hcfg.prevailing_wind,
            "game": label.game, "dealer_no": label.dealer_no, "repeat": label.repeat,
            "hand_in_game": label.hand_in_game,
            "styles": styles, "initial": ds[0]["table"] if ds else None, "decisions": ds,
            "final": table_json(end), "result": result_json(r),
        })

    selfplay.play_full_game(cfg, game_no, lambda g, seat, a, _: decisions.append(decision_json(g, seat, a)), on_hand)
    return hands


def trace_hand(cfg: SelfPlayConfig, hand_id: int) -> dict:
    """One dataset hand, replayed exactly, with everything recorded at each decision."""
    seed = selfplay.hand_seed(cfg.seed, hand_id)
    gcfg = selfplay.hand_config(hand_id)
    ids, bots = selfplay.bots_for(cfg, hand_id)
    start = Game(gcfg.clone(), seed)
    initial = table_json(start)
    decisions: List[dict] = []
    end, result = selfplay.play_hand(start, bots, lambda g, seat, a: decisions.append(decision_json(g, seat, a)))
    return {
        "hand_id": hand_id, "seed": str(seed), "dealer": gcfg.dealer, "prevailing": gcfg.prevailing_wind,
        "styles": [cfg.styles[i].name for i in ids], "initial": initial, "decisions": decisions,
        "final": table_json(end), "result": result_json(result),
    }
