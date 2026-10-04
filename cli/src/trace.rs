//! Full game traces for the replay viewer (web/replay).

use base64::Engine as _;
use dsy_engine::game::{Game, Phase, Player};
use dsy_engine::obs::{self, N_ACTIONS, OBS_LEN, SHANTEN_LEN, WAITS_LEN};
use dsy_engine::scoring::MeldKind;
use dsy_engine::selfplay::{self, SelfPlayConfig};
use serde_json::{json, Value};

fn meld_kind(k: MeldKind) -> &'static str {
    match k {
        MeldKind::Chow => "chow",
        MeldKind::Pong => "pong",
        MeldKind::ExposedKong => "kong",
        MeldKind::AddedKong => "added_kong",
        MeldKind::ConcealedKong => "concealed_kong",
    }
}

fn player_json(p: &Player) -> Value {
    let mut hand = Vec::new();
    for (k, &n) in p.hand.iter().enumerate() {
        for _ in 0..n {
            hand.push(k as u8);
        }
    }
    json!({
        "hand": hand,
        "bonus": p.bonus,
        "melds": p.melds.iter().map(|m| json!({"kind": meld_kind(m.kind), "tiles": m.tiles(), "from": m.from})).collect::<Vec<_>>(),
        "discards": p.discards.iter().map(|d| json!([d.tile, d.claimed, d.tsumogiri])).collect::<Vec<_>>(),
    })
}

fn table_json(g: &Game) -> Value {
    json!(g.players.iter().map(player_json).collect::<Vec<_>>())
}

fn decision_json(g: &Game, seat: u8, a: dsy_engine::game::Action) -> Value {
    let b64 = base64::engine::general_purpose::STANDARD;
    let (phase, tile, from) = match &g.phase {
        Phase::SelfTurn { drawn, .. } => ("own_turn", *drawn, None),
        Phase::Claim { from, tile } => ("claim", Some(*tile), Some(*from)),
        Phase::RobKong { seat, tile } => ("rob_kong", Some(*tile), Some(*seat)),
        Phase::Over => ("over", None, None),
    };
    let mut o = vec![0u8; OBS_LEN];
    obs::encode_obs(g, seat, &mut o);
    let mut waits = vec![0u8; WAITS_LEN];
    let mut sh = vec![0i8; SHANTEN_LEN];
    obs::encode_labels(g, seat, &mut waits, &mut sh);
    let mask = obs::legal_mask(g, seat);
    let legal: Vec<usize> = (0..N_ACTIONS).filter(|&i| mask[i] == 1).collect();
    json!({
        "seat": seat, "phase": phase, "tile": tile, "from": from, "legal": legal,
        "action": obs::action_index(a), "draws_left": g.draws_left(), "turn": g.turns,
        "table": table_json(g),
        "waits": waits.chunks(34).map(|c| c.to_vec()).collect::<Vec<_>>(),
        "shanten": sh, "obs": b64.encode(&o),
    })
}

fn result_json(result: &dsy_engine::game::HandResult) -> Value {
    let score = result.score.as_ref().map(|s| {
        json!({
            "tai": s.tai, "raw_tai": s.raw_tai,
            "patterns": s.items.iter().map(|(p, t)| json!([p.name(), t])).collect::<Vec<_>>(),
        })
    });
    json!({
        "winner": result.winner, "discarder": result.discarder, "self_draw": result.self_draw,
        "deltas": result.deltas, "turns": result.turns, "score": score,
    })
}

/// Every hand of full game `game_no` (0-based), replayed exactly as `selfplay --games` played it.
pub fn trace_game(cfg: &SelfPlayConfig, game_no: u64) -> Vec<Value> {
    let ids = selfplay::game_lineup(cfg, game_no);
    let styles: Vec<String> = ids.iter().map(|&i| cfg.styles[i].name.clone()).collect();
    let decisions = std::cell::RefCell::new(Vec::new());
    let mut hands = Vec::new();
    selfplay::play_full_game(
        cfg,
        game_no,
        |g, seat, a, _| decisions.borrow_mut().push(decision_json(g, seat, a)),
        |hand_id, seed, hcfg, label, r, end| {
            let ds: Vec<Value> = decisions.borrow_mut().drain(..).collect();
            let initial = ds.first().map(|d| d["table"].clone()).unwrap_or(Value::Null);
            hands.push(json!({
                "hand_id": hand_id, "seed": seed.to_string(),
                "dealer": hcfg.dealer, "prevailing": hcfg.prevailing_wind,
                "game": label.game, "dealer_no": label.dealer_no, "repeat": label.repeat, "hand_in_game": label.hand_in_game,
                "styles": styles, "initial": initial, "decisions": ds,
                "final": table_json(end), "result": result_json(r),
            }));
        },
    );
    hands
}

/// One dataset hand, replayed exactly, with everything recorded at each decision.
pub fn trace_hand(cfg: &SelfPlayConfig, hand_id: u64) -> Value {
    let seed = selfplay::hand_seed(cfg.seed, hand_id);
    let gcfg = selfplay::hand_config(hand_id);
    let (ids, mut bots) = selfplay::bots_for(cfg, hand_id);
    let start = Game::new(gcfg.clone(), seed);
    let initial = table_json(&start);
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut decisions = Vec::new();
    let (end, result) = selfplay::play_hand(start, &mut bots, |g, seat, a| {
        let (phase, tile, from) = match &g.phase {
            Phase::SelfTurn { drawn, .. } => ("own_turn", *drawn, None),
            Phase::Claim { from, tile } => ("claim", Some(*tile), Some(*from)),
            Phase::RobKong { seat, tile } => ("rob_kong", Some(*tile), Some(*seat)),
            Phase::Over => ("over", None, None),
        };
        let mut o = vec![0u8; OBS_LEN];
        obs::encode_obs(g, seat, &mut o);
        let mut waits = vec![0u8; WAITS_LEN];
        let mut sh = vec![0i8; SHANTEN_LEN];
        obs::encode_labels(g, seat, &mut waits, &mut sh);
        let mask = obs::legal_mask(g, seat);
        let legal: Vec<usize> = (0..N_ACTIONS).filter(|&i| mask[i] == 1).collect();
        decisions.push(json!({
            "seat": seat,
            "phase": phase,
            "tile": tile,
            "from": from,
            "legal": legal,
            "action": obs::action_index(a),
            "draws_left": g.draws_left(),
            "turn": g.turns,
            "table": table_json(g),
            "waits": waits.chunks(34).map(|c| c.to_vec()).collect::<Vec<_>>(),
            "shanten": sh,
            "obs": b64.encode(&o),
        }));
    });
    let score = result.score.as_ref().map(|s| {
        json!({
            "tai": s.tai,
            "raw_tai": s.raw_tai,
            "patterns": s.items.iter().map(|(p, t)| json!([p.name(), t])).collect::<Vec<_>>(),
        })
    });
    json!({
        "hand_id": hand_id,
        "seed": seed.to_string(),
        "dealer": gcfg.dealer,
        "prevailing": gcfg.prevailing_wind,
        "styles": ids.iter().map(|&i| cfg.styles[i].name.clone()).collect::<Vec<_>>(),
        "initial": initial,
        "decisions": decisions,
        "final": table_json(&end),
        "result": {
            "winner": result.winner,
            "discarder": result.discarder,
            "self_draw": result.self_draw,
            "deltas": result.deltas,
            "turns": result.turns,
            "score": score,
        },
    })
}
