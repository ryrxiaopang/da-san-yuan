//! Hand-built test positions ("scenarios") with checkable expectations.
//!
//! Used two ways:
//!  * as rule tests (the engine must offer / refuse a win), and
//!  * as a skill benchmark for any policy, heuristic or learned: does it fold
//!    against an obvious threat, read a flush, keep efficient shapes, protect tai?
//!
//! File format (`scenarios/*.txt`): blocks separated by a line `---`.
//! Each block is `key: value` lines; `#` starts a comment. Seats are absolute
//! (0..3) and the dealer is East.
//!
//! ```text
//! name: fold-vs-bamboo-flush
//! desc: Right player has two bamboo melds and has thrown no bamboo.
//! tags: defence, flush-read
//! hero: 0              # seat being tested (default 0)
//! dealer: 0            # default 0
//! prevailing: 0        # default 0
//! draws_left: 20       # live draws remaining (default 30)
//! seed: 1              # for filling unspecified hidden hands
//! phase: turn          # turn | claim 5z from 3 | rob 4p from 2
//! drawn: 9m            # optional, the tile hero just drew (turn phase)
//! hand.0: 123m 456p ...   # concealed tiles (no bonus tiles here)
//! bonus.0: f1 a2
//! melds.1: pong:2s from 0, chow:678s
//! discards.1: 1m 9p 4p 3z
//! hand.1: 34s 11z      # optional hidden hand; random filler if omitted
//! assert_ready: 1      # validation: these seats must be ready (one tile from winning)
//! expect: safe         # see Expectation below; several expect lines allowed
//! ```

use crate::game::{Action, Config, Discard, Game, Phase, Player};
use crate::obs::{action_name, View};
use crate::rng::Rng;
use crate::scoring::{Meld, MeldKind};
use crate::shanten::{shanten, ukeire};
use crate::tile::*;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Expectation {
    /// The chosen discard must not be a winning tile for any opponent (exact, uses hidden hands).
    Safe,
    /// Must declare a win (tsumo or ron).
    Win,
    /// Must pass (claim / rob phases).
    Pass,
    /// The engine must not offer a win (rule check, independent of the policy).
    NoWinOffered,
    /// The engine must offer a win (rule check).
    WinOffered,
    /// Chosen discard keeps the lowest shanten and the most useful tiles (public information only).
    MaxUkeire,
    /// Chosen action must be one of these.
    AnyOf(Vec<Action>),
    /// Chosen action must not be any of these.
    NoneOf(Vec<Action>),
}

#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: String,
    pub desc: String,
    pub tags: Vec<String>,
    pub hero: u8,
    pub expectations: Vec<Expectation>,
    pub assert_ready: Vec<u8>,
    fields: HashMap<String, String>,
}

fn parse_action(s: &str) -> Result<Action, String> {
    let s = s.trim();
    let (verb, rest) = s.split_once(' ').unwrap_or((s, ""));
    let tile = || parse_tiles(rest.trim()).ok().and_then(|v| v.first().copied()).ok_or(format!("bad tile in '{}'", s));
    Ok(match verb {
        "discard" => Action::Discard(tile()?),
        "kong" => Action::Kong(tile()?),
        "tsumo" => Action::Tsumo,
        "ron" => Action::Ron,
        "pong" => Action::Pong,
        "exposed_kong" => Action::ExposedKong,
        "chow" => Action::Chow(rest.trim().parse().map_err(|_| format!("chow needs position 0-2: '{}'", s))?),
        "pass" => Action::Pass,
        _ => return Err(format!("unknown action '{}'", s)),
    })
}

fn parse_meld(s: &str, owner: u8) -> Result<Meld, String> {
    let s = s.trim();
    let (body, from) = match s.split_once(" from ") {
        Some((b, f)) => (b.trim(), Some(f.trim().parse::<u8>().map_err(|_| format!("bad seat in '{}'", s))?)),
        None => (s, None),
    };
    let (kind, tiles) = body.split_once(':').ok_or(format!("meld must be kind:tiles, got '{}'", s))?;
    let t = parse_tiles(tiles)?;
    let (kind, tile) = match kind {
        "chow" => {
            let mut v = t.clone();
            v.sort();
            if v.len() != 3 || !is_suited(v[0]) || v[1] != v[0] + 1 || v[2] != v[0] + 2 || suit(v[0]) != suit(v[2]) {
                return Err(format!("bad chow '{}'", s));
            }
            (MeldKind::Chow, v[0])
        }
        "pong" => (MeldKind::Pong, t[0]),
        "kong" => (MeldKind::ExposedKong, t[0]),
        "ckong" => (MeldKind::ConcealedKong, t[0]),
        _ => return Err(format!("unknown meld kind in '{}'", s)),
    };
    let from = if kind == MeldKind::ConcealedKong { None } else { Some(from.unwrap_or((owner + 3) % 4)) };
    Ok(Meld { kind, tile, claimed: Some(tile), from })
}

impl Scenario {
    fn get(&self, k: &str) -> Option<&str> {
        self.fields.get(k).map(|s| s.as_str())
    }
    fn num(&self, k: &str, default: u64) -> Result<u64, String> {
        match self.get(k) {
            Some(v) => v.parse().map_err(|_| format!("{}: '{}' is not a number", k, v)),
            None => Ok(default),
        }
    }

    /// Build the position. Hidden hands not given are filled randomly from unused tiles.
    pub fn build(&self) -> Result<Game, String> {
        let dealer = self.num("dealer", 0)? as u8;
        let prevailing = self.num("prevailing", 0)? as u8;
        let draws_left = self.num("draws_left", 30)? as usize;
        let seed = self.num("seed", 1)?;
        let mut players: [Player; 4] = Default::default();
        let mut used = [0u8; NUM_TILE_IDS];
        let add_used = |t: Tile, used: &mut [u8; NUM_TILE_IDS]| -> Result<(), String> {
            used[t as usize] += 1;
            let cap = if is_bonus(t) { 1 } else { 4 };
            if used[t as usize] > cap {
                return Err(format!("too many copies of {}", tile_name(t)));
            }
            Ok(())
        };
        let mut turns = 0u32;
        // Discarded claim tile (claim/rob phase) is placed at the end of the source's discards.
        let phase_spec = self.get("phase").unwrap_or("turn").to_string();
        let words: Vec<&str> = phase_spec.split_whitespace().collect();
        for s in 0..4u8 {
            let p = &mut players[s as usize];
            if let Some(m) = self.get(&format!("melds.{}", s)) {
                for part in m.split(',').filter(|x| !x.trim().is_empty()) {
                    let meld = parse_meld(part, s)?;
                    for t in meld.tiles() {
                        add_used(t, &mut used)?;
                    }
                    p.melds.push(meld);
                }
            }
            if let Some(b) = self.get(&format!("bonus.{}", s)) {
                for t in parse_tiles(b)? {
                    if !is_bonus(t) {
                        return Err(format!("bonus.{} has non-bonus tile {}", s, tile_name(t)));
                    }
                    add_used(t, &mut used)?;
                    p.bonus.push(t);
                }
            }
            if let Some(d) = self.get(&format!("discards.{}", s)) {
                for (i, t) in parse_tiles(d)?.into_iter().enumerate() {
                    add_used(t, &mut used)?;
                    p.discards.push(Discard { tile: t, claimed: false, tsumogiri: false, turn: (i * 4 + s as usize) as u32 });
                    turns += 1;
                }
            }
            if let Some(h) = self.get(&format!("hand.{}", s)) {
                for t in parse_tiles(h)? {
                    if is_bonus(t) {
                        return Err(format!("put bonus tiles in bonus.{}, not hand.{}", s, s));
                    }
                    add_used(t, &mut used)?;
                    p.hand[t as usize] += 1;
                }
            }
            p.draws = p.discards.len() as u32 + 1;
        }
        let (phase, decider) = match words.as_slice() {
            ["turn"] => {
                let drawn = match self.get("drawn") {
                    Some(d) => Some(parse_tiles(d)?[0]),
                    None => None,
                };
                (Phase::SelfTurn { seat: self.hero, drawn, after_kong: false, after_claim: false }, self.hero)
            }
            ["claim", t, "from", f] | ["rob", t, "from", f] => {
                let tile = parse_tiles(t)?[0];
                let from: u8 = f.parse().map_err(|_| "bad seat in phase".to_string())?;
                if words[0] == "claim" {
                    add_used(tile, &mut used)?;
                    players[from as usize].discards.push(Discard { tile, claimed: false, tsumogiri: false, turn: turns as u32 });
                    turns += 1;
                    (Phase::Claim { from, tile }, self.hero)
                } else {
                    // The robbed tile is the 4th copy being added to `from`'s pong.
                    add_used(tile, &mut used)?;
                    if !players[from as usize].melds.iter().any(|m| m.kind == MeldKind::Pong && m.tile == tile) {
                        return Err(format!("rob phase needs seat {} to hold a pong of {}", from, tile_name(tile)));
                    }
                    (Phase::RobKong { seat: from, tile }, self.hero)
                }
            }
            _ => return Err(format!("bad phase '{}'", phase_spec)),
        };
        // Fill hidden hands that were not specified.
        let mut pool: Vec<Tile> = Vec::new();
        for k in 0..NUM_KINDS as Tile {
            for _ in used[k as usize]..4 {
                pool.push(k);
            }
        }
        for b in FIRST_BONUS..NUM_TILE_IDS as Tile {
            if used[b as usize] == 0 {
                pool.push(b);
            }
        }
        let mut rng = Rng::new(seed);
        rng.shuffle(&mut pool);
        // Bonus tiles stay in the wall; filler hands only take playing tiles.
        let mut playing: Vec<Tile> = pool.iter().copied().filter(|&t| !is_bonus(t)).collect();
        let mut bonus_left: Vec<Tile> = pool.iter().copied().filter(|&t| is_bonus(t)).collect();
        for s in 0..4u8 {
            let p = &mut players[s as usize];
            let want = 13 - 3 * p.melds.len()
                + if matches!(phase, Phase::SelfTurn { seat, .. } if seat == s) { 1 } else { 0 };
            let have = p.hand_size();
            if self.get(&format!("hand.{}", s)).is_none() {
                for _ in 0..want {
                    let t = playing.pop().ok_or("ran out of tiles while filling hands")?;
                    p.hand[t as usize] += 1;
                }
            } else if have != want {
                return Err(format!("hand.{} has {} tiles, expected {} (13 - 3 per meld, +1 on own turn)", s, have, want));
            }
        }
        // Wall: everything already in play first ("drawn"), then the live tiles.
        let mut wall: Vec<Tile> = Vec::with_capacity(WALL_SIZE);
        for p in &players {
            for k in 0..NUM_KINDS {
                for _ in 0..p.hand[k] {
                    wall.push(k as Tile);
                }
            }
            for m in &p.melds {
                wall.extend(m.tiles());
            }
            wall.extend(p.bonus.iter().copied());
            wall.extend(p.discards.iter().map(|d| d.tile));
        }
        if let Phase::RobKong { tile, .. } = phase {
            wall.push(tile); // the tile being added to the kong is in play but in nobody's hand
        }
        let front = wall.len();
        let mut live: Vec<Tile> = Vec::new();
        live.append(&mut playing);
        live.append(&mut bonus_left);
        rng.shuffle(&mut live);
        wall.extend(live.iter().copied());
        if wall.len() != WALL_SIZE {
            return Err(format!("tile accounting error: {} tiles", wall.len()));
        }
        if live.len() < draws_left {
            return Err(format!("only {} live tiles but draws_left = {}", live.len(), draws_left));
        }
        let cfg = Config { reserve: live.len() - draws_left, prevailing_wind: prevailing, dealer };
        Ok(Game::from_parts(cfg, wall, front, players, phase, decider, turns))
    }

    /// Structural checks: position builds, ready seats really are ready, and each
    /// expectation is satisfiable and non-trivial.
    pub fn validate(&self) -> Result<(), String> {
        let g = self.build()?;
        if g.to_act() != Some(self.hero) {
            return Err(format!("hero {} is not the seat to act", self.hero));
        }
        for &s in &self.assert_ready {
            let p = &g.players[s as usize];
            if shanten(&p.hand, p.melds_needed()) != 0 {
                return Err(format!("seat {} is not ready", s));
            }
            if (0..34u8).all(|k| g.ron_tai_if_discarded(s, k) == 0) {
                return Err(format!("seat {} is ready but cannot win any tile with 1+ tai", s));
            }
        }
        let legal = g.legal_actions(self.hero);
        if self.get("trap") == Some("yes") {
            // The purely efficient discard(s) must deal in, so passing needs real reading.
            let rank = efficiency_ranking(&View::new(&g, self.hero));
            let top = rank[0].1;
            for &(t, sc) in rank.iter().filter(|r| r.1 == top) {
                if deal_in_tai(&g, self.hero, t) == 0 {
                    return Err(format!("trap: most efficient discard {} (score {}) is already safe", tile_name(t), sc));
                }
            }
        }
        for e in &self.expectations {
            match e {
                Expectation::Safe => {
                    let discards: Vec<Tile> =
                        legal.iter().filter_map(|a| if let Action::Discard(t) = a { Some(*t) } else { None }).collect();
                    let safe = discards.iter().filter(|&&t| deal_in_tai(&g, self.hero, t) == 0).count();
                    if safe == 0 {
                        return Err("expect safe, but no discard is safe".into());
                    }
                    if safe == discards.len() {
                        return Err("expect safe, but every discard is safe (trivial)".into());
                    }
                }
                Expectation::AnyOf(v) | Expectation::NoneOf(v) => {
                    for a in v {
                        if !legal.contains(a) {
                            return Err(format!("{} is not legal here", action_name(*a)));
                        }
                    }
                }
                Expectation::Win => {
                    if !legal.contains(&Action::Tsumo) && !legal.contains(&Action::Ron) {
                        return Err("expect win, but no win is legal".into());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Check a policy's chosen action. Ok(()) = passed.
    pub fn check(&self, g: &Game, a: Action) -> Result<(), String> {
        let legal = g.legal_actions(self.hero);
        if !legal.contains(&a) {
            return Err(format!("illegal action {}", action_name(a)));
        }
        for e in &self.expectations {
            let ok = match e {
                Expectation::Safe => match a {
                    Action::Discard(t) => {
                        let tai = deal_in_tai(g, self.hero, t);
                        if tai > 0 {
                            return Err(format!("{} deals in for {} tai", action_name(a), tai));
                        }
                        true
                    }
                    _ => true,
                },
                Expectation::Win => matches!(a, Action::Tsumo | Action::Ron),
                Expectation::Pass => a == Action::Pass,
                Expectation::NoWinOffered => !legal.contains(&Action::Tsumo) && !legal.contains(&Action::Ron),
                Expectation::WinOffered => legal.contains(&Action::Tsumo) || legal.contains(&Action::Ron),
                Expectation::MaxUkeire => match a {
                    Action::Discard(t) => {
                        let v = View::new(g, self.hero);
                        let best = efficiency_ranking(&v);
                        let top = best[0].1;
                        best.iter().any(|&(bt, score)| bt == t && score == top)
                    }
                    _ => false,
                },
                Expectation::AnyOf(v) => v.contains(&a),
                Expectation::NoneOf(v) => !v.contains(&a),
            };
            if !ok {
                return Err(format!("{} fails {:?}", action_name(a), e));
            }
        }
        Ok(())
    }
}

/// Exact: the most tai any opponent would win if `hero` discarded `t` now (0 = safe).
/// Respects claim priority only loosely: any opponent able to win counts.
pub fn deal_in_tai(g: &Game, hero: u8, t: Tile) -> u8 {
    (1..4u8).map(|i| g.ron_tai_if_discarded((hero + i) % 4, t)).max().unwrap_or(0)
}

/// Discards ranked by (lower shanten, more useful tiles), public information only.
/// Returns (tile, combined score) best first; equal scores are equally good.
pub fn efficiency_ranking(v: &View) -> Vec<(Tile, i64)> {
    let hand = *v.hand();
    let need = 4 - v.my_melds().len();
    let unseen = v.unseen();
    let mut out = Vec::new();
    let mut h = hand;
    for k in 0..34 {
        if h[k] == 0 {
            continue;
        }
        h[k] -= 1;
        let sh = shanten(&h, need) as i64;
        let (uk, _) = ukeire(&h, need, &unseen);
        out.push((k as Tile, -sh * 1000 + uk as i64));
        h[k] += 1;
    }
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

pub fn parse_scenarios(text: &str) -> Result<Vec<Scenario>, String> {
    let mut out = Vec::new();
    for (bi, block) in text.split("\n---").enumerate() {
        let mut fields = HashMap::new();
        let mut expectations = Vec::new();
        let mut assert_ready = Vec::new();
        for line in block.lines() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() || line == "---" {
                continue;
            }
            let (k, v) = line.split_once(':').ok_or(format!("block {}: expected 'key: value', got '{}'", bi, line))?;
            let (k, v) = (k.trim(), v.trim());
            match k {
                "expect" => {
                    let (kind, rest) = v.split_once(' ').unwrap_or((v, ""));
                    let list = || -> Result<Vec<Action>, String> { rest.split(',').map(parse_action).collect() };
                    expectations.push(match kind {
                        "safe" => Expectation::Safe,
                        "win" => Expectation::Win,
                        "pass" => Expectation::Pass,
                        "no_win_offered" => Expectation::NoWinOffered,
                        "win_offered" => Expectation::WinOffered,
                        "max_ukeire" => Expectation::MaxUkeire,
                        "any_of" => Expectation::AnyOf(list()?),
                        "none_of" => Expectation::NoneOf(list()?),
                        _ => return Err(format!("unknown expectation '{}'", v)),
                    });
                }
                "assert_ready" => {
                    for s in v.split(',') {
                        assert_ready.push(s.trim().parse().map_err(|_| format!("bad seat '{}'", s))?);
                    }
                }
                _ => {
                    fields.insert(k.to_string(), v.to_string());
                }
            }
        }
        if fields.is_empty() && expectations.is_empty() {
            continue;
        }
        let name = fields.get("name").cloned().ok_or(format!("block {} has no name", bi))?;
        let hero = fields.get("hero").map(|h| h.parse().unwrap_or(0)).unwrap_or(0);
        let tags = fields.get("tags").map(|t| t.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
        let desc = fields.get("desc").cloned().unwrap_or_default();
        out.push(Scenario { name, desc, tags, hero, expectations, assert_ready, fields });
    }
    Ok(out)
}

pub fn load_dir(dir: &std::path::Path) -> Result<Vec<(String, Scenario)>, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {}", dir.display(), e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x == "txt"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).map_err(|e| e.to_string())?;
        for s in parse_scenarios(&text).map_err(|e| format!("{}: {}", f.display(), e))? {
            out.push((f.file_name().unwrap().to_string_lossy().to_string(), s));
        }
    }
    Ok(out)
}
