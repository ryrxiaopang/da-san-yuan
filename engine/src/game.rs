//! Game state machine for one hand of Singapore 4-player mahjong.
//!
//! Drive it with `to_act()` -> `legal_actions(seat)` -> `apply(seat, action)` until
//! `is_over()`. Every decision point is explicit, which is what both the
//! heuristic bots and the RL environment need.

use crate::rng::Rng;
use crate::scoring::{self, Meld, MeldKind, Score, WinContext};
use crate::tile::*;

/// Every action a player can take. The numeric layout used for RL lives in `obs.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Discard(Tile),
    /// Concealed kong or adding a fourth tile to an exposed pong.
    Kong(Tile),
    /// Declare a win on a self-drawn tile.
    Tsumo,
    /// Declare a win on a discard (or by robbing a kong).
    Ron,
    Pong,
    /// Exposed kong from a discard.
    ExposedKong,
    /// Chow with the claimed tile at position 0 (lowest), 1 (middle) or 2 (highest).
    Chow(u8),
    Pass,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Live tiles left undrawn when the hand is declared a draw.
    pub reserve: usize,
    /// 0 = East round.
    pub prevailing_wind: u8,
    /// Seat that deals (sits East).
    pub dealer: u8,
}

impl Default for Config {
    fn default() -> Self {
        Config { reserve: 15, prevailing_wind: 0, dealer: 0 }
    }
}

/// One discarded tile as everyone at the table saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Discard {
    pub tile: Tile,
    /// Later claimed by another player (pong/chow/kong/win).
    pub claimed: bool,
    /// The tile thrown was the one just drawn (tsumogiri), visible at a real table.
    pub tsumogiri: bool,
    /// Global action counter when it was discarded, to recover the order across players.
    pub turn: u32,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub hand: Counts,
    pub bonus: Vec<Tile>,
    pub melds: Vec<Meld>,
    /// Discards in order.
    pub discards: Vec<Discard>,
    pub draws: u32,
    pub(crate) pending_bonus: usize,
}

impl Default for Player {
    fn default() -> Self {
        Player { hand: [0; NUM_KINDS], bonus: Vec::new(), melds: Vec::new(), discards: Vec::new(), draws: 0, pending_bonus: 0 }
    }
}

impl Player {
    pub fn hand_size(&self) -> usize {
        self.hand.iter().map(|&x| x as usize).sum()
    }
    pub fn melds_needed(&self) -> usize {
        4 - self.melds.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    /// `seat` holds 3n+2 tiles and must discard, kong or declare tsumo.
    SelfTurn { seat: u8, drawn: Option<Tile>, after_kong: bool, after_claim: bool },
    /// Others may claim `tile` discarded by `from`.
    Claim { from: u8, tile: Tile },
    /// Others may rob the kong `seat` is adding with `tile`.
    RobKong { seat: u8, tile: Tile },
    Over,
}

#[derive(Clone, Debug)]
pub struct HandResult {
    pub winner: Option<u8>,
    pub discarder: Option<u8>,
    pub self_draw: bool,
    pub score: Option<Score>,
    pub deltas: [i32; 4],
    pub turns: u32,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub cfg: Config,
    pub wall: Vec<Tile>,
    /// Next live draw position.
    pub front: usize,
    /// One past the next replacement tile (taken from the back).
    pub back: usize,
    pub players: [Player; 4],
    pub phase: Phase,
    /// Responses collected during a claim / rob-kong window.
    responses: [Option<Action>; 4],
    pub result: Option<HandResult>,
    pub any_claim: bool,
    pub turns: u32,
    /// Set once the last live tile has been drawn.
    pub last_draw: bool,
}

impl Game {
    /// Deal a new hand from a seeded shuffle.
    pub fn new(cfg: Config, seed: u64) -> Self {
        let mut wall: Vec<Tile> = Vec::with_capacity(WALL_SIZE);
        for k in 0..NUM_KINDS as Tile {
            for _ in 0..4 {
                wall.push(k);
            }
        }
        for b in FIRST_BONUS..NUM_TILE_IDS as Tile {
            wall.push(b);
        }
        let mut rng = Rng::new(seed);
        rng.shuffle(&mut wall);
        Self::from_wall(cfg, wall)
    }

    /// Deal from an explicit wall order (used by duplicate tournaments and tests).
    pub fn from_wall(cfg: Config, wall: Vec<Tile>) -> Self {
        assert_eq!(wall.len(), WALL_SIZE);
        let back = wall.len();
        let mut g = Game {
            cfg,
            wall,
            front: 0,
            back,
            players: Default::default(),
            phase: Phase::Over,
            responses: [None; 4],
            result: None,
            any_claim: false,
            turns: 0,
            last_draw: false,
        };
        g.deal();
        g
    }

    /// Build a mid-hand position directly (used by scenarios).
    /// `wall` holds every tile; the first `front` are treated as already drawn.
    /// For a Claim or RobKong phase only `decider` is asked; the other seats pass.
    pub fn from_parts(
        cfg: Config,
        wall: Vec<Tile>,
        front: usize,
        players: [Player; 4],
        phase: Phase,
        decider: u8,
        turns: u32,
    ) -> Self {
        assert_eq!(wall.len(), WALL_SIZE);
        let back = wall.len();
        let mut responses = [None; 4];
        if matches!(phase, Phase::Claim { .. } | Phase::RobKong { .. }) {
            for s in 0..4u8 {
                if s != decider {
                    responses[s as usize] = Some(Action::Pass);
                }
            }
        }
        let any_claim = players.iter().any(|p| p.melds.iter().any(|m| m.is_open()));
        let mut g = Game { cfg, wall, front, back, players, phase, responses, result: None, any_claim, turns, last_draw: false };
        g.last_draw = g.live_remaining() <= g.cfg.reserve;
        g
    }

    pub fn dealer(&self) -> u8 {
        self.cfg.dealer
    }
    /// Seat wind 0..=3 (East..North) for a seat.
    pub fn seat_wind(&self, seat: u8) -> u8 {
        (seat + 4 - self.cfg.dealer) % 4
    }
    pub fn live_remaining(&self) -> usize {
        self.back - self.front
    }
    pub fn draws_left(&self) -> usize {
        self.live_remaining().saturating_sub(self.cfg.reserve)
    }
    pub fn is_over(&self) -> bool {
        self.phase == Phase::Over
    }

    fn deal(&mut self) {
        let d = self.cfg.dealer;
        for i in 0..4u8 {
            let seat = (d + i) % 4;
            let n = if i == 0 { 14 } else { 13 };
            for _ in 0..n {
                let t = self.wall[self.front];
                self.front += 1;
                self.give(seat, t);
            }
        }
        // Replace bonus tiles in seat order starting from the dealer.
        for i in 0..4u8 {
            let seat = (d + i) % 4;
            while self.players[seat as usize].bonus_pending() > 0 {
                self.players[seat as usize].take_pending_bonus();
                let t = self.draw_back();
                self.give(seat, t);
            }
        }
        self.players[d as usize].draws = 1;
        self.phase = Phase::SelfTurn { seat: d, drawn: None, after_kong: false, after_claim: false };
    }

    fn give(&mut self, seat: u8, t: Tile) {
        let p = &mut self.players[seat as usize];
        if is_bonus(t) {
            p.bonus.push(t);
            p.pending_bonus += 1;
        } else {
            p.hand[t as usize] += 1;
        }
    }

    fn draw_back(&mut self) -> Tile {
        self.back -= 1;
        self.wall[self.back]
    }

    /// Draw for `seat`: front tile, replacing any bonus tiles from the back.
    /// Returns the playing tile finally drawn, or None if the wall ran out.
    fn draw_for(&mut self, seat: u8, from_back: bool) -> Option<Tile> {
        if self.live_remaining() <= self.cfg.reserve {
            return None;
        }
        let mut t = if from_back {
            self.draw_back()
        } else {
            let t = self.wall[self.front];
            self.front += 1;
            t
        };
        loop {
            if self.live_remaining() <= self.cfg.reserve {
                self.last_draw = true;
            }
            if !is_bonus(t) {
                break;
            }
            self.players[seat as usize].bonus.push(t);
            if self.live_remaining() <= self.cfg.reserve {
                return None;
            }
            t = self.draw_back();
        }
        self.players[seat as usize].hand[t as usize] += 1;
        self.players[seat as usize].draws += 1;
        Some(t)
    }

    /// Seat that must decide next, if any.
    pub fn to_act(&self) -> Option<u8> {
        match &self.phase {
            Phase::SelfTurn { seat, .. } => Some(*seat),
            Phase::Claim { from, .. } | Phase::RobKong { seat: from, .. } => {
                for i in 1..4u8 {
                    let s = (from + i) % 4;
                    if self.responses[s as usize].is_none() {
                        return Some(s);
                    }
                }
                None
            }
            Phase::Over => None,
        }
    }

    fn win_context(&self, seat: u8, win_tile: Tile, self_draw: bool) -> WinContext {
        let (after_kong, robbing) = match &self.phase {
            Phase::SelfTurn { after_kong, .. } => (*after_kong, false),
            Phase::RobKong { .. } => (false, true),
            _ => (false, false),
        };
        let p = &self.players[seat as usize];
        let first_turn = p.draws == 1 && !self.any_claim && p.discards.is_empty();
        WinContext {
            seat_wind: self.seat_wind(seat),
            prevailing_wind: self.cfg.prevailing_wind,
            self_draw,
            win_tile,
            kong_replacement: self_draw && after_kong,
            robbing_kong: robbing,
            last_tile: self.last_draw,
            heavenly: self_draw && first_turn && seat == self.cfg.dealer,
            earthly: self_draw && first_turn && seat != self.cfg.dealer,
        }
    }

    /// Score `seat` winning now (tsumo in SelfTurn, ron in Claim/RobKong). None if not a legal win.
    pub fn win_score(&self, seat: u8) -> Option<Score> {
        let p = &self.players[seat as usize];
        let (hand, win_tile, self_draw) = match &self.phase {
            Phase::SelfTurn { seat: s, drawn, after_claim, .. } if *s == seat && !after_claim => {
                // Dealer's first turn has no drawn tile; any tile can serve as the "winning" one.
                let wt = drawn.unwrap_or_else(|| (0..NUM_KINDS).find(|&k| p.hand[k] > 0).unwrap() as Tile);
                (p.hand, wt, true)
            }
            Phase::Claim { from, tile } | Phase::RobKong { seat: from, tile } if *from != seat => {
                let mut h = p.hand;
                h[*tile as usize] += 1;
                (h, *tile, false)
            }
            _ => return None,
        };
        if !scoring::is_complete_shape(&hand, p.melds.len()) {
            return None;
        }
        let ctx = self.win_context(seat, win_tile, self_draw);
        let mut best = scoring::score_hand(&hand, &p.melds, &p.bonus, &ctx)?;
        // On the dealer's opening hand, try every tile as the winning tile.
        if self_draw && matches!(self.phase, Phase::SelfTurn { drawn: None, .. }) {
            for k in 0..NUM_KINDS {
                if hand[k] > 0 {
                    let mut c = ctx.clone();
                    c.win_tile = k as Tile;
                    if let Some(s) = scoring::score_hand(&hand, &p.melds, &p.bonus, &c) {
                        if s.tai > best.tai {
                            best = s;
                        }
                    }
                }
            }
        }
        if best.tai < scoring::MIN_TAI {
            return None;
        }
        Some(best)
    }

    /// Tai `seat` would score by winning on `tile` if it were discarded right now
    /// (0 if that tile does not complete a hand worth at least the minimum).
    /// This is exact ground truth that uses hidden information: use it for labels
    /// and evaluation, never as an input to a policy.
    pub fn ron_tai_if_discarded(&self, seat: u8, tile: Tile) -> u8 {
        let p = &self.players[seat as usize];
        if p.hand[tile as usize] >= 4 || p.hand_size() % 3 != 1 {
            return 0;
        }
        let mut h = p.hand;
        h[tile as usize] += 1;
        if !scoring::is_complete_shape(&h, p.melds.len()) {
            return 0;
        }
        let ctx = WinContext {
            seat_wind: self.seat_wind(seat),
            prevailing_wind: self.cfg.prevailing_wind,
            self_draw: false,
            win_tile: tile,
            last_tile: self.last_draw,
            ..Default::default()
        };
        match scoring::score_hand(&h, &p.melds, &p.bonus, &ctx) {
            Some(s) if s.tai >= scoring::MIN_TAI => s.tai,
            _ => 0,
        }
    }

    pub fn legal_actions(&self, seat: u8) -> Vec<Action> {
        let mut acts = Vec::new();
        let p = &self.players[seat as usize];
        match &self.phase {
            Phase::SelfTurn { seat: s, after_claim, .. } if *s == seat => {
                if self.win_score(seat).is_some() {
                    acts.push(Action::Tsumo);
                }
                if !after_claim && self.draws_left() > 0 {
                    for k in 0..NUM_KINDS {
                        if p.hand[k] == 4 {
                            acts.push(Action::Kong(k as Tile));
                        } else if p.hand[k] >= 1
                            && p.melds.iter().any(|m| m.kind == MeldKind::Pong && m.tile as usize == k)
                        {
                            acts.push(Action::Kong(k as Tile));
                        }
                    }
                }
                for k in 0..NUM_KINDS {
                    if p.hand[k] > 0 {
                        acts.push(Action::Discard(k as Tile));
                    }
                }
            }
            Phase::Claim { from, tile } if *from != seat && self.responses[seat as usize].is_none() => {
                let t = *tile;
                if self.win_score(seat).is_some() {
                    acts.push(Action::Ron);
                }
                let n = p.hand[t as usize];
                if n >= 2 {
                    acts.push(Action::Pong);
                }
                if n == 3 && self.draws_left() > 0 {
                    acts.push(Action::ExposedKong);
                }
                if seat == (from + 1) % 4 && is_suited(t) {
                    let r = rank(t);
                    let h = &p.hand;
                    let ti = t as usize;
                    if r <= 6 && h[ti + 1] > 0 && h[ti + 2] > 0 {
                        acts.push(Action::Chow(0));
                    }
                    if (1..=7).contains(&r) && h[ti - 1] > 0 && h[ti + 1] > 0 {
                        acts.push(Action::Chow(1));
                    }
                    if r >= 2 && h[ti - 1] > 0 && h[ti - 2] > 0 {
                        acts.push(Action::Chow(2));
                    }
                }
                acts.push(Action::Pass);
            }
            Phase::RobKong { seat: from, .. } if *from != seat && self.responses[seat as usize].is_none() => {
                if self.win_score(seat).is_some() {
                    acts.push(Action::Ron);
                }
                acts.push(Action::Pass);
            }
            _ => {}
        }
        acts
    }

    /// Apply an action. Panics on illegal actions in debug builds; callers should
    /// pick from `legal_actions`.
    pub fn apply(&mut self, seat: u8, a: Action) {
        debug_assert!(self.legal_actions(seat).contains(&a), "illegal {:?} by {} in {:?}", a, seat, self.phase);
        match self.phase.clone() {
            Phase::SelfTurn { .. } => self.apply_self(seat, a),
            Phase::Claim { from, tile } => {
                self.responses[seat as usize] = Some(a);
                self.skip_trivial_responders(from);
                if self.to_act().is_none() {
                    self.resolve_claims(from, tile);
                }
            }
            Phase::RobKong { seat: konger, tile } => {
                self.responses[seat as usize] = Some(a);
                self.skip_trivial_responders(konger);
                if self.to_act().is_none() {
                    self.resolve_rob(konger, tile);
                }
            }
            Phase::Over => panic!("game is over"),
        }
    }

    fn apply_self(&mut self, seat: u8, a: Action) {
        self.turns += 1;
        match a {
            Action::Tsumo => {
                let score = self.win_score(seat).expect("tsumo not legal");
                self.finish(Some(seat), None, Some(score));
            }
            Action::Discard(t) => {
                let tsumogiri = matches!(self.phase, Phase::SelfTurn { drawn: Some(d), .. } if d == t);
                let turn = self.turns;
                let p = &mut self.players[seat as usize];
                p.hand[t as usize] -= 1;
                p.discards.push(Discard { tile: t, claimed: false, tsumogiri, turn });
                self.open_window(Phase::Claim { from: seat, tile: t }, seat);
            }
            Action::Kong(t) => {
                let p = &mut self.players[seat as usize];
                if p.hand[t as usize] == 4 {
                    p.hand[t as usize] = 0;
                    p.melds.push(Meld { kind: MeldKind::ConcealedKong, tile: t, claimed: None, from: None });
                    self.replacement_draw(seat);
                } else {
                    p.hand[t as usize] -= 1;
                    self.open_window(Phase::RobKong { seat, tile: t }, seat);
                }
            }
            _ => panic!("action {:?} not valid on own turn", a),
        }
    }

    fn replacement_draw(&mut self, seat: u8) {
        match self.draw_for(seat, true) {
            Some(t) => {
                self.phase = Phase::SelfTurn { seat, drawn: Some(t), after_kong: true, after_claim: false }
            }
            None => self.finish(None, None, None),
        }
    }

    fn open_window(&mut self, phase: Phase, actor: u8) {
        self.responses = [None; 4];
        self.responses[actor as usize] = Some(Action::Pass);
        self.phase = phase;
        self.skip_trivial_responders(actor);
        if self.to_act().is_none() {
            match self.phase.clone() {
                Phase::Claim { from, tile } => self.resolve_claims(from, tile),
                Phase::RobKong { seat, tile } => self.resolve_rob(seat, tile),
                _ => unreachable!(),
            }
        }
    }

    /// Auto-pass seats whose only option is Pass so callers only see real decisions.
    fn skip_trivial_responders(&mut self, actor: u8) {
        for i in 1..4u8 {
            let s = (actor + i) % 4;
            if self.responses[s as usize].is_none() {
                let acts = self.legal_actions(s);
                if acts.len() == 1 && acts[0] == Action::Pass {
                    self.responses[s as usize] = Some(Action::Pass);
                }
            }
        }
    }

    fn resolve_claims(&mut self, from: u8, tile: Tile) {
        // Priority: win (nearest seat after the discarder), then pong/kong, then chow.
        let order: Vec<u8> = (1..4u8).map(|i| (from + i) % 4).collect();
        if let Some(&w) = order.iter().find(|&&s| self.responses[s as usize] == Some(Action::Ron)) {
            let score = self.win_score(w).expect("ron not legal");
            self.players[w as usize].hand[tile as usize] += 1;
            self.mark_claimed(from);
            self.finish(Some(w), Some(from), Some(score));
            return;
        }
        for &s in &order {
            match self.responses[s as usize] {
                Some(Action::Pong) | Some(Action::ExposedKong) => {
                    let kong = self.responses[s as usize] == Some(Action::ExposedKong);
                    let p = &mut self.players[s as usize];
                    p.hand[tile as usize] -= if kong { 3 } else { 2 };
                    p.melds.push(Meld {
                        kind: if kong { MeldKind::ExposedKong } else { MeldKind::Pong },
                        tile,
                        claimed: Some(tile),
                        from: Some(from),
                    });
                    self.mark_claimed(from);
                    self.any_claim = true;
                    if kong {
                        self.replacement_draw(s);
                    } else {
                        self.phase = Phase::SelfTurn { seat: s, drawn: None, after_kong: false, after_claim: true };
                    }
                    return;
                }
                _ => {}
            }
        }
        let next = (from + 1) % 4;
        if let Some(Action::Chow(pos)) = self.responses[next as usize] {
            let low = tile - pos;
            let p = &mut self.players[next as usize];
            for t in low..low + 3 {
                if t != tile {
                    p.hand[t as usize] -= 1;
                }
            }
            p.melds.push(Meld { kind: MeldKind::Chow, tile: low, claimed: Some(tile), from: Some(from) });
            self.mark_claimed(from);
            self.any_claim = true;
            self.phase = Phase::SelfTurn { seat: next, drawn: None, after_kong: false, after_claim: true };
            return;
        }
        // Nobody claimed: next player draws.
        match self.draw_for(next, false) {
            Some(t) => {
                self.phase = Phase::SelfTurn { seat: next, drawn: Some(t), after_kong: false, after_claim: false }
            }
            None => self.finish(None, None, None),
        }
    }

    fn resolve_rob(&mut self, konger: u8, tile: Tile) {
        let order: Vec<u8> = (1..4u8).map(|i| (konger + i) % 4).collect();
        if let Some(&w) = order.iter().find(|&&s| self.responses[s as usize] == Some(Action::Ron)) {
            let score = self.win_score(w).expect("rob not legal");
            self.players[w as usize].hand[tile as usize] += 1;
            // The robbed tile leaves the konger's pong as is.
            self.finish(Some(w), Some(konger), Some(score));
            return;
        }
        let p = &mut self.players[konger as usize];
        let m = p.melds.iter_mut().find(|m| m.kind == MeldKind::Pong && m.tile == tile).unwrap();
        m.kind = MeldKind::AddedKong;
        self.phase = Phase::SelfTurn { seat: konger, drawn: None, after_kong: false, after_claim: false };
        self.replacement_draw(konger);
    }

    fn mark_claimed(&mut self, from: u8) {
        if let Some(last) = self.players[from as usize].discards.last_mut() {
            last.claimed = true;
        }
    }

    fn finish(&mut self, winner: Option<u8>, discarder: Option<u8>, score: Option<Score>) {
        let deltas = match (&winner, &score) {
            (Some(w), Some(s)) => scoring::payments(s.tai, *w as usize, discarder.map(|d| d as usize)),
            _ => [0; 4],
        };
        self.result = Some(HandResult {
            winner,
            discarder,
            self_draw: winner.is_some() && discarder.is_none(),
            score,
            deltas,
            turns: self.turns,
        });
        self.phase = Phase::Over;
    }

    /// Copies of each kind `seat` cannot see (wall + other players' concealed hands).
    pub fn unseen_for(&self, seat: u8) -> Counts {
        let mut seen = self.players[seat as usize].hand;
        for p in &self.players {
            for m in &p.melds {
                for t in m.tiles() {
                    seen[t as usize] += 1;
                }
            }
            for d in &p.discards {
                if !d.claimed {
                    seen[d.tile as usize] += 1;
                }
            }
        }
        let mut u = [0u8; NUM_KINDS];
        for k in 0..NUM_KINDS {
            u[k] = 4u8.saturating_sub(seen[k]);
        }
        u
    }

    /// (tiles drawn from the wall, tiles accounted for in hands/melds/discards).
    /// The two must always match; tests use this as a conservation check.
    pub fn tile_census(&self) -> (usize, usize) {
        let drawn = self.front + (self.wall.len() - self.back);
        let mut held = 0;
        for p in &self.players {
            held += p.hand_size() + p.bonus.len();
            for m in &p.melds {
                held += m.tiles().len();
            }
            held += p.discards.iter().filter(|d| !d.claimed).count();
        }
        (drawn, held)
    }
}

// Pending-bonus bookkeeping lives on Player but only matters during the deal.
impl Player {
    fn bonus_pending(&self) -> usize {
        self.pending_bonus
    }
    fn take_pending_bonus(&mut self) {
        self.pending_bonus -= 1;
    }
}
