//! Fixed action space and observation encoding shared by bots, the data logger
//! and the Python RL environment. Everything here only uses information the
//! acting player can legally see, except `encode_oracle`, which is stored
//! separately for oracle guiding / opponent-modelling targets.

use crate::game::{Action, Game, Phase};
use crate::scoring::{Meld, MeldKind};
use crate::tile::*;

// ------------------------------------------------------------------ actions

pub const A_DISCARD: usize = 0; // + tile kind (34)
pub const A_KONG: usize = 34; // + tile kind (34)
pub const A_TSUMO: usize = 68;
pub const A_RON: usize = 69;
pub const A_PONG: usize = 70;
pub const A_EXPOSED_KONG: usize = 71;
pub const A_CHOW: usize = 72; // + position 0..=2
pub const A_PASS: usize = 75;
pub const N_ACTIONS: usize = 76;

pub fn action_index(a: Action) -> usize {
    match a {
        Action::Discard(t) => A_DISCARD + t as usize,
        Action::Kong(t) => A_KONG + t as usize,
        Action::Tsumo => A_TSUMO,
        Action::Ron => A_RON,
        Action::Pong => A_PONG,
        Action::ExposedKong => A_EXPOSED_KONG,
        Action::Chow(p) => A_CHOW + p as usize,
        Action::Pass => A_PASS,
    }
}

pub fn action_from_index(i: usize) -> Option<Action> {
    Some(match i {
        0..=33 => Action::Discard(i as Tile),
        34..=67 => Action::Kong((i - 34) as Tile),
        A_TSUMO => Action::Tsumo,
        A_RON => Action::Ron,
        A_PONG => Action::Pong,
        A_EXPOSED_KONG => Action::ExposedKong,
        72..=74 => Action::Chow((i - 72) as u8),
        A_PASS => Action::Pass,
        _ => return None,
    })
}

pub fn action_name(a: Action) -> String {
    match a {
        Action::Discard(t) => format!("discard {}", tile_name(t)),
        Action::Kong(t) => format!("kong {}", tile_name(t)),
        Action::Chow(p) => format!("chow(pos {})", p),
        other => format!("{:?}", other).to_lowercase(),
    }
}

pub fn legal_mask(g: &Game, seat: u8) -> [u8; N_ACTIONS] {
    let mut m = [0u8; N_ACTIONS];
    for a in g.legal_actions(seat) {
        m[action_index(a)] = 1;
    }
    m
}

// ------------------------------------------------------------- observation

pub const MAX_DISCARD_SEQ: usize = 32;

pub const OFF_HAND: usize = 0;
pub const OFF_MELD_COUNTS: usize = OFF_HAND + 34; // 4 x 34
pub const OFF_DISCARD_COUNTS: usize = OFF_MELD_COUNTS + 4 * 34; // 4 x 34
pub const OFF_DISCARD_SEQ: usize = OFF_DISCARD_COUNTS + 4 * 34; // 4 x 32
pub const OFF_BONUS: usize = OFF_DISCARD_SEQ + 4 * MAX_DISCARD_SEQ; // 4 x 12
pub const OFF_MELDS: usize = OFF_BONUS + 4 * NUM_BONUS; // 4 x 4 x 2
pub const OFF_META: usize = OFF_MELDS + 4 * 4 * 2;
pub const META_LEN: usize = 14;
pub const OBS_LEN: usize = OFF_META + META_LEN;

pub const ORACLE_LEN: usize = 3 * 34;

pub const EMPTY: u8 = 255;
/// Added to a tile id in the discard sequence when that discard was claimed.
pub const CLAIMED_FLAG: u8 = 64;

/// Relative seat: 0 = self, 1 = next (right, plays after you), 2 = opposite, 3 = previous (left).
#[inline]
pub fn rel(me: u8, other: u8) -> usize {
    ((other + 4 - me) % 4) as usize
}
#[inline]
pub fn abs_seat(me: u8, r: usize) -> u8 {
    (me + r as u8) % 4
}

fn meld_code(m: &Meld) -> u8 {
    match m.kind {
        MeldKind::Chow => 1,
        MeldKind::Pong => 2,
        MeldKind::ExposedKong => 3,
        MeldKind::AddedKong => 4,
        MeldKind::ConcealedKong => 5,
    }
}

/// Encode what `seat` can see into a flat u8 vector of length OBS_LEN.
/// Layout (all seats relative: 0 = self, 1 = next, 2 = opposite, 3 = previous):
///   hand counts [34] | meld tile counts [4][34] | discard counts [4][34]
///   | discard sequences [4][32] (tile id, +64 if claimed, 255 = empty)
///   | bonus tiles held [4][12] (0/1) | melds [4][4][kind, tile]
///   | meta [14]: seat wind, prevailing wind, dealer (rel), draws left,
///     turn (capped 255), phase (0 own turn, 1 claim, 2 rob kong),
///     target tile (drawn/claimable tile or 255), target seat (rel or 255),
///     after kong, after claim, last draw, concealed tile counts of seats 1..3
pub fn encode_obs(g: &Game, seat: u8, out: &mut [u8]) {
    debug_assert!(out.len() >= OBS_LEN);
    out[..OBS_LEN].fill(0);
    let me = &g.players[seat as usize];
    for k in 0..34 {
        out[OFF_HAND + k] = me.hand[k];
    }
    for (abs, p) in g.players.iter().enumerate() {
        let r = rel(seat, abs as u8);
        for m in &p.melds {
            // Concealed kongs are shown face-down at many tables but the kind is public.
            for t in m.tiles() {
                out[OFF_MELD_COUNTS + r * 34 + t as usize] += 1;
            }
        }
        for (i, &(t, claimed)) in p.discards.iter().enumerate() {
            out[OFF_DISCARD_COUNTS + r * 34 + t as usize] += 1;
            if i < MAX_DISCARD_SEQ {
                out[OFF_DISCARD_SEQ + r * MAX_DISCARD_SEQ + i] = t + if claimed { CLAIMED_FLAG } else { 0 };
            }
        }
        for i in p.discards.len().min(MAX_DISCARD_SEQ)..MAX_DISCARD_SEQ {
            out[OFF_DISCARD_SEQ + r * MAX_DISCARD_SEQ + i] = EMPTY;
        }
        for &b in &p.bonus {
            out[OFF_BONUS + r * NUM_BONUS + (b - FIRST_BONUS) as usize] = 1;
        }
        for (i, m) in p.melds.iter().take(4).enumerate() {
            out[OFF_MELDS + (r * 4 + i) * 2] = meld_code(m);
            out[OFF_MELDS + (r * 4 + i) * 2 + 1] = m.tile;
        }
    }
    let mo = OFF_META;
    out[mo] = g.seat_wind(seat);
    out[mo + 1] = g.cfg.prevailing_wind;
    out[mo + 2] = rel(seat, g.cfg.dealer) as u8;
    out[mo + 3] = g.draws_left().min(255) as u8;
    out[mo + 4] = g.turns.min(255) as u8;
    let (phase, tile, from, ak, ac) = match &g.phase {
        Phase::SelfTurn { drawn, after_kong, after_claim, .. } => {
            (0, drawn.unwrap_or(EMPTY), EMPTY, *after_kong, *after_claim)
        }
        Phase::Claim { from, tile } => (1, *tile, rel(seat, *from) as u8, false, false),
        Phase::RobKong { seat: s, tile } => (2, *tile, rel(seat, *s) as u8, false, false),
        Phase::Over => (3, EMPTY, EMPTY, false, false),
    };
    out[mo + 5] = phase;
    out[mo + 6] = tile;
    out[mo + 7] = from;
    out[mo + 8] = ak as u8;
    out[mo + 9] = ac as u8;
    out[mo + 10] = g.last_draw as u8;
    for r in 1..4 {
        out[mo + 10 + r] = g.players[abs_seat(seat, r) as usize].hand_size() as u8;
    }
}

/// Hidden information: concealed hands of seats 1..3 relative to `seat`.
/// Never feed this to a policy at play time; it is a training target.
pub fn encode_oracle(g: &Game, seat: u8, out: &mut [u8]) {
    for r in 1..4 {
        let p = &g.players[abs_seat(seat, r) as usize];
        for k in 0..34 {
            out[(r - 1) * 34 + k] = p.hand[k];
        }
    }
}

// --------------------------------------------------------------------- view

/// The public view one seat has of the table. Bots only receive this, so they
/// cannot peek at other hands or the wall.
pub struct View<'a> {
    g: &'a Game,
    pub seat: u8,
}

impl<'a> View<'a> {
    pub fn new(g: &'a Game, seat: u8) -> Self {
        View { g, seat }
    }
    pub fn hand(&self) -> &Counts {
        &self.g.players[self.seat as usize].hand
    }
    pub fn my_melds(&self) -> &[Meld] {
        &self.g.players[self.seat as usize].melds
    }
    pub fn melds_of(&self, seat: u8) -> &[Meld] {
        &self.g.players[seat as usize].melds
    }
    pub fn my_bonus(&self) -> &[Tile] {
        &self.g.players[self.seat as usize].bonus
    }
    pub fn discards_of(&self, seat: u8) -> &[(Tile, bool)] {
        &self.g.players[seat as usize].discards
    }
    pub fn hand_size_of(&self, seat: u8) -> usize {
        self.g.players[seat as usize].hand_size()
    }
    pub fn unseen(&self) -> Counts {
        self.g.unseen_for(self.seat)
    }
    pub fn seat_wind(&self) -> u8 {
        self.g.seat_wind(self.seat)
    }
    pub fn seat_wind_of(&self, seat: u8) -> u8 {
        self.g.seat_wind(seat)
    }
    pub fn prevailing_wind(&self) -> u8 {
        self.g.cfg.prevailing_wind
    }
    pub fn draws_left(&self) -> usize {
        self.g.draws_left()
    }
    pub fn phase(&self) -> &Phase {
        &self.g.phase
    }
    pub fn legal_actions(&self) -> Vec<Action> {
        self.g.legal_actions(self.seat)
    }
    /// Tai this seat would score by winning right now, if winning is legal.
    pub fn win_tai(&self) -> Option<u8> {
        self.g.win_score(self.seat).map(|s| s.tai)
    }
}
