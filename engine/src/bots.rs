//! Heuristic bots. They see only a `View` (public information + own hand).
//!
//! Each bot is the same algorithm driven by a `Style` of weights, so the four
//! archetypes and any random variations of them live in one parameter space.
//! That is what the evolution / league scripts mutate.

use crate::game::{Action, Phase};
use crate::obs::View;
use crate::rng::Rng;
use crate::scoring::{bonus_tai, Meld, MeldKind};
use crate::shanten::{shanten, ukeire};
use crate::tile::*;

pub trait Bot: Send {
    fn act(&mut self, view: &View) -> Action;
    fn name(&self) -> String;
}

/// Weights for the heuristic. All non-negative; scale is roughly "tiles of ukeire".
#[derive(Clone, Debug, PartialEq)]
pub struct Style {
    pub name: String,
    /// Weight on number of useful tiles (speed to ready).
    pub speed: f32,
    /// Weight on keeping a hand that scores tai (honour pongs, flush, all pongs).
    pub value: f32,
    /// Weight on avoiding dangerous discards when opponents look close.
    pub defence: f32,
    /// Willingness to claim pong/chow (0 = almost never, 1 = whenever it speeds up).
    pub claim: f32,
    /// Preference for flush building.
    pub flush: f32,
    /// Preference for all-pongs shapes.
    pub pongs: f32,
}

impl Style {
    pub fn fast() -> Self {
        Style { name: "fast".into(), speed: 1.0, value: 0.6, defence: 0.2, claim: 0.9, flush: 0.2, pongs: 0.2 }
    }
    pub fn high_tai() -> Self {
        Style { name: "high_tai".into(), speed: 0.6, value: 2.0, defence: 0.4, claim: 0.4, flush: 1.5, pongs: 1.0 }
    }
    pub fn defensive() -> Self {
        Style { name: "defensive".into(), speed: 0.8, value: 0.8, defence: 2.5, claim: 0.3, flush: 0.4, pongs: 0.3 }
    }
    pub fn balanced() -> Self {
        Style { name: "balanced".into(), speed: 0.9, value: 1.0, defence: 1.0, claim: 0.6, flush: 0.6, pongs: 0.5 }
    }
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "fast" => Some(Self::fast()),
            "high_tai" | "value" => Some(Self::high_tai()),
            "defensive" => Some(Self::defensive()),
            "balanced" => Some(Self::balanced()),
            _ => None,
        }
    }
    pub fn archetypes() -> Vec<Style> {
        vec![Self::fast(), Self::high_tai(), Self::defensive(), Self::balanced()]
    }
    /// Random variation for evolutionary search: each weight scaled by exp(N(0, sigma)).
    pub fn mutate(&self, rng: &mut Rng, sigma: f32, name: &str) -> Style {
        let mut g = || {
            // Box-Muller
            let u1 = rng.unit().max(1e-12);
            let u2 = rng.unit();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32 * sigma
        };
        Style {
            name: name.to_string(),
            speed: self.speed * g().exp(),
            value: self.value * g().exp(),
            defence: self.defence * g().exp(),
            claim: (self.claim * g().exp()).min(1.5),
            flush: self.flush * g().exp(),
            pongs: self.pongs * g().exp(),
        }
    }
    pub fn to_csv(&self) -> String {
        format!("{},{},{},{},{},{},{}", self.name, self.speed, self.value, self.defence, self.claim, self.flush, self.pongs)
    }
    pub fn from_csv(s: &str) -> Option<Style> {
        let f: Vec<&str> = s.trim().split(',').collect();
        if f.len() != 7 {
            return None;
        }
        let p = |i: usize| f[i].parse::<f32>().ok();
        Some(Style {
            name: f[0].to_string(),
            speed: p(1)?,
            value: p(2)?,
            defence: p(3)?,
            claim: p(4)?,
            flush: p(5)?,
            pongs: p(6)?,
        })
    }
}

pub struct HeuristicBot {
    pub style: Style,
    rng: Rng,
    /// Small random noise added to scores so identical styles don't play identically.
    pub noise: f32,
}

impl HeuristicBot {
    pub fn new(style: Style, seed: u64) -> Self {
        HeuristicBot { style, rng: Rng::new(seed), noise: 0.05 }
    }
}

/// Is `t` a tile whose pong scores tai for this seat?
fn valued_honor(t: Tile, seat_wind: u8, prevailing: u8) -> bool {
    is_dragon(t) || t == wind_tile(seat_wind) || t == wind_tile(prevailing)
}

struct Ctx<'a> {
    melds: &'a [Meld],
    bonus_tai: u8,
    seat_wind: u8,
    prevailing: u8,
    unseen: Counts,
    threat: f32,
    /// discards seen from each opponent, for safety
    opp_discards: [u64; 4],
    threat_by_seat: [f32; 4],
    /// Suit an opponent appears to be collecting for a flush (from their melds).
    suit_focus: [Option<u8>; 4],
    me: u8,
}

impl Ctx<'_> {
    fn open(&self) -> bool {
        self.melds.iter().any(|m| m.is_open())
    }
}

/// How likely the hand can reach the 1-tai minimum, plus how much tai it is building.
/// Returns (secured, potential).
fn tai_outlook(hand: &Counts, melds: &[Meld], ctx: &Ctx, style: &Style) -> (bool, f32) {
    let open = melds.iter().any(|m| m.is_open());
    let mut secured = ctx.bonus_tai >= 1 || !open;
    let mut potential = 0.0f32;
    for m in melds {
        if m.is_triplet_like() && valued_honor(m.tile, ctx.seat_wind, ctx.prevailing) {
            secured = true;
            potential += 1.0;
        }
    }
    for t in 27..34u8 {
        if !valued_honor(t, ctx.seat_wind, ctx.prevailing) {
            continue;
        }
        let n = hand[t as usize];
        if n >= 3 {
            secured = true;
            potential += 1.0;
        } else if n == 2 && ctx.unseen[t as usize] > 0 {
            potential += 0.45;
        }
    }
    // Flush: share of playing tiles in the dominant suit (+ honours for half flush)
    let mut by_suit = [0u32; 4];
    for k in 0..34 {
        by_suit[suit(k as Tile) as usize] += hand[k] as u32;
    }
    for m in melds {
        by_suit[suit(m.tile) as usize] += 3;
    }
    let total: u32 = by_suit.iter().sum();
    if total > 0 {
        let main = by_suit[..3].iter().copied().max().unwrap();
        let others = total - main - by_suit[3];
        let frac = (main + by_suit[3]) as f32 / total as f32;
        if others == 0 {
            secured = true; // a half flush is already guaranteed
            potential += 2.0 + if by_suit[3] == 0 { 2.0 } else { 0.0 };
        } else if frac >= 0.7 {
            potential += style.flush * (frac - 0.6) * 5.0 / (others as f32);
        }
    }
    // All pongs shape
    let pairs_trips = (0..34).filter(|&k| hand[k] >= 2).count() as f32
        + melds.iter().filter(|m| m.is_triplet_like()).count() as f32;
    let chow_melds = melds.iter().filter(|m| m.kind == MeldKind::Chow).count();
    if chow_melds == 0 && pairs_trips >= 4.0 {
        potential += style.pongs * (pairs_trips - 3.0) * 0.6;
    }
    (secured, potential)
}

/// How risky is discarding `t` right now (0 = safe, ~1 = dangerous middle tile).
fn danger(t: Tile, ctx: &Ctx) -> f32 {
    if ctx.threat <= 0.0 {
        return 0.0;
    }
    let base = if is_honor(t) {
        match ctx.unseen[t as usize] {
            0 => 0.0,
            1 => 0.15,
            _ => 0.45,
        }
    } else {
        let r = match rank(t) {
            0 | 8 => 0.5,
            1 | 7 => 0.75,
            _ => 1.0,
        };
        // No unseen copies: nobody can be waiting on it as a pair or pong, only in a sequence.
        if ctx.unseen[t as usize] == 0 {
            r * 0.6
        } else {
            r
        }
    };
    let mut d = 0.0;
    for s in 0..4u8 {
        if s == ctx.me {
            continue;
        }
        let w = ctx.threat_by_seat[s as usize];
        if w <= 0.0 {
            continue;
        }
        // Tiles an opponent threw recently are less likely to be what they wait on.
        let safe = ctx.opp_discards[s as usize] & (1u64 << t) != 0;
        // Flush read: an opponent whose melds are all one suit wants that suit (and honours).
        let focus = match ctx.suit_focus[s as usize] {
            Some(f) if is_suited(t) => {
                if suit(t) == f {
                    2.0
                } else {
                    0.3
                }
            }
            _ => 1.0,
        };
        d += w * base * focus * if safe { 0.25 } else { 1.0 };
    }
    d
}

fn eval_hand(hand: &Counts, need: usize, ctx: &Ctx, style: &Style, melds: &[Meld]) -> (i32, f32) {
    let sh = shanten(hand, need);
    let (uk, _) = ukeire(hand, need, &ctx.unseen);
    let (secured, potential) = tai_outlook(hand, melds, ctx, style);
    let mut score = -(sh as f32) * 40.0 + style.speed * uk as f32 + style.value * potential * 3.0;
    if !secured {
        // No route to the 1-tai minimum yet: heavily discourage unless potential exists.
        score -= (2.0 - potential).max(0.0) * 6.0;
    }
    (sh, score)
}

impl HeuristicBot {
    fn make_ctx<'a>(&self, v: &'a View) -> Ctx<'a> {
        let me = v.seat;
        let (bt, _) = bonus_tai(v.my_bonus(), v.seat_wind());
        let mut threat_by_seat = [0f32; 4];
        let mut opp_discards = [0u64; 4];
        let mut suit_focus = [None; 4];
        let late = 1.0 - (v.draws_left() as f32 / 80.0).min(1.0);
        for s in 0..4u8 {
            if s == me {
                continue;
            }
            let open = v.melds_of(s).len() as f32;
            let small_hand = v.hand_size_of(s) <= 7;
            threat_by_seat[s as usize] = (open * 0.3 + late * 0.6 + if small_hand { 0.4 } else { 0.0 } - 0.35).max(0.0);
            let d = v.discards_of(s);
            for x in d.iter().rev().take(8) {
                opp_discards[s as usize] |= 1u64 << x.tile;
            }
            let melds = v.melds_of(s);
            let suits: Vec<u8> = melds.iter().filter(|m| is_suited(m.tile)).map(|m| suit(m.tile)).collect();
            if melds.len() >= 2 && !suits.is_empty() && suits.iter().all(|&x| x == suits[0]) {
                // and they have not been throwing that suit away
                let thrown = d.iter().rev().take(10).filter(|x| is_suited(x.tile) && suit(x.tile) == suits[0]).count();
                if thrown <= 1 {
                    suit_focus[s as usize] = Some(suits[0]);
                    threat_by_seat[s as usize] += 0.3;
                }
            }
        }
        Ctx {
            melds: v.my_melds(),
            bonus_tai: bt,
            seat_wind: v.seat_wind(),
            prevailing: v.prevailing_wind(),
            unseen: v.unseen(),
            threat: threat_by_seat.iter().sum(),
            opp_discards,
            threat_by_seat,
            suit_focus,
            me,
        }
    }

    fn jitter(&mut self) -> f32 {
        (self.rng.unit() as f32 - 0.5) * self.noise
    }

    /// Best discard for a hand with 3n+2 tiles: (tile, score, shanten after discard).
    fn best_discard(&mut self, hand: &Counts, melds: &[Meld], ctx: &Ctx) -> (Tile, f32, i32) {
        let need = 4 - melds.len();
        let mut best = (0u8, f32::NEG_INFINITY, 99);
        let mut h = *hand;
        for k in 0..34 {
            if h[k] == 0 {
                continue;
            }
            h[k] -= 1;
            let (sh, mut s) = eval_hand(&h, need, ctx, &self.style, melds);
            s -= self.style.defence * danger(k as Tile, ctx) * 25.0;
            s += self.jitter();
            if s > best.1 {
                best = (k as Tile, s, sh);
            }
            h[k] += 1;
        }
        best
    }

    fn own_turn(&mut self, v: &View, acts: &[Action]) -> Action {
        if acts.contains(&Action::Tsumo) {
            return Action::Tsumo;
        }
        let ctx = self.make_ctx(v);
        let hand = *v.hand();
        let (tile, score, _) = self.best_discard(&hand, ctx.melds, &ctx);
        // Consider kongs: take one if the hand after it is at least as good.
        for &a in acts {
            if let Action::Kong(t) = a {
                let mut h = hand;
                let mut melds = ctx.melds.to_vec();
                if h[t as usize] == 4 {
                    h[t as usize] = 0;
                    melds.push(Meld { kind: MeldKind::ConcealedKong, tile: t, claimed: None, from: None });
                } else {
                    h[t as usize] -= 1;
                    // Added kongs can be robbed; defensive styles shy away when someone is close.
                    if self.style.defence * ctx.threat > 1.5 {
                        continue;
                    }
                    if let Some(m) = melds.iter_mut().find(|m| m.tile == t && m.kind == MeldKind::Pong) {
                        m.kind = MeldKind::AddedKong;
                    }
                }
                let need = 4 - melds.len();
                let before = shanten(&{
                    let mut x = hand;
                    x[tile as usize] -= 1;
                    x
                }, 4 - ctx.melds.len());
                // After a kong we draw a replacement, so compare 3n+1-tile shapes.
                let after = shanten(&h, need);
                if after <= before {
                    return a;
                }
            }
        }
        let _ = score;
        Action::Discard(tile)
    }

    fn claim_turn(&mut self, v: &View, acts: &[Action], tile: Tile) -> Action {
        if acts.contains(&Action::Ron) {
            return Action::Ron;
        }
        let ctx = self.make_ctx(v);
        let hand = *v.hand();
        let need = 4 - ctx.melds.len();
        let (cur_sh, cur_score) = eval_hand(&hand, need, &ctx, &self.style, ctx.melds);
        let mut best: (Action, f32) = (Action::Pass, cur_score);
        for &a in acts {
            let (h, meld) = match a {
                Action::Pong | Action::ExposedKong => {
                    let mut h = hand;
                    let n = if a == Action::Pong { 2 } else { 3 };
                    h[tile as usize] -= n;
                    let kind = if a == Action::Pong { MeldKind::Pong } else { MeldKind::ExposedKong };
                    (h, Meld { kind, tile, claimed: Some(tile), from: None })
                }
                Action::Chow(p) => {
                    let low = tile - p;
                    let mut h = hand;
                    for t in low..low + 3 {
                        if t != tile {
                            h[t as usize] -= 1;
                        }
                    }
                    (h, Meld { kind: MeldKind::Chow, tile: low, claimed: Some(tile), from: None })
                }
                _ => continue,
            };
            let mut melds = ctx.melds.to_vec();
            melds.push(meld);
            let score = if a == Action::ExposedKong {
                // replacement draw follows: evaluate the 3n+1 hand directly
                eval_hand(&h, 4 - melds.len(), &ctx, &self.style, &melds).1
            } else {
                let (_, s, sh) = self.best_discard(&h, &melds, &ctx);
                if sh >= cur_sh {
                    continue; // claim must move us closer to ready
                }
                s
            };
            // Opening the hand forfeits men qing; demand more gain the less the style likes claiming.
            let breaks_men_qing = !ctx.open();
            let valued = matches!(a, Action::Pong | Action::ExposedKong)
                && valued_honor(tile, ctx.seat_wind, ctx.prevailing);
            let mut threshold = (1.0 - self.style.claim) * 30.0;
            if breaks_men_qing && !valued {
                threshold += 10.0;
            }
            let (secured, _) = tai_outlook(&h, &melds, &ctx, &self.style);
            if !secured && !valued {
                continue;
            }
            let gain = score - cur_score - threshold + self.jitter();
            if gain > 0.0 && score - threshold > best.1 {
                best = (a, score - threshold);
            }
        }
        best.0
    }
}

impl Bot for HeuristicBot {
    fn act(&mut self, v: &View) -> Action {
        let acts = v.legal_actions();
        debug_assert!(!acts.is_empty());
        if acts.len() == 1 {
            return acts[0];
        }
        match v.phase().clone() {
            Phase::SelfTurn { .. } => self.own_turn(v, &acts),
            Phase::Claim { tile, .. } => self.claim_turn(v, &acts, tile),
            Phase::RobKong { .. } => {
                if acts.contains(&Action::Ron) {
                    Action::Ron
                } else {
                    Action::Pass
                }
            }
            Phase::Over => unreachable!(),
        }
    }
    fn name(&self) -> String {
        self.style.name.clone()
    }
}

/// Uniformly random legal moves, but always takes a win. Useful as a floor baseline.
pub struct RandomBot {
    rng: Rng,
}
impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot { rng: Rng::new(seed) }
    }
}
impl Bot for RandomBot {
    fn act(&mut self, v: &View) -> Action {
        let acts = v.legal_actions();
        for w in [Action::Tsumo, Action::Ron] {
            if acts.contains(&w) {
                return w;
            }
        }
        let discards: Vec<Action> = acts.iter().copied().filter(|a| matches!(a, Action::Discard(_))).collect();
        if !discards.is_empty() {
            return discards[self.rng.below(discards.len() as u64) as usize];
        }
        Action::Pass
    }
    fn name(&self) -> String {
        "random".into()
    }
}

/// Pure tile efficiency: always wins when it can, never claims, and discards the
/// tile that keeps the lowest shanten with the most useful tiles. No defence, no
/// tai planning. The baseline every learned model must beat on the scenario bank.
pub struct EfficiencyBot;
impl Bot for EfficiencyBot {
    fn act(&mut self, v: &View) -> Action {
        let acts = v.legal_actions();
        for w in [Action::Tsumo, Action::Ron] {
            if acts.contains(&w) {
                return w;
            }
        }
        if matches!(v.phase(), Phase::SelfTurn { .. }) {
            let r = crate::scenario::efficiency_ranking(v);
            return Action::Discard(r[0].0);
        }
        Action::Pass
    }
    fn name(&self) -> String {
        "efficiency".into()
    }
}

pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    if name == "random" {
        return Some(Box::new(RandomBot::new(seed)));
    }
    if name == "efficiency" {
        return Some(Box::new(EfficiencyBot));
    }
    Style::by_name(name).map(|s| Box::new(HeuristicBot::new(s, seed)) as Box<dyn Bot>)
}
