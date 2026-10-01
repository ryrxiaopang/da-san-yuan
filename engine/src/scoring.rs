//! Melds, win decomposition and tai scoring for the team's Singapore ruleset.
//! RULES.md in the repo root describes every rule here in plain language.

use crate::tile::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeldKind {
    /// Sequence starting at `tile`.
    Chow,
    Pong,
    /// Exposed kong claimed from a discard.
    ExposedKong,
    /// Pong upgraded with a self-drawn fourth tile.
    AddedKong,
    ConcealedKong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Meld {
    pub kind: MeldKind,
    /// Lowest tile for a chow, the tile itself otherwise.
    pub tile: Tile,
    /// Which tile was claimed (for chows: which of the three), if claimed.
    pub claimed: Option<Tile>,
    /// Seat the claimed tile came from.
    pub from: Option<u8>,
}

impl Meld {
    pub fn is_kong(&self) -> bool {
        matches!(self.kind, MeldKind::ExposedKong | MeldKind::AddedKong | MeldKind::ConcealedKong)
    }
    pub fn is_triplet_like(&self) -> bool {
        !matches!(self.kind, MeldKind::Chow)
    }
    /// True when the meld was built from another player's discard (breaks men qing).
    pub fn is_open(&self) -> bool {
        !matches!(self.kind, MeldKind::ConcealedKong)
    }
    pub fn tiles(&self) -> Vec<Tile> {
        match self.kind {
            MeldKind::Chow => vec![self.tile, self.tile + 1, self.tile + 2],
            MeldKind::Pong => vec![self.tile; 3],
            _ => vec![self.tile; 4],
        }
    }
}

/// A set inside a winning decomposition of the concealed tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Set {
    pub chow: bool,
    pub tile: Tile,
}

#[derive(Clone, Debug)]
pub struct Decomposition {
    pub pair: Tile,
    pub sets: Vec<Set>,
}

/// All ways to split `c` into `n` sets plus one pair (standard form).
pub fn decompositions(c: &Counts, n: usize) -> Vec<Decomposition> {
    let total: usize = c.iter().map(|&x| x as usize).sum();
    if total != 3 * n + 2 {
        return vec![];
    }
    let mut out = Vec::new();
    let mut cc = *c;
    for p in 0..NUM_KINDS {
        if cc[p] >= 2 {
            cc[p] -= 2;
            let mut sets = Vec::with_capacity(4);
            split_sets(&mut cc, 0, &mut sets, &mut |s| {
                out.push(Decomposition { pair: p as Tile, sets: s.to_vec() })
            });
            cc[p] += 2;
        }
    }
    out
}

fn split_sets(c: &mut Counts, start: usize, sets: &mut Vec<Set>, emit: &mut dyn FnMut(&[Set])) {
    let mut i = start;
    while i < NUM_KINDS && c[i] == 0 {
        i += 1;
    }
    if i == NUM_KINDS {
        emit(sets);
        return;
    }
    if c[i] >= 3 {
        c[i] -= 3;
        sets.push(Set { chow: false, tile: i as Tile });
        split_sets(c, i, sets, emit);
        sets.pop();
        c[i] += 3;
    }
    if i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        c[i + 2] -= 1;
        sets.push(Set { chow: true, tile: i as Tile });
        split_sets(c, i, sets, emit);
        sets.pop();
        c[i] += 1;
        c[i + 1] += 1;
        c[i + 2] += 1;
    }
}

pub fn is_thirteen_orphans(c: &Counts) -> bool {
    let mut pair = false;
    for &t in ORPHANS.iter() {
        match c[t as usize] {
            0 => return false,
            1 => {}
            2 => pair = true,
            _ => return false,
        }
    }
    pair && c.iter().map(|&x| x as u32).sum::<u32>() == 14
}

/// Is the concealed part plus exposed melds a complete hand shape?
pub fn is_complete_shape(c: &Counts, n_melds: usize) -> bool {
    if n_melds == 0 && is_thirteen_orphans(c) {
        return true;
    }
    crate::shanten::shanten_standard(c, 4 - n_melds) == -1
}

// ---------------------------------------------------------------- scoring

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    SeatFlower,
    FlowerSet,
    Animal,
    DragonPong,
    SeatWindPong,
    PrevailingWindPong,
    AllPongs,
    HalfFlush,
    PingHu,
    PingHuWithFlowers,
    MenQing,
    SmallThreeDragons,
    SmallFourWinds,
    KongReplacementWin,
    RobbingKong,
    LastTile,
    // limit hands
    FullFlush,
    ThirteenOrphans,
    BigThreeDragons,
    BigFourWinds,
    AllHonours,
    FourConcealedPongs,
    FourKongs,
    NineGates,
    HeavenlyHand,
    EarthlyHand,
}

impl Pattern {
    pub fn name(&self) -> &'static str {
        use Pattern::*;
        match self {
            SeatFlower => "Flower matching seat",
            FlowerSet => "Complete flower set",
            Animal => "Animal",
            DragonPong => "Dragon pong",
            SeatWindPong => "Seat wind pong",
            PrevailingWindPong => "Prevailing wind pong",
            AllPongs => "All pongs (pong pong hu)",
            HalfFlush => "Half flush",
            PingHu => "Ping hu",
            PingHuWithFlowers => "Open ping hu with non-scoring flowers",
            MenQing => "Men qing (fully concealed)",
            SmallThreeDragons => "Small three dragons",
            SmallFourWinds => "Small four winds",
            KongReplacementWin => "Win on kong replacement",
            RobbingKong => "Robbing a kong",
            LastTile => "Win on last tile",
            FullFlush => "Full flush",
            ThirteenOrphans => "Thirteen orphans",
            BigThreeDragons => "Big three dragons",
            BigFourWinds => "Big four winds",
            AllHonours => "All honours",
            FourConcealedPongs => "Four concealed pongs",
            FourKongs => "Four kongs",
            NineGates => "Nine gates",
            HeavenlyHand => "Heavenly hand",
            EarthlyHand => "Earthly hand",
        }
    }
}

pub const MIN_TAI: u8 = 1;
pub const MAX_TAI: u8 = 5;

#[derive(Clone, Debug, Default)]
pub struct WinContext {
    /// 0 = East .. 3 = North, relative to the dealer.
    pub seat_wind: u8,
    pub prevailing_wind: u8,
    pub self_draw: bool,
    pub win_tile: Tile,
    pub kong_replacement: bool,
    pub robbing_kong: bool,
    pub last_tile: bool,
    pub heavenly: bool,
    pub earthly: bool,
}

#[derive(Clone, Debug)]
pub struct Score {
    /// Tai after the cap.
    pub tai: u8,
    /// Tai before the cap.
    pub raw_tai: u8,
    pub items: Vec<(Pattern, u8)>,
}

/// Tai from bonus tiles alone (flowers and animals).
pub fn bonus_tai(bonus: &[Tile], seat_wind: u8) -> (u8, Vec<(Pattern, u8)>) {
    let mut items = Vec::new();
    let mut tai = 0;
    for set in 0..2u8 {
        let held: Vec<Tile> = bonus.iter().copied().filter(|&b| is_flower(b) && flower_set(b) == set).collect();
        if held.len() == 4 {
            items.push((Pattern::FlowerSet, 2));
            tai += 2;
        } else if held.iter().any(|&b| flower_number(b) == seat_wind + 1) {
            items.push((Pattern::SeatFlower, 1));
            tai += 1;
        }
    }
    for &b in bonus {
        if is_animal(b) {
            items.push((Pattern::Animal, 1));
            tai += 1;
        }
    }
    (tai, items)
}

/// Score a complete hand. `concealed` includes the winning tile.
/// Returns None if the tiles do not form a winning shape.
pub fn score_hand(concealed: &Counts, melds: &[Meld], bonus: &[Tile], ctx: &WinContext) -> Option<Score> {
    let n_melds = melds.len();
    let n_sets = 4 - n_melds;
    let men_qing = melds.iter().all(|m| !m.is_open());
    let (btai, bitems) = bonus_tai(bonus, ctx.seat_wind);
    let has_bonus = !bonus.is_empty();

    let mut situational: Vec<(Pattern, u8)> = Vec::new();
    if ctx.kong_replacement {
        situational.push((Pattern::KongReplacementWin, 1));
    }
    if ctx.robbing_kong {
        situational.push((Pattern::RobbingKong, 1));
    }
    if ctx.last_tile {
        situational.push((Pattern::LastTile, 1));
    }

    // All playing tiles in the hand (for flush/honour checks).
    let mut all = *concealed;
    for m in melds {
        for t in m.tiles() {
            all[t as usize] += 1;
        }
    }

    let mut best: Option<Score> = None;
    let mut consider = |items: Vec<(Pattern, u8)>| {
        let raw: u32 = items.iter().map(|&(_, t)| t as u32).sum();
        let raw = raw.min(255) as u8;
        let s = Score { tai: raw.min(MAX_TAI), raw_tai: raw, items };
        if best.as_ref().map_or(true, |b| (s.tai, s.raw_tai) > (b.tai, b.raw_tai)) {
            best = Some(s);
        }
    };

    let mut limit_common: Vec<(Pattern, u8)> = Vec::new();
    if ctx.heavenly {
        limit_common.push((Pattern::HeavenlyHand, MAX_TAI));
    }
    if ctx.earthly {
        limit_common.push((Pattern::EarthlyHand, MAX_TAI));
    }

    // Thirteen orphans
    if n_melds == 0 && is_thirteen_orphans(concealed) {
        let mut items = vec![(Pattern::ThirteenOrphans, MAX_TAI)];
        items.extend(limit_common.iter().copied());
        consider(items);
    }

    let suits_used: Vec<u8> = (0..3u8).filter(|&s| (0..9).any(|r| all[(s * 9 + r) as usize] > 0)).collect();
    let has_honor = (27..34).any(|t| all[t] > 0);
    let full_flush = suits_used.len() == 1 && !has_honor;
    let half_flush = suits_used.len() == 1 && has_honor;
    let all_honours = suits_used.is_empty();

    for d in decompositions(concealed, n_sets) {
        // Each placement of the winning tile can change wait-dependent patterns.
        let mut placements: Vec<Option<usize>> = Vec::new(); // None = in pair
        if d.pair == ctx.win_tile {
            placements.push(None);
        }
        for (i, s) in d.sets.iter().enumerate() {
            let covers = if s.chow {
                ctx.win_tile >= s.tile && ctx.win_tile <= s.tile + 2
            } else {
                s.tile == ctx.win_tile
            };
            if covers {
                placements.push(Some(i));
            }
        }
        for placement in placements {
            let mut items: Vec<(Pattern, u8)> = Vec::new();
            items.extend(limit_common.iter().copied());
            items.extend(situational.iter().copied());

            // triplet-like sets: (tile, concealed?)
            let mut trips: Vec<(Tile, bool)> = Vec::new();
            let mut chows = 0;
            for (i, s) in d.sets.iter().enumerate() {
                if s.chow {
                    chows += 1;
                } else {
                    let completed_by_discard = !ctx.self_draw && placement == Some(i);
                    trips.push((s.tile, !completed_by_discard));
                }
            }
            for m in melds {
                if m.is_triplet_like() {
                    trips.push((m.tile, m.kind == MeldKind::ConcealedKong));
                } else {
                    chows += 1;
                }
            }
            let n_kongs = melds.iter().filter(|m| m.is_kong()).count();

            let dragon_trips = trips.iter().filter(|(t, _)| is_dragon(*t)).count();
            let wind_trips = trips.iter().filter(|(t, _)| is_wind(*t)).count();
            let seat_w = wind_tile(ctx.seat_wind);
            let prev_w = wind_tile(ctx.prevailing_wind);

            // limit hands
            if dragon_trips == 3 {
                items.push((Pattern::BigThreeDragons, MAX_TAI));
            }
            if wind_trips == 4 {
                items.push((Pattern::BigFourWinds, MAX_TAI));
            }
            if all_honours {
                items.push((Pattern::AllHonours, MAX_TAI));
            }
            if full_flush {
                items.push((Pattern::FullFlush, MAX_TAI));
            }
            if trips.len() == 4 && trips.iter().all(|&(_, conc)| conc) {
                items.push((Pattern::FourConcealedPongs, MAX_TAI));
            }
            if n_kongs == 4 {
                items.push((Pattern::FourKongs, MAX_TAI));
            }
            if n_melds == 0 && full_flush && is_nine_gates(concealed, ctx.win_tile) {
                items.push((Pattern::NineGates, MAX_TAI));
            }

            // dragons
            if dragon_trips == 2 && is_dragon(d.pair) {
                items.push((Pattern::SmallThreeDragons, 4));
            } else if dragon_trips < 3 {
                for _ in 0..dragon_trips {
                    items.push((Pattern::DragonPong, 1));
                }
            }
            // winds
            if wind_trips == 3 && is_wind(d.pair) {
                items.push((Pattern::SmallFourWinds, 4));
            } else if wind_trips < 4 {
                if trips.iter().any(|(t, _)| *t == seat_w) {
                    items.push((Pattern::SeatWindPong, 1));
                }
                if trips.iter().any(|(t, _)| *t == prev_w) {
                    items.push((Pattern::PrevailingWindPong, 1));
                }
            }

            if trips.len() == 4 {
                items.push((Pattern::AllPongs, 2));
            }
            if half_flush {
                items.push((Pattern::HalfFlush, 2));
            }

            // ping hu: four chows, plain pair, two-sided wait
            if chows == 4 && !is_dragon(d.pair) && d.pair != seat_w && d.pair != prev_w {
                let two_sided = match placement {
                    Some(i) => {
                        let s = d.sets[i];
                        s.chow && {
                            let r = rank(s.tile);
                            (ctx.win_tile == s.tile && r <= 5) || (ctx.win_tile == s.tile + 2 && r >= 1)
                        }
                    }
                    None => false,
                };
                if two_sided {
                    // Scoring bonus tiles (seat flower, animal, flower set) void ping hu.
                    // Non-scoring flowers: a fully concealed ping hu keeps its 4 tai,
                    // an open one (exposed chows) drops to 1 tai.
                    let concealed_hand = n_melds == 0;
                    if btai == 0 {
                        if !has_bonus || concealed_hand {
                            items.push((Pattern::PingHu, 4));
                        } else {
                            items.push((Pattern::PingHuWithFlowers, 1));
                        }
                    }
                }
            }

            if men_qing {
                items.push((Pattern::MenQing, 1));
            }
            items.extend(bitems.iter().copied());
            consider(items);
        }
    }
    best
}

fn is_nine_gates(c: &Counts, _win: Tile) -> bool {
    let s = (0..3).find(|&s| (0..9).any(|r| c[s * 9 + r] > 0));
    let Some(s) = s else { return false };
    let base = [3u8, 1, 1, 1, 1, 1, 1, 1, 3];
    let mut extra = 0;
    for r in 0..9 {
        let n = c[s * 9 + r];
        if n < base[r] {
            return false;
        }
        extra += n - base[r];
    }
    extra == 1
}

/// Unit price of a hand: 1 tai = 1, doubling per tai.
pub fn unit_points(tai: u8) -> i32 {
    1 << (tai.max(1) - 1)
}

/// Point transfers for a win. Index = seat. Sum is always zero.
/// Discard win: the discarder pays for everyone (4 / 8 / 16 / 32 / 64 for 1-5 tai).
/// Self-draw: each other player pays double (2 / 4 / 8 / 16 / 32).
pub fn payments(tai: u8, winner: usize, discarder: Option<usize>) -> [i32; 4] {
    let mut d = [0i32; 4];
    let unit = unit_points(tai);
    match discarder {
        Some(p) => {
            let amt = 4 * unit;
            d[p] -= amt;
            d[winner] += amt;
        }
        None => {
            for (s, v) in d.iter_mut().enumerate() {
                if s != winner {
                    *v -= 2 * unit;
                }
            }
            d[winner] += 6 * unit;
        }
    }
    d
}
