//! Shanten (tiles-away-from-ready) calculation.
//!
//! Shanten -1 means the hand is complete, 0 means ready (one tile away), etc.
//! Standard form: 4 sets + 1 pair. Thirteen Orphans is handled separately.
//!
//! Speed matters because bots call this hundreds of times per decision, so each
//! suit is looked up in a precomputed table over all 5^9 count vectors. For every
//! suit vector the table stores, for (has_pair, melds) the maximum number of
//! partial sets ("taatsu"). Suits are then merged with a tiny max-plus convolution.

use crate::tile::{Counts, NUM_KINDS, ORPHANS};
use std::sync::OnceLock;

const SUIT_STATES: usize = 1_953_125; // 5^9
const NEG: i8 = -1;

/// table[p][m] = max taatsu, or -1 if (p, m) is unreachable.
type Entry = [[i8; 5]; 2];

static SUIT_TABLE: OnceLock<Vec<Entry>> = OnceLock::new();
static HONOR_TABLE: OnceLock<Vec<Entry>> = OnceLock::new();

const POW5: [usize; 10] = [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125];

fn empty_entry() -> Entry {
    let mut e = [[NEG; 5]; 2];
    e[0][0] = 0;
    e
}

fn merge_into(dst: &mut Entry, src: &Entry, dm: usize, dt: i8, dp: usize) {
    for p in 0..2 {
        let np = p + dp;
        if np > 1 {
            continue;
        }
        for m in 0..5 {
            let t = src[p][m];
            if t < 0 {
                continue;
            }
            let nm = m + dm;
            if nm > 4 {
                continue;
            }
            let nt = (t + dt).min(4);
            if nt > dst[np][nm] {
                dst[np][nm] = nt;
            }
        }
    }
}

fn build_suit_table() -> Vec<Entry> {
    let mut table: Vec<Entry> = vec![[[NEG; 5]; 2]; SUIT_STATES];
    table[0] = empty_entry();
    let mut c = [0usize; 9];
    for idx in 1..SUIT_STATES {
        // decode
        let mut x = idx;
        for d in c.iter_mut() {
            *d = x % 5;
            x /= 5;
        }
        let i = c.iter().position(|&v| v > 0).unwrap();
        let mut e = [[NEG; 5]; 2];
        // leave one tile isolated
        merge_into(&mut e, &table[idx - POW5[i]], 0, 0, 0);
        if c[i] >= 3 {
            merge_into(&mut e, &table[idx - 3 * POW5[i]], 1, 0, 0);
        }
        if c[i] >= 2 {
            let s = table[idx - 2 * POW5[i]];
            merge_into(&mut e, &s, 0, 0, 1); // pair as head
            merge_into(&mut e, &s, 0, 1, 0); // pair as partial set
        }
        if i <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
            merge_into(&mut e, &table[idx - POW5[i] - POW5[i + 1] - POW5[i + 2]], 1, 0, 0);
        }
        if i <= 7 && c[i + 1] > 0 {
            merge_into(&mut e, &table[idx - POW5[i] - POW5[i + 1]], 0, 1, 0);
        }
        if i <= 6 && c[i + 2] > 0 {
            merge_into(&mut e, &table[idx - POW5[i] - POW5[i + 2]], 0, 1, 0);
        }
        table[idx] = e;
    }
    table
}

fn build_honor_table() -> Vec<Entry> {
    // Honours can't form sequences, so each kind is independent: index 5^7 states.
    let n = POW5[7];
    let mut table: Vec<Entry> = vec![[[NEG; 5]; 2]; n];
    table[0] = empty_entry();
    let mut c = [0usize; 7];
    for idx in 1..n {
        let mut x = idx;
        for d in c.iter_mut() {
            *d = x % 5;
            x /= 5;
        }
        let i = c.iter().position(|&v| v > 0).unwrap();
        let mut e = [[NEG; 5]; 2];
        merge_into(&mut e, &table[idx - POW5[i]], 0, 0, 0);
        if c[i] >= 3 {
            merge_into(&mut e, &table[idx - 3 * POW5[i]], 1, 0, 0);
        }
        if c[i] >= 2 {
            let s = table[idx - 2 * POW5[i]];
            merge_into(&mut e, &s, 0, 0, 1);
            merge_into(&mut e, &s, 0, 1, 0);
        }
        table[idx] = e;
    }
    table
}

/// Force table construction (about a second); call once up front in benchmarks.
pub fn init_tables() {
    SUIT_TABLE.get_or_init(build_suit_table);
    HONOR_TABLE.get_or_init(build_honor_table);
}

#[inline]
fn suit_index(c: &Counts, base: usize) -> usize {
    let mut idx = 0;
    for i in (0..9).rev() {
        idx = idx * 5 + c[base + i] as usize;
    }
    idx
}

#[inline]
fn honor_index(c: &Counts) -> usize {
    let mut idx = 0;
    for i in (0..7).rev() {
        idx = idx * 5 + c[27 + i] as usize;
    }
    idx
}

fn combine(a: &Entry, b: &Entry) -> Entry {
    let mut out = [[NEG; 5]; 2];
    for pa in 0..2 {
        for ma in 0..5 {
            let ta = a[pa][ma];
            if ta < 0 {
                continue;
            }
            for pb in 0..(2 - pa) {
                for mb in 0..(5 - ma) {
                    let tb = b[pb][mb];
                    if tb < 0 {
                        continue;
                    }
                    let t = (ta + tb).min(4);
                    let slot = &mut out[pa + pb][ma + mb];
                    if t > *slot {
                        *slot = t;
                    }
                }
            }
        }
    }
    out
}

/// Shanten of the standard 4-sets-plus-pair form.
/// `melds_needed` = 4 minus the number of melds already exposed (or declared kongs).
pub fn shanten_standard(c: &Counts, melds_needed: usize) -> i32 {
    let st = SUIT_TABLE.get_or_init(build_suit_table);
    let ht = HONOR_TABLE.get_or_init(build_honor_table);
    let mut acc = st[suit_index(c, 0)];
    acc = combine(&acc, &st[suit_index(c, 9)]);
    acc = combine(&acc, &st[suit_index(c, 18)]);
    acc = combine(&acc, &ht[honor_index(c)]);
    let n = melds_needed as i32;
    let mut best = 8;
    for p in 0..2 {
        for m in 0..5 {
            let t = acc[p][m];
            if t < 0 || m as i32 > n {
                continue;
            }
            let t = (t as i32).min(n - m as i32);
            let s = 2 * n - 2 * m as i32 - t - p as i32;
            if s < best {
                best = s;
            }
        }
    }
    best
}

/// Thirteen Orphans shanten (only possible with no exposed melds).
pub fn shanten_orphans(c: &Counts) -> i32 {
    let mut kinds = 0;
    let mut pair = 0;
    for &t in ORPHANS.iter() {
        let n = c[t as usize];
        if n > 0 {
            kinds += 1;
            if n >= 2 {
                pair = 1;
            }
        }
    }
    13 - kinds - pair
}

/// Overall shanten: min over standard form and (when fully concealed) Thirteen Orphans.
pub fn shanten(c: &Counts, melds_needed: usize) -> i32 {
    let s = shanten_standard(c, melds_needed);
    if melds_needed == 4 {
        s.min(shanten_orphans(c))
    } else {
        s
    }
}

/// Tiles that would lower shanten if drawn, weighted by how many copies remain unseen.
/// `unseen[k]` is the number of copies of kind k this player cannot see.
/// Returns (total useful copies, mask of useful kinds).
pub fn ukeire(c: &Counts, melds_needed: usize, unseen: &Counts) -> (u32, u64) {
    let base = shanten(c, melds_needed);
    let mut cc = *c;
    let mut total = 0u32;
    let mut mask = 0u64;
    for k in 0..NUM_KINDS {
        if unseen[k] == 0 || cc[k] >= 4 {
            continue;
        }
        cc[k] += 1;
        if shanten(&cc, melds_needed) < base {
            total += unseen[k] as u32;
            mask |= 1 << k;
        }
        cc[k] -= 1;
    }
    (total, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{counts_of, parse_tiles};

    fn sh(s: &str) -> i32 {
        let t = parse_tiles(s).unwrap();
        shanten(&counts_of(&t), 4)
    }

    #[test]
    fn complete_hands() {
        assert_eq!(sh("123m456p789s11122z"), -1);
        assert_eq!(sh("19m19p19s12345677z"), -1);
        assert_eq!(sh("11122233344455m"), -1);
    }

    #[test]
    fn ready_hands() {
        assert_eq!(sh("123m456p789s1112z"), 0);
        assert_eq!(sh("19m19p19s1234567z"), 0); // 13-sided orphans wait
        assert_eq!(sh("1112345678999m"), 0); // nine gates shape
    }

    #[test]
    fn far_hands() {
        // 13 unrelated tiles: no pairs, no partials beyond what exists
        assert!(sh("147m258p369s1234z") >= 4);
    }

    #[test]
    fn exposed_melds() {
        // one exposed meld: 10 concealed tiles, 3 sets + pair needed
        let t = parse_tiles("123m456p789s1z").unwrap();
        // 10 tiles: 123m 456p 789s + single -> tenpai on tanki
        assert_eq!(shanten(&counts_of(&t), 3), 0);
    }
}
