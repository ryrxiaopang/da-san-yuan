//! Tile encoding.
//!
//! Playing tiles (34 kinds, 4 copies each = 136):
//!   0..=8   characters 1-9  (m, 万)
//!   9..=17  dots 1-9        (p, 筒)
//!   18..=26 bamboo 1-9      (s, 条)
//!   27..=30 winds E S W N   (1z-4z)
//!   31..=33 dragons White Green Red (5z 白, 6z 发, 7z 中)
//!
//! Bonus tiles (one copy each = 12):
//!   34..=37 seasons 1-4  (春夏秋冬), notation f1-f4
//!   38..=41 plants 1-4   (梅兰菊竹), notation g1-g4
//!   42..=45 animals      cat, rat, rooster, centipede, notation a1-a4
//!
//! Total 148 tiles.

pub type Tile = u8;

pub const NUM_KINDS: usize = 34;
pub const NUM_TILE_IDS: usize = 46;
pub const NUM_BONUS: usize = 12;
pub const WALL_SIZE: usize = 148;

pub const EAST: Tile = 27;
pub const SOUTH: Tile = 28;
pub const WEST: Tile = 29;
pub const NORTH: Tile = 30;
pub const WHITE: Tile = 31;
pub const GREEN: Tile = 32;
pub const RED: Tile = 33;

pub const FIRST_BONUS: Tile = 34;
pub const FIRST_PLANT: Tile = 38;
pub const FIRST_ANIMAL: Tile = 42;

/// The 13 terminal and honour kinds used by Thirteen Orphans.
pub const ORPHANS: [Tile; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

#[inline]
pub fn is_bonus(t: Tile) -> bool {
    t >= FIRST_BONUS
}
#[inline]
pub fn is_flower(t: Tile) -> bool {
    (FIRST_BONUS..FIRST_ANIMAL).contains(&t)
}
#[inline]
pub fn is_animal(t: Tile) -> bool {
    t >= FIRST_ANIMAL && (t as usize) < NUM_TILE_IDS
}
/// Flower number 1-4 for a season/plant tile.
#[inline]
pub fn flower_number(t: Tile) -> u8 {
    debug_assert!(is_flower(t));
    (t - FIRST_BONUS) % 4 + 1
}
/// 0 = seasons, 1 = plants.
#[inline]
pub fn flower_set(t: Tile) -> u8 {
    (t - FIRST_BONUS) / 4
}
#[inline]
pub fn is_honor(t: Tile) -> bool {
    (27..34).contains(&t)
}
#[inline]
pub fn is_wind(t: Tile) -> bool {
    (27..31).contains(&t)
}
#[inline]
pub fn is_dragon(t: Tile) -> bool {
    (31..34).contains(&t)
}
#[inline]
pub fn is_suited(t: Tile) -> bool {
    t < 27
}
/// 0 = characters, 1 = dots, 2 = bamboo, 3 = honours.
#[inline]
pub fn suit(t: Tile) -> u8 {
    if t < 27 {
        t / 9
    } else {
        3
    }
}
/// Rank 0..=8 within a suit.
#[inline]
pub fn rank(t: Tile) -> u8 {
    t % 9
}
#[inline]
pub fn is_terminal(t: Tile) -> bool {
    t < 27 && (t % 9 == 0 || t % 9 == 8)
}
#[inline]
pub fn is_terminal_or_honor(t: Tile) -> bool {
    is_terminal(t) || is_honor(t)
}
/// Wind tile for a wind index 0..=3 (E, S, W, N).
#[inline]
pub fn wind_tile(w: u8) -> Tile {
    EAST + w
}

pub fn tile_name(t: Tile) -> String {
    match t {
        0..=26 => format!("{}{}", t % 9 + 1, ['m', 'p', 's'][(t / 9) as usize]),
        27..=33 => format!("{}z", t - 26),
        34..=37 => format!("f{}", t - 33),
        38..=41 => format!("g{}", t - 37),
        42..=45 => format!("a{}", t - 41),
        _ => "??".to_string(),
    }
}

/// Parse a hand string like "123m456p789s1155z f1 a2" into tile ids.
/// Bonus tiles are written as f1-f4, g1-g4, a1-a4 separated by spaces.
pub fn parse_tiles(s: &str) -> Result<Vec<Tile>, String> {
    let mut out = Vec::new();
    for word in s.split_whitespace() {
        let b = word.as_bytes();
        if b.len() == 2 && matches!(b[0], b'f' | b'g' | b'a') && (b'1'..=b'4').contains(&b[1]) {
            let n = b[1] - b'1';
            out.push(match b[0] {
                b'f' => FIRST_BONUS + n,
                b'g' => FIRST_PLANT + n,
                _ => FIRST_ANIMAL + n,
            });
            continue;
        }
        let mut pending: Vec<u8> = Vec::new();
        for &c in b {
            match c {
                b'0'..=b'9' => pending.push(c - b'0'),
                b'm' | b'p' | b's' | b'z' => {
                    if pending.is_empty() {
                        return Err(format!("suit '{}' with no digits in '{}'", c as char, word));
                    }
                    for &d in &pending {
                        let t = match c {
                            b'z' => {
                                if !(1..=7).contains(&d) {
                                    return Err(format!("bad honour {}z", d));
                                }
                                26 + d
                            }
                            _ => {
                                if !(1..=9).contains(&d) {
                                    return Err(format!("bad rank {}", d));
                                }
                                let base = match c {
                                    b'm' => 0,
                                    b'p' => 9,
                                    _ => 18,
                                };
                                base + d - 1
                            }
                        };
                        out.push(t);
                    }
                    pending.clear();
                }
                _ => return Err(format!("unexpected character '{}'", c as char)),
            }
        }
        if !pending.is_empty() {
            return Err(format!("digits without suit in '{}'", word));
        }
    }
    Ok(out)
}

pub fn hand_string(tiles: &[Tile]) -> String {
    let mut v: Vec<Tile> = tiles.to_vec();
    v.sort();
    let mut s = String::new();
    for suit_c in 0..4u8 {
        let digits: String = v
            .iter()
            .filter(|&&t| t < 34 && suit(t) == suit_c)
            .map(|&t| {
                if suit_c == 3 {
                    char::from(b'0' + (t - 26))
                } else {
                    char::from(b'1' + rank(t))
                }
            })
            .collect();
        if !digits.is_empty() {
            s.push_str(&digits);
            s.push(['m', 'p', 's', 'z'][suit_c as usize]);
        }
    }
    for &t in v.iter().filter(|&&t| t >= 34) {
        s.push(' ');
        s.push_str(&tile_name(t));
    }
    s
}

/// Counts of the 34 playing kinds.
pub type Counts = [u8; NUM_KINDS];

pub fn counts_of(tiles: &[Tile]) -> Counts {
    let mut c = [0u8; NUM_KINDS];
    for &t in tiles {
        if (t as usize) < NUM_KINDS {
            c[t as usize] += 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let t = parse_tiles("123m456p789s1257z f1 g4 a2").unwrap();
        assert_eq!(t.len(), 16);
        assert_eq!(t[9], EAST);
        assert_eq!(t[11], WHITE);
        assert_eq!(t[12], RED);
        assert_eq!(flower_number(t[13]), 1);
        assert_eq!(flower_number(t[14]), 4);
        assert!(is_animal(t[15]));
        assert_eq!(hand_string(&t), "123m456p789s1257z f1 g4 a2");
    }
}
