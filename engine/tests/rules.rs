//! Rule tests for the team's Singapore ruleset (see RULES.md).
//! If the team changes a rule, change RULES.md and the matching test together.

use dsy_engine::bots::{Bot, HeuristicBot, RandomBot, Style};
use dsy_engine::game::{Action, Config, Game, Phase};
use dsy_engine::obs::{self, View};
use dsy_engine::scoring::*;
use dsy_engine::selfplay::{play_hand, verify_hand_invariants};
use dsy_engine::tile::*;

fn t(s: &str) -> Vec<Tile> {
    parse_tiles(s).unwrap()
}
fn tile(s: &str) -> Tile {
    t(s)[0]
}

/// Score a hand string; `melds` like "pong:5z,chow:123m".
fn score(hand: &str, win: &str, melds: &[Meld], seat: u8, prev: u8, tsumo: bool) -> Option<Score> {
    let tiles = t(hand);
    let bonus: Vec<Tile> = tiles.iter().copied().filter(|&x| is_bonus(x)).collect();
    let ctx = WinContext { seat_wind: seat, prevailing_wind: prev, self_draw: tsumo, win_tile: tile(win), ..Default::default() };
    score_hand(&counts_of(&tiles), melds, &bonus, &ctx)
}
fn has(s: &Score, p: Pattern) -> bool {
    s.items.iter().any(|(x, _)| *x == p)
}
fn pong(s: &str) -> Meld {
    Meld { kind: MeldKind::Pong, tile: tile(s), claimed: None, from: Some(1) }
}
fn chow(s: &str) -> Meld {
    Meld { kind: MeldKind::Chow, tile: tile(s), claimed: None, from: Some(3) }
}

// ------------------------------------------------------------ tai patterns

#[test]
fn men_qing_is_one_tai() {
    // Concealed, no other tai source: 123m 456p 789s 234s + 99m pair, closed wait on 3s
    let s = score("123m456p789s234s99m", "3s", &[], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::MenQing));
    assert_eq!(s.tai, 1);
}

#[test]
fn open_hand_without_tai_is_zero() {
    let s = score("456p789s234s99m", "3s", &[chow("1m")], 1, 0, false).unwrap();
    assert_eq!(s.tai, 0, "{:?}", s.items);
}

#[test]
fn ping_hu_clean_is_four_tai() {
    // all chows, plain pair, two-sided wait (won on 1s completing 123s from 23s)
    let s = score("123m456p789s123s99m", "1s", &[], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::PingHu));
    // ping hu 4 + men qing 1 = 5
    assert_eq!(s.raw_tai, 5);
}

#[test]
fn ping_hu_needs_two_sided_wait() {
    // 3s completes 1-2 edge -> not ping hu
    let s = score("123m456p789s123s99m", "3s", &[], 1, 0, false).unwrap();
    // 3s could also be the top of 123s only, which is an edge wait from 12s
    assert!(!has(&s, Pattern::PingHu), "{:?}", s.items);
}

#[test]
fn ping_hu_with_non_matching_flowers_is_one_tai() {
    // seat South (1) -> matching flowers are f2/g2. f1 does not match.
    let s = score("123m456p789s123s99m f1", "1s", &[], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::PingHuWithFlowers));
    assert!(!has(&s, Pattern::PingHu));
    assert_eq!(s.raw_tai, 2); // 1 ping hu w/ flowers + 1 men qing
}

#[test]
fn ping_hu_void_with_scoring_bonus() {
    let s = score("123m456p789s123s99m f2", "1s", &[], 1, 0, false).unwrap();
    assert!(!has(&s, Pattern::PingHu) && !has(&s, Pattern::PingHuWithFlowers));
    assert!(has(&s, Pattern::SeatFlower));
}

#[test]
fn ping_hu_pair_cannot_be_valued_honour() {
    let s = score("123m456p789s123s55z", "1s", &[], 1, 0, false).unwrap();
    assert!(!has(&s, Pattern::PingHu));
}

#[test]
fn full_flush_is_five_tai() {
    let s = score("11123455678999m", "5m", &[], 1, 0, false);
    // 14 tiles needed; build a clean full flush
    let s = s.or_else(|| score("1112345678999m5m", "5m", &[], 1, 0, false)).unwrap();
    assert!(has(&s, Pattern::FullFlush));
    assert_eq!(s.tai, 5);
}

#[test]
fn half_flush_two_tai() {
    let s = score("123m456m789m22m111z", "2m", &[], 2, 0, false).unwrap();
    // seat West(2), prevailing East(0): 111z East pong = prevailing wind
    assert!(has(&s, Pattern::HalfFlush));
    assert!(has(&s, Pattern::PrevailingWindPong));
}

#[test]
fn dragon_and_wind_pongs() {
    let s = score("123m456p789s22p", "2p", &[pong("4p")], 0, 0, false).unwrap();
    assert_eq!(s.tai, 0, "open hand with no tai source: {:?}", s.items);
    let s = score("123m456p22p111z", "2p", &[pong("5z")], 0, 0, false).unwrap();
    assert!(has(&s, Pattern::DragonPong));
    assert!(has(&s, Pattern::SeatWindPong));
    assert!(has(&s, Pattern::PrevailingWindPong));
    assert_eq!(s.raw_tai, 3);
}

#[test]
fn all_pongs_two_tai() {
    let s = score("222m555p888s99s", "9s", &[pong("4p")], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::AllPongs));
}

#[test]
fn small_three_dragons() {
    let s = score("555z666z77z123m", "7z", &[pong("9p")], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::SmallThreeDragons));
    assert!(!has(&s, Pattern::DragonPong));
}

#[test]
fn limit_hands() {
    let s = score("19m19p19s12345677z", "7z", &[], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::ThirteenOrphans));
    assert_eq!(s.tai, 5);
    let s = score("555z666z777z123m99p", "9p", &[], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::BigThreeDragons));
    let s = score("111z222z333z444z99p", "9p", &[], 1, 0, true).unwrap();
    assert!(has(&s, Pattern::BigFourWinds));
}

#[test]
fn four_concealed_pongs_needs_self_draw_or_pair_wait() {
    // completed pong by discard -> not four concealed
    let s = score("222m555p888s999s11p", "9s", &[], 1, 0, false).unwrap();
    assert!(!has(&s, Pattern::FourConcealedPongs));
    let s = score("222m555p888s999s11p", "9s", &[], 1, 0, true).unwrap();
    assert!(has(&s, Pattern::FourConcealedPongs));
    // pair wait on discard is fine
    let s = score("222m555p888s999s11p", "1p", &[], 1, 0, false).unwrap();
    assert!(has(&s, Pattern::FourConcealedPongs));
}

#[test]
fn bonus_tiles() {
    // seat East: f1 and g1 match; a1 animal
    let (tai, _) = bonus_tai(&t("f1 g1 a1 f3"), 0);
    assert_eq!(tai, 3);
    let (tai, items) = bonus_tai(&t("f1 f2 f3 f4"), 2);
    assert_eq!(tai, 2);
    assert!(items.iter().any(|(p, _)| *p == Pattern::FlowerSet));
}

#[test]
fn tai_cap_is_five() {
    let s = score("1112345678999m5m f1 a1 a2", "5m", &[], 0, 0, true).unwrap();
    assert_eq!(s.tai, 5);
    assert!(s.raw_tai > 5);
}

// ---------------------------------------------------------------- payments

#[test]
fn payments_match_team_table() {
    // discarder pays for everyone: 4, 8, 16, 32, 64
    for (tai, amt) in [(1u8, 4), (2, 8), (3, 16), (4, 32), (5, 64)] {
        let d = payments(tai, 0, Some(2));
        assert_eq!(d, [amt, 0, -amt, 0]);
    }
    // self draw: each pays 2, 4, 8, 16, 32
    for (tai, each) in [(1u8, 2), (2, 4), (3, 8), (4, 16), (5, 32)] {
        let d = payments(tai, 1, None);
        assert_eq!(d, [-each, 3 * each, -each, -each]);
    }
}

// ----------------------------------------------------------- game engine

#[test]
fn deal_is_correct() {
    let g = Game::new(Config::default(), 42);
    assert_eq!(g.players[0].hand_size(), 14);
    for s in 1..4 {
        assert_eq!(g.players[s].hand_size(), 13);
    }
    let bonus: usize = g.players.iter().map(|p| p.bonus.len()).sum();
    let (drawn, held) = g.tile_census();
    assert_eq!(drawn, held);
    assert_eq!(drawn, 53 + bonus);
}

#[test]
fn same_seed_same_game() {
    let run = |seed| {
        let mut bots: [Box<dyn Bot>; 4] =
            std::array::from_fn(|i| Box::new(HeuristicBot::new(Style::archetypes()[i].clone(), 9)) as Box<dyn Bot>);
        let (_, r) = play_hand(Game::new(Config::default(), seed), &mut bots, |_, _, _| {});
        (r.winner, r.deltas, r.turns)
    };
    assert_eq!(run(5), run(5));
}

#[test]
fn thousands_of_hands_keep_invariants() {
    let styles = Style::archetypes();
    let mut wins = 0;
    for seed in 0..3000u64 {
        let cfg = Config { dealer: (seed % 4) as u8, prevailing_wind: ((seed / 4) % 4) as u8, ..Default::default() };
        let mut bots: [Box<dyn Bot>; 4] = std::array::from_fn(|i| {
            if seed % 7 == 0 && i == 2 {
                Box::new(RandomBot::new(seed)) as Box<dyn Bot>
            } else {
                Box::new(HeuristicBot::new(styles[(i + seed as usize) % 4].clone(), seed)) as Box<dyn Bot>
            }
        });
        let mut checked = 0;
        let (g, r) = play_hand(Game::new(cfg, seed), &mut bots, |g, seat, a| {
            // every chosen action must be legal and the observation must encode
            assert!(g.legal_actions(seat).contains(&a));
            if checked < 3 {
                let mut buf = vec![0u8; obs::OBS_LEN];
                obs::encode_obs(g, seat, &mut buf);
                assert_eq!(buf[obs::OFF_HAND..obs::OFF_HAND + 34].iter().map(|&x| x as usize).sum::<usize>(), g.players[seat as usize].hand_size());
                checked += 1;
            }
        });
        verify_hand_invariants(&g).unwrap_or_else(|e| panic!("seed {}: {}", seed, e));
        if r.winner.is_some() {
            wins += 1;
            assert!(r.score.as_ref().unwrap().tai >= 1);
        }
    }
    assert!(wins > 1500, "bots should win most hands, got {}", wins);
}

#[test]
fn bots_only_see_public_info() {
    // A bot's decision must not change if hidden information (other hands / wall) changes
    // while everything public stays the same. We check by swapping two unseen tiles
    // between opponents before the first decision.
    let g = Game::new(Config::default(), 77);
    let mut g2 = g.clone();
    // swap one tile between seat 1 and seat 2 hands (both hidden from seat 0)
    let a = (0..34).find(|&k| g2.players[1].hand[k] > 0 && g2.players[2].hand[k] == 0).unwrap();
    let b = (0..34).find(|&k| g2.players[2].hand[k] > 0 && g2.players[1].hand[k] == 0).unwrap();
    g2.players[1].hand[a] -= 1;
    g2.players[1].hand[b] += 1;
    g2.players[2].hand[b] -= 1;
    g2.players[2].hand[a] += 1;
    let mut b1 = HeuristicBot::new(Style::balanced(), 3);
    let mut b2 = HeuristicBot::new(Style::balanced(), 3);
    assert_eq!(b1.act(&View::new(&g, 0)), b2.act(&View::new(&g2, 0)));
    let mut o1 = vec![0u8; obs::OBS_LEN];
    let mut o2 = vec![0u8; obs::OBS_LEN];
    obs::encode_obs(&g, 0, &mut o1);
    obs::encode_obs(&g2, 0, &mut o2);
    assert_eq!(o1, o2);
}

#[test]
fn claim_priority_ron_beats_pong() {
    // Find a game state where someone can ron and someone else can pong the same discard,
    // by brute force over seeds with heuristic play, and check the winner.
    let mut found = false;
    for seed in 0..4000u64 {
        let mut g = Game::new(Config::default(), seed);
        let mut bots: [Box<dyn Bot>; 4] =
            std::array::from_fn(|i| Box::new(HeuristicBot::new(Style::archetypes()[i].clone(), seed)) as Box<dyn Bot>);
        while let Some(seat) = g.to_act() {
            if let Phase::Claim { .. } = g.phase {
                let ron: Vec<u8> = (0..4).filter(|&s| g.legal_actions(s).contains(&Action::Ron)).collect();
                let pong: Vec<u8> = (0..4).filter(|&s| g.legal_actions(s).contains(&Action::Pong)).collect();
                if !ron.is_empty() && pong.iter().any(|p| !ron.contains(p)) {
                    let w = ron[0];
                    // everyone answers: ron seats ron, pong seats pong
                    while let Some(s) = g.to_act() {
                        let a = if ron.contains(&s) { Action::Ron } else if pong.contains(&s) { Action::Pong } else { Action::Pass };
                        g.apply(s, a);
                    }
                    let r = g.result.as_ref().unwrap();
                    assert!(r.winner.is_some());
                    assert!(ron.contains(&r.winner.unwrap()));
                    let _ = w;
                    found = true;
                    break;
                }
            }
            let a = bots[seat as usize].act(&View::new(&g, seat));
            g.apply(seat, a);
        }
        if found {
            break;
        }
    }
    assert!(found, "no ron-vs-pong situation found in 4000 seeds");
}
