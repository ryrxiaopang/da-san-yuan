//! Per-move expected value by Monte Carlo rollouts.
//!
//! For one decision, every legal move is tried in the same set of sampled "worlds", and each
//! world is played to the end of the hand by the bots. A move's expected value (EV) is the
//! average final points of the player who made it. Regret = best move's EV - chosen move's EV.
//!
//! Fair mode (default) samples worlds the player could not tell apart from the real one: the
//! tiles they cannot see (opponents' concealed tiles and the wall) are reshuffled, keeping every
//! opponent's tile count. Oracle mode keeps the real hidden tiles and wall order, so it answers
//! "what would actually have happened", using information the player did not have.
//!
//! Every move is evaluated in the same worlds with the same bot seeds (common random numbers),
//! so differences between moves are measured far more precisely than each EV on its own.

use crate::bots::{Bot, HeuristicBot, Style};
use crate::game::{Action, Game};
use crate::rng::Rng;
use crate::selfplay::play_hand;
use crate::tile::{is_bonus, Tile, NUM_KINDS};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct MoveValue {
    pub action: Action,
    /// Mean final points of the acting seat over all worlds.
    pub ev: f64,
    /// Standard error of `ev`.
    pub se: f64,
    /// Standard error of (this move's EV - the chosen move's EV), using the paired worlds.
    pub se_vs_chosen: f64,
}

/// Re-deal everything `seat` cannot see, keeping what it can (its own hand, all melds,
/// discards, bonus tiles, every player's tile count and the number of tiles in the wall).
pub fn sample_world(g: &Game, seat: u8, rng: &mut Rng) -> Game {
    let mut w = g.clone();
    let mut pool: Vec<Tile> = Vec::new();
    let mut need = [0usize; 4];
    for o in 0..4u8 {
        if o == seat {
            continue;
        }
        let p = &mut w.players[o as usize];
        for k in 0..NUM_KINDS {
            for _ in 0..p.hand[k] {
                pool.push(k as Tile);
            }
            p.hand[k] = 0;
        }
        need[o as usize] = pool.len() - need.iter().sum::<usize>();
    }
    pool.extend_from_slice(&w.wall[w.front..w.back]);
    rng.shuffle(&mut pool);
    // Opponents' concealed hands never hold bonus tiles (they are set aside at once),
    // so hands take the first playing tiles; everything left becomes the unseen wall.
    let mut rest = Vec::with_capacity(pool.len());
    let mut it = pool.into_iter();
    for o in 0..4u8 {
        let mut n = need[o as usize];
        while n > 0 {
            let t = it.next().expect("enough tiles");
            if is_bonus(t) {
                rest.push(t);
            } else {
                w.players[o as usize].hand[t as usize] += 1;
                n -= 1;
            }
        }
    }
    rest.extend(it);
    let (front, back) = (w.front, w.back);
    w.wall[front..back].copy_from_slice(&rest);
    w.reopen_window(seat);
    w
}

/// Expected final points of `seat` for every legal move at this decision.
pub fn evaluate(g: &Game, seat: u8, styles: &[Style; 4], chosen: Action, worlds: usize, seed: u64, oracle: bool)
    -> Vec<MoveValue>
{
    let acts = g.legal_actions(seat);
    // results[a][k] = points for move a in world k
    let results: Vec<Vec<f64>> = acts
        .par_iter()
        .map(|&a| {
            (0..worlds)
                .map(|k| {
                    let mut rng = Rng::derive(seed, k as u64);
                    let mut world = if oracle { g.clone() } else { sample_world(g, seat, &mut rng) };
                    let mut bots: [Box<dyn Bot>; 4] = std::array::from_fn(|i| {
                        Box::new(HeuristicBot::new(styles[i].clone(), rng.next_u64() ^ i as u64)) as Box<dyn Bot>
                    });
                    world.apply(seat, a);
                    let (_, r) = play_hand(world, &mut bots, |_, _, _| {});
                    r.deltas[seat as usize] as f64
                })
                .collect()
        })
        .collect();
    let ci = acts.iter().position(|&a| a == chosen).expect("chosen move is legal");
    let n = worlds as f64;
    acts.iter()
        .zip(&results)
        .map(|(&a, xs)| {
            let mean = xs.iter().sum::<f64>() / n;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
            let diffs: Vec<f64> = xs.iter().zip(&results[ci]).map(|(x, c)| x - c).collect();
            let dm = diffs.iter().sum::<f64>() / n;
            let dvar = diffs.iter().map(|d| (d - dm).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
            MoveValue { action: a, ev: mean, se: (var / n).sqrt(), se_vs_chosen: (dvar / n).sqrt() }
        })
        .collect()
}
