//! Self-play: run hands between bots, record every decision, and run
//! duplicate-format tournaments to compare styles fairly.

use crate::bots::{Bot, HeuristicBot, Style};
use crate::game::{Config, Game, HandResult};
use crate::obs::{self, View, N_ACTIONS, OBS_LEN, OBS_VERSION, ORACLE_LEN, SHANTEN_LEN, WAITS_LEN};
use crate::rng::Rng;
use crate::tile::*;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Config for a hand number within a session: dealer rotates every hand,
/// prevailing wind advances every 4 hands.
pub fn hand_config(hand_no: u64) -> Config {
    Config { dealer: (hand_no % 4) as u8, prevailing_wind: ((hand_no / 4) % 4) as u8, ..Default::default() }
}

pub fn hand_seed(master: u64, hand_no: u64) -> u64 {
    Rng::derive(master, hand_no).next_u64()
}

/// Play one hand to completion. `on_decision(game, seat, action)` is called before
/// each action is applied, which is where the logger snapshots the state.
pub fn play_hand<F: FnMut(&Game, u8, crate::game::Action)>(
    mut game: Game,
    bots: &mut [Box<dyn Bot>; 4],
    mut on_decision: F,
) -> (Game, HandResult) {
    let mut guard = 0;
    while let Some(seat) = game.to_act() {
        let action = bots[seat as usize].act(&View::new(&game, seat));
        on_decision(&game, seat, action);
        game.apply(seat, action);
        guard += 1;
        assert!(guard < 2000, "hand did not terminate");
    }
    let r = game.result.clone().expect("finished game has a result");
    (game, r)
}

// ----------------------------------------------------------------- logging

#[derive(Default)]
pub struct Buffers {
    pub obs: Vec<u8>,
    pub mask: Vec<u8>,
    pub action: Vec<u8>,
    pub oracle: Vec<u8>,
    /// exact per-opponent winning tiles (tai) [3 x 34]
    pub waits: Vec<u8>,
    /// shanten of every seat, relative [4]
    pub shanten: Vec<i8>,
    /// per decision: hand_id, seat, bot_id, points (filled after hand), won, dealt_in
    pub meta: Vec<i32>,
    pub rows: usize,
}

pub const META_COLS: usize = 6;

impl Buffers {
    fn record(&mut self, g: &Game, seat: u8, a: crate::game::Action, hand_id: i32, bot_id: i32) {
        let start = self.obs.len();
        self.obs.resize(start + OBS_LEN, 0);
        obs::encode_obs(g, seat, &mut self.obs[start..]);
        self.mask.extend_from_slice(&obs::legal_mask(g, seat));
        self.action.push(obs::action_index(a) as u8);
        let start = self.oracle.len();
        self.oracle.resize(start + ORACLE_LEN, 0);
        obs::encode_oracle(g, seat, &mut self.oracle[start..]);
        let ws = self.waits.len();
        self.waits.resize(ws + WAITS_LEN, 0);
        let ss = self.shanten.len();
        self.shanten.resize(ss + SHANTEN_LEN, 0);
        obs::encode_labels(g, seat, &mut self.waits[ws..], &mut self.shanten[ss..]);
        self.meta.extend_from_slice(&[hand_id, seat as i32, bot_id, 0, 0, 0]);
        self.rows += 1;
    }
    fn fill_outcome(&mut self, from_row: usize, r: &HandResult) {
        for row in from_row..self.rows {
            let m = &mut self.meta[row * META_COLS..(row + 1) * META_COLS];
            let seat = m[1] as usize;
            m[3] = r.deltas[seat];
            m[4] = (r.winner == Some(seat as u8)) as i32;
            m[5] = (r.discarder == Some(seat as u8)) as i32;
        }
    }
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_s = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shape_s);
    let total = 10 + dict.len() + 1;
    let pad = (64 - total % 64) % 64;
    dict.push_str(&" ".repeat(pad));
    dict.push('\n');
    let mut h = Vec::new();
    h.extend_from_slice(b"\x93NUMPY\x01\x00");
    h.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    h.extend_from_slice(dict.as_bytes());
    h
}

/// Write a .npy file, or a gzip-compressed .npy.gz when `compress` is set
/// (observations are mostly zeros and shrink roughly 15x).
fn write_npy(path: &Path, descr: &str, shape: &[usize], body: &[u8], compress: bool) -> std::io::Result<()> {
    let header = npy_header(descr, shape);
    if compress {
        let mut p = path.as_os_str().to_owned();
        p.push(".gz");
        let f = BufWriter::new(File::create(PathBuf::from(p))?);
        let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
        gz.write_all(&header)?;
        gz.write_all(body)?;
        gz.finish()?.flush()
    } else {
        let mut f = BufWriter::new(File::create(path)?);
        f.write_all(&header)?;
        f.write_all(body)?;
        f.flush()
    }
}

pub fn write_npy_u8(path: &Path, data: &[u8], shape: &[usize], compress: bool) -> std::io::Result<()> {
    write_npy(path, "|u1", shape, data, compress)
}

pub fn write_npy_i8(path: &Path, data: &[i8], shape: &[usize], compress: bool) -> std::io::Result<()> {
    let body: Vec<u8> = data.iter().map(|&v| v as u8).collect();
    write_npy(path, "|i1", shape, &body, compress)
}

pub fn write_npy_i32(path: &Path, data: &[i32], shape: &[usize], compress: bool) -> std::io::Result<()> {
    let body: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy(path, "<i4", shape, &body, compress)
}

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub hands: u64,
    pub seed: u64,
    pub styles: Vec<Style>,
    /// true: each hand samples 4 styles at random from `styles`; false: styles[0..4] fixed by seat.
    pub random_lineup: bool,
    pub shard_size: u64,
    pub out_dir: PathBuf,
    /// Write gzip-compressed .npy.gz files.
    pub compress: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ShardSummary {
    pub hands: u64,
    pub decisions: usize,
    pub wins: u64,
    pub draws: u64,
}

pub fn hand_csv_header() -> &'static str {
    "hand_id,seed,dealer,prevailing,bot0,bot1,bot2,bot3,winner,discarder,self_draw,tai,raw_tai,patterns,d0,d1,d2,d3,turns"
}

fn hand_csv_row(hand_id: u64, seed: u64, cfg: &Config, names: &[String; 4], r: &HandResult) -> String {
    let (tai, raw, pats) = match &r.score {
        Some(s) => (
            s.tai as i32,
            s.raw_tai as i32,
            s.items.iter().map(|(p, t)| format!("{}:{}", p.name(), t)).collect::<Vec<_>>().join("|"),
        ),
        None => (0, 0, String::new()),
    };
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},\"{}\",{},{},{},{},{}",
        hand_id,
        seed,
        cfg.dealer,
        cfg.prevailing_wind,
        names[0],
        names[1],
        names[2],
        names[3],
        r.winner.map_or(-1, |w| w as i32),
        r.discarder.map_or(-1, |w| w as i32),
        r.self_draw as i32,
        tai,
        raw,
        pats,
        r.deltas[0],
        r.deltas[1],
        r.deltas[2],
        r.deltas[3],
        r.turns
    )
}

/// Run self-play and write shards to `out_dir/shard_XXXXX/`:
/// obs.npy [N, OBS_LEN] u8, mask.npy [N, 76] u8, action.npy [N] u8,
/// oracle.npy [N, 102] u8, waits.npy [N, 3, 34] u8, shanten.npy [N, 4] i8,
/// meta.npy [N, 6] i32 (each .npy.gz when compressed), hands.csv.
pub fn run_selfplay(cfg: &SelfPlayConfig) -> std::io::Result<ShardSummary> {
    fs::create_dir_all(&cfg.out_dir)?;
    {
        let mut f = File::create(cfg.out_dir.join("bots.csv"))?;
        writeln!(f, "bot_id,name,speed,value,defence,claim,flush,pongs")?;
        for (i, s) in cfg.styles.iter().enumerate() {
            writeln!(f, "{},{}", i, s.to_csv())?;
        }
    }
    {
        let mut f = File::create(cfg.out_dir.join("format.txt"))?;
        writeln!(
            f,
            "obs_version={}\nobs_len={}\nn_actions={}\noracle_len={}\nwaits_shape=3,34\nshanten_len={}\nmeta_cols=hand_id,seat,bot_id,points,won,dealt_in",
            OBS_VERSION, OBS_LEN, N_ACTIONS, ORACLE_LEN, SHANTEN_LEN
        )?;
        writeln!(f, "seed={}\nhands={}\nrandom_lineup={}", cfg.seed, cfg.hands, cfg.random_lineup)?;
    }
    let n_shards = cfg.hands.div_ceil(cfg.shard_size);
    let summaries: Vec<std::io::Result<ShardSummary>> = (0..n_shards)
        .into_par_iter()
        .map(|shard| {
            let first = shard * cfg.shard_size;
            let last = ((shard + 1) * cfg.shard_size).min(cfg.hands);
            run_shard(cfg, shard, first, last)
        })
        .collect();
    let mut total = ShardSummary::default();
    for s in summaries {
        let s = s?;
        total.hands += s.hands;
        total.decisions += s.decisions;
        total.wins += s.wins;
        total.draws += s.draws;
    }
    Ok(total)
}

fn run_shard(cfg: &SelfPlayConfig, shard: u64, first: u64, last: u64) -> std::io::Result<ShardSummary> {
    let dir = cfg.out_dir.join(format!("shard_{:05}", shard));
    fs::create_dir_all(&dir)?;
    let mut buf = Buffers::default();
    let mut csv = BufWriter::new(File::create(dir.join("hands.csv"))?);
    writeln!(csv, "{}", hand_csv_header())?;
    let mut summary = ShardSummary::default();
    let mut lineup_rng = Rng::derive(cfg.seed ^ 0x5EED, shard);
    for hand_id in first..last {
        let seed = hand_seed(cfg.seed, hand_id);
        let gcfg = hand_config(hand_id);
        let ids: [usize; 4] = if cfg.random_lineup {
            std::array::from_fn(|_| lineup_rng.below(cfg.styles.len() as u64) as usize)
        } else {
            std::array::from_fn(|i| i % cfg.styles.len())
        };
        let mut bots: [Box<dyn Bot>; 4] = std::array::from_fn(|i| {
            Box::new(HeuristicBot::new(cfg.styles[ids[i]].clone(), seed ^ (i as u64 + 1) * 0x9E37)) as Box<dyn Bot>
        });
        let names: [String; 4] = std::array::from_fn(|i| cfg.styles[ids[i]].name.clone());
        let start_row = buf.rows;
        let game = Game::new(gcfg.clone(), seed);
        let (_, r) = play_hand(game, &mut bots, |g, seat, a| {
            buf.record(g, seat, a, hand_id as i32, ids[seat as usize] as i32)
        });
        buf.fill_outcome(start_row, &r);
        writeln!(csv, "{}", hand_csv_row(hand_id, seed, &gcfg, &names, &r))?;
        summary.hands += 1;
        if r.winner.is_some() {
            summary.wins += 1;
        } else {
            summary.draws += 1;
        }
    }
    csv.flush()?;
    let n = buf.rows;
    let z = cfg.compress;
    write_npy_u8(&dir.join("obs.npy"), &buf.obs, &[n, OBS_LEN], z)?;
    write_npy_u8(&dir.join("mask.npy"), &buf.mask, &[n, N_ACTIONS], z)?;
    write_npy_u8(&dir.join("action.npy"), &buf.action, &[n], z)?;
    write_npy_u8(&dir.join("oracle.npy"), &buf.oracle, &[n, ORACLE_LEN], z)?;
    write_npy_u8(&dir.join("waits.npy"), &buf.waits, &[n, 3, 34], z)?;
    write_npy_i8(&dir.join("shanten.npy"), &buf.shanten, &[n, SHANTEN_LEN], z)?;
    write_npy_i32(&dir.join("meta.npy"), &buf.meta, &[n, META_COLS], z)?;
    summary.decisions = n;
    Ok(summary)
}

// ------------------------------------------------------------- tournament

#[derive(Clone, Debug, Default)]
pub struct EntrantStats {
    pub name: String,
    pub hands: u64,
    pub points: i64,
    pub sum_sq: f64,
    pub wins: u64,
    pub self_draws: u64,
    pub deal_ins: u64,
    pub tai_sum: u64,
}

impl EntrantStats {
    pub fn mean(&self) -> f64 {
        self.points as f64 / self.hands.max(1) as f64
    }
    /// Standard error of the mean points per hand, estimated across wall-sets
    /// (each wall-set = the 4 seat rotations of one wall).
    pub fn stderr(&self) -> f64 {
        let w = (self.hands / 4).max(2) as f64;
        let m = self.mean();
        ((self.sum_sq / w - m * m).max(0.0) / (w - 1.0)).sqrt()
    }
}

/// Duplicate format: every wall is played 4 times with the entrants rotated
/// through all seats, so the luck of the deal cancels out.
/// Points are accumulated per wall-set (the 4 rotations) for the variance estimate.
pub fn duplicate_tournament(styles: &[Style; 4], walls: u64, seed: u64) -> Vec<EntrantStats> {
    let per_wall: Vec<[(i64, u64, u64, u64, u64); 4]> = (0..walls)
        .into_par_iter()
        .map(|w| {
            let mut acc = [(0i64, 0u64, 0u64, 0u64, 0u64); 4]; // points, wins, tsumo, deal-ins, tai
            let gseed = hand_seed(seed, w);
            let cfg = hand_config(w);
            let base = Game::new(cfg.clone(), gseed).wall;
            for rot in 0..4usize {
                // entrant e sits at seat (e + rot) % 4
                let mut bots: [Box<dyn Bot>; 4] = std::array::from_fn(|seat| {
                    let e = (seat + 4 - rot) % 4;
                    Box::new(HeuristicBot::new(styles[e].clone(), gseed ^ ((rot * 4 + seat) as u64 + 7))) as Box<dyn Bot>
                });
                let game = Game::from_wall(cfg.clone(), base.clone());
                let (_, r) = play_hand(game, &mut bots, |_, _, _| {});
                for seat in 0..4usize {
                    let e = (seat + 4 - rot) % 4;
                    acc[e].0 += r.deltas[seat] as i64;
                    if r.winner == Some(seat as u8) {
                        acc[e].1 += 1;
                        if r.self_draw {
                            acc[e].2 += 1;
                        }
                        acc[e].4 += r.score.as_ref().map_or(0, |s| s.tai as u64);
                    }
                    if r.discarder == Some(seat as u8) {
                        acc[e].3 += 1;
                    }
                }
            }
            acc
        })
        .collect();
    let mut stats: Vec<EntrantStats> =
        styles.iter().map(|s| EntrantStats { name: s.name.clone(), ..Default::default() }).collect();
    for acc in per_wall {
        for e in 0..4 {
            let st = &mut stats[e];
            st.hands += 4;
            st.points += acc[e].0;
            // variance over wall-sets, scaled to per-hand
            let per_hand = acc[e].0 as f64 / 4.0;
            st.sum_sq += per_hand * per_hand;
            st.wins += acc[e].1;
            st.self_draws += acc[e].2;
            st.deal_ins += acc[e].3;
            st.tai_sum += acc[e].4;
        }
    }
    stats
}

/// Quick sanity helper used by tests and the CLI `check` command.
pub fn verify_hand_invariants(g: &Game) -> Result<(), String> {
    let (drawn, held) = g.tile_census();
    if drawn != held {
        return Err(format!("tile census mismatch: drawn {} held {}", drawn, held));
    }
    if let Some(r) = &g.result {
        let s: i32 = r.deltas.iter().sum();
        if s != 0 {
            return Err(format!("points not zero-sum: {:?}", r.deltas));
        }
        if let Some(sc) = &r.score {
            if sc.tai < 1 || sc.tai > 5 {
                return Err(format!("tai out of range: {}", sc.tai));
            }
        }
    }
    for p in &g.players {
        for k in 0..NUM_KINDS {
            if p.hand[k] > 4 {
                return Err("more than 4 copies in a hand".into());
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ replay

/// Style ids seated at each seat for `hand_id` of a self-play run, exactly as
/// `run_selfplay` drew them (same shard RNG stream), so any hand in a dataset
/// can be replayed move for move.
pub fn lineup_for(cfg: &SelfPlayConfig, hand_id: u64) -> [usize; 4] {
    if !cfg.random_lineup {
        return std::array::from_fn(|i| i % cfg.styles.len());
    }
    let shard = hand_id / cfg.shard_size;
    let mut rng = Rng::derive(cfg.seed ^ 0x5EED, shard);
    let skip = (hand_id - shard * cfg.shard_size) * 4;
    for _ in 0..skip {
        rng.below(cfg.styles.len() as u64);
    }
    std::array::from_fn(|_| rng.below(cfg.styles.len() as u64) as usize)
}

/// The four bots for a dataset hand, seeded exactly as in `run_selfplay`.
pub fn bots_for(cfg: &SelfPlayConfig, hand_id: u64) -> ([usize; 4], [Box<dyn Bot>; 4]) {
    let ids = lineup_for(cfg, hand_id);
    let seed = hand_seed(cfg.seed, hand_id);
    let bots = std::array::from_fn(|i| {
        Box::new(HeuristicBot::new(cfg.styles[ids[i]].clone(), seed ^ (i as u64 + 1) * 0x9E37)) as Box<dyn Bot>
    });
    (ids, bots)
}
