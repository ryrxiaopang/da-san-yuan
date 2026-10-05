//! `dsy` command-line tool: self-play data generation, duplicate tournaments,
//! a simple evolutionary loop over bot styles, and a hand scorer for checking rules.

use clap::{Parser, Subcommand};
use dsy_engine::bots::Style;
use dsy_engine::rng::Rng;
use dsy_engine::scoring::{self, Meld, MeldKind, WinContext};
use dsy_engine::selfplay::{self, SelfPlayConfig};
use dsy_engine::tile::*;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

mod trace;

#[derive(Parser)]
#[command(name = "dsy", about = "da-san-yuan: Singapore mahjong engine tools")]
struct Cli {
    /// Worker threads (default: all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate self-play training data as .npy shards.
    Selfplay {
        /// Play this many complete games (East to North round, real dealer rules).
        /// Recommended: about 21 hands per game. Overrides --hands.
        #[arg(long, default_value_t = 0)]
        games: u64,
        /// Independent hands with the dealer passing every hand (used when --games is 0).
        #[arg(long, default_value_t = 10_000)]
        hands: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Comma-separated styles (fast, high_tai, defensive, balanced) or a styles CSV file.
        #[arg(long, default_value = "fast,high_tai,defensive,balanced")]
        styles: String,
        /// Keep styles fixed by seat instead of sampling a random lineup each hand (or game).
        #[arg(long)]
        fixed_lineup: bool,
        /// Hands per shard, or games per shard with --games (default 2,000 hands / 100 games).
        #[arg(long)]
        shard_size: Option<u64>,
        #[arg(long, default_value = "data/selfplay")]
        out: PathBuf,
        /// Write plain .npy instead of gzip-compressed .npy.gz (about 15x larger on disk).
        #[arg(long)]
        no_compress: bool,
    },
    /// Duplicate-format tournament between exactly 4 styles.
    Tournament {
        #[arg(long, default_value = "fast,high_tai,defensive,balanced")]
        styles: String,
        /// Number of walls; each is played 4 times with seats rotated.
        #[arg(long, default_value_t = 5_000)]
        walls: u64,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Evolve styles: each generation, the champion plays 3 mutated copies of itself.
    Evolve {
        #[arg(long, default_value_t = 20)]
        generations: u32,
        #[arg(long, default_value_t = 3_000)]
        walls: u64,
        #[arg(long, default_value_t = 0.25)]
        sigma: f32,
        #[arg(long, default_value_t = 11)]
        seed: u64,
        /// Starting style (or a CSV line from a previous run).
        #[arg(long, default_value = "balanced")]
        start: String,
        #[arg(long, default_value = "data/evolve.csv")]
        log: PathBuf,
    },
    /// Score a winning hand under the team ruleset.
    Score {
        /// Concealed tiles including the winning tile, plus bonus tiles, e.g. "123m456p789s11z55z f1"
        hand: String,
        /// Winning tile, e.g. 5z
        #[arg(long)]
        win: String,
        /// Exposed melds, e.g. "pong:5z,chow:123m,kong:1z,ckong:9p"
        #[arg(long, default_value = "")]
        melds: String,
        /// Seat wind 0=E 1=S 2=W 3=N
        #[arg(long, default_value_t = 0)]
        seat: u8,
        #[arg(long, default_value_t = 0)]
        prevailing: u8,
        #[arg(long)]
        tsumo: bool,
    },
    /// Run the scenario bank: validate positions and score bots on them.
    Scenarios {
        #[arg(long, default_value = "scenarios")]
        dir: PathBuf,
        /// Comma-separated bots to test (styles, "efficiency" or "random").
        #[arg(long, default_value = "random,efficiency,fast,high_tai,defensive,balanced")]
        bots: String,
        /// Print every scenario result, not just the summary.
        #[arg(long)]
        verbose: bool,
        /// Runs per bot with different bot seeds (random/noisy bots vary).
        #[arg(long, default_value_t = 20)]
        trials: u64,
    },
    /// Export complete hands from a self-play run as JSON for the replay viewer:
    /// every decision with the table state, legal moves, the move taken and the
    /// exact row recorded for training. Hands are reproduced move for move.
    Trace {
        /// Export every hand of these full games (1-based game numbers, comma-separated),
        /// from a run made with `selfplay --games`.
        #[arg(long)]
        game: Option<String>,
        /// Hand ids to export from an independent-hands run (0-based, as in hands.csv).
        #[arg(long, default_value = "0")]
        hands: String,
        /// Seed and shard size of the run to reproduce (defaults match `dsy selfplay`).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 2_000)]
        shard_size: u64,
        #[arg(long, default_value = "fast,high_tai,defensive,balanced")]
        styles: String,
        #[arg(long)]
        fixed_lineup: bool,
        #[arg(long, default_value = "replay.json")]
        out: PathBuf,
    },
    /// Expected points of every legal move at every decision of some full games (Monte Carlo rollouts).
    Evaluate {
        /// Games to evaluate, 1-based, e.g. "1-10" or "1,5,9".
        #[arg(long, default_value = "1")]
        games: String,
        /// Seed of the `selfplay --games` run to reproduce.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value = "fast,high_tai,defensive,balanced")]
        styles: String,
        #[arg(long)]
        fixed_lineup: bool,
        /// Sampled worlds per move; the error shrinks with the square root (64 is about +-1 point).
        #[arg(long, default_value_t = 64)]
        worlds: usize,
        /// Use the real hidden tiles instead of re-dealing what the player cannot see.
        #[arg(long)]
        oracle: bool,
        #[arg(long, default_value = "move_values.csv")]
        out: PathBuf,
    },
    /// Measure engine + bot throughput.
    Bench {
        #[arg(long, default_value_t = 2_000)]
        hands: u64,
    },
}

fn parse_styles(s: &str) -> Vec<Style> {
    let p = PathBuf::from(s);
    if p.exists() {
        let text = std::fs::read_to_string(&p).expect("read styles file");
        return text
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with("bot_id") && !l.starts_with('#'))
            .map(|l| {
                // accept either "name,..." or "id,name,..."
                Style::from_csv(l)
                    .or_else(|| Style::from_csv(l.split_once(',').map(|x| x.1).unwrap_or("")))
                    .unwrap_or_else(|| panic!("bad style line: {}", l))
            })
            .collect();
    }
    s.split(',')
        .map(|n| Style::by_name(n.trim()).or_else(|| Style::from_csv(n)).unwrap_or_else(|| panic!("unknown style {}", n)))
        .collect()
}

fn parse_melds(s: &str) -> Vec<Meld> {
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|m| {
            let (kind, tiles) = m.split_once(':').expect("meld must be kind:tiles");
            let t = parse_tiles(tiles).expect("bad meld tiles");
            let (kind, tile) = match kind {
                "chow" => (MeldKind::Chow, *t.iter().min().unwrap()),
                "pong" => (MeldKind::Pong, t[0]),
                "kong" => (MeldKind::ExposedKong, t[0]),
                "ckong" => (MeldKind::ConcealedKong, t[0]),
                other => panic!("unknown meld kind {}", other),
            };
            Meld { kind, tile, claimed: None, from: None }
        })
        .collect()
}

fn parse_game_list(s: &str) -> Vec<u64> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let (a, b): (u64, u64) = (a.trim().parse().expect("game number"), b.trim().parse().expect("game number"));
        v.extend(a..=b);
    }
    v
}

fn move_label(a: dsy_engine::Action) -> String {
    use dsy_engine::Action::*;
    match a {
        Discard(t) => format!("discard {}", tile_name(t)),
        Kong(t) => format!("kong {}", tile_name(t)),
        Tsumo => "self-drawn win".into(),
        Ron => "win on discard".into(),
        Pong => "pong".into(),
        ExposedKong => "kong from discard".into(),
        Chow(p) => format!("chow ({} of the run)", ["low", "middle", "high"][p as usize]),
        Pass => "pass".into(),
    }
}

/// Writes one CSV row per decision: the expected points of the move played, of the best move,
/// the regret between them, and the expected points of every legal move.
fn evaluate_games(games: &str, seed: u64, styles: &str, fixed_lineup: bool, worlds: usize, oracle: bool, out: &PathBuf) {
    use dsy_engine::evaluate::evaluate;
    use dsy_engine::obs::action_index;
    let cfg = SelfPlayConfig {
        hands: 0, seed, styles: parse_styles(styles), random_lineup: !fixed_lineup,
        shard_size: 100, out_dir: PathBuf::new(), compress: false, games: 0,
    };
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("create output"));
    writeln!(f, "game,hand_id,idx,seat,style,phase,n_legal,chosen_action,chosen_move,chosen_ev,best_action,best_move,best_ev,regret,regret_se,hand_points,all_moves").unwrap();
    let t0 = Instant::now();
    let mut n_dec = 0usize;
    for game_no in parse_game_list(games) {
        let ids = selfplay::game_lineup(&cfg, game_no - 1);
        let names: Vec<String> = ids.iter().map(|&i| cfg.styles[i].name.clone()).collect();
        let lineup: [Style; 4] = std::array::from_fn(|i| cfg.styles[ids[i]].clone());
        let rows = std::cell::RefCell::new(Vec::<(u64, u8, String)>::new());
        let idx = std::cell::Cell::new(0usize);
        selfplay::play_full_game(
            &cfg,
            game_no - 1,
            |g, seat, a, hand_id| {
                let phase = match g.phase {
                    dsy_engine::Phase::SelfTurn { .. } => "own_turn",
                    dsy_engine::Phase::Claim { .. } => "claim",
                    dsy_engine::Phase::RobKong { .. } => "rob_kong",
                    dsy_engine::Phase::Over => "over",
                };
                let vals = evaluate(g, seat, &lineup, a, worlds, seed ^ hand_id.wrapping_mul(0x9E37) ^ idx.get() as u64, oracle);
                let chosen = vals.iter().find(|v| v.action == a).unwrap();
                let best = vals.iter().max_by(|x, y| x.ev.partial_cmp(&y.ev).unwrap()).unwrap();
                let all: Vec<String> = vals.iter().map(|v| format!("{}:{:+.2}", move_label(v.action), v.ev)).collect();
                let line = format!(
                    "{},{},{},{},{},{},{},{},{},{:.3},{},{},{:.3},{:.3},{:.3}",
                    game_no, hand_id, idx.get(), seat, names[seat as usize], phase, vals.len(),
                    action_index(a), move_label(a), chosen.ev, action_index(best.action), move_label(best.action),
                    best.ev, best.ev - chosen.ev, best.se_vs_chosen,
                );
                rows.borrow_mut().push((hand_id, seat, format!("{},\"{}\"", line, all.join("; "))));
                idx.set(idx.get() + 1);
            },
            |_, _, _, _, r, _| {
                for (_, seat, line) in rows.borrow_mut().drain(..) {
                    let (head, tail) = line.split_once(",\"").unwrap();
                    writeln!(f, "{},{},\"{}", head, r.deltas[seat as usize], tail).unwrap();
                    n_dec += 1;
                }
                idx.set(0);
            },
        );
        eprintln!("game {}: {} decisions so far, {:.0}s", game_no, n_dec, t0.elapsed().as_secs_f64());
    }
    f.flush().unwrap();
    println!("wrote {} decisions to {} ({:.0}s)", n_dec, out.display(), t0.elapsed().as_secs_f64());
}

fn print_table(stats: &[selfplay::EntrantStats]) {
    println!("{:<14} {:>10} {:>8} {:>8} {:>8} {:>9} {:>8}", "style", "pts/hand", "±se", "win%", "tsumo%", "deal-in%", "avg tai");
    for s in stats {
        let h = s.hands.max(1) as f64;
        println!(
            "{:<14} {:>10.3} {:>8.3} {:>7.1}% {:>7.1}% {:>8.1}% {:>8.2}",
            s.name,
            s.mean(),
            s.stderr(),
            100.0 * s.wins as f64 / h,
            100.0 * s.self_draws as f64 / h,
            100.0 * s.deal_ins as f64 / h,
            s.tai_sum as f64 / s.wins.max(1) as f64
        );
    }
}

fn main() {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().unwrap();
    }
    dsy_engine::shanten::init_tables();
    match cli.cmd {
        Cmd::Selfplay { games, hands, seed, styles, fixed_lineup, shard_size, out, no_compress } => {
            let styles = parse_styles(&styles);
            let t = Instant::now();
            let shard_size = shard_size.unwrap_or(if games > 0 { 100 } else { 2_000 });
            let cfg = SelfPlayConfig { hands, seed, styles, random_lineup: !fixed_lineup, shard_size, out_dir: out.clone(), compress: !no_compress, games };
            let s = selfplay::run_selfplay(&cfg).expect("self-play failed");
            let secs = t.elapsed().as_secs_f64();
            if games > 0 {
                println!("{} full games, {:.1} hands per game on average.", games, s.hands as f64 / games as f64);
            }
            println!(
                "{} hands, {} decisions in {:.1}s ({:.0} hands/s). Wins {:.1}%, draws {:.1}%. Data in {}",
                s.hands,
                s.decisions,
                secs,
                s.hands as f64 / secs,
                100.0 * s.wins as f64 / s.hands as f64,
                100.0 * s.draws as f64 / s.hands as f64,
                out.display()
            );
        }
        Cmd::Tournament { styles, walls, seed } => {
            let st = parse_styles(&styles);
            assert_eq!(st.len(), 4, "tournament needs exactly 4 styles");
            let arr: [Style; 4] = std::array::from_fn(|i| st[i].clone());
            let t = Instant::now();
            let stats = selfplay::duplicate_tournament(&arr, walls, seed);
            println!("{} walls x 4 rotations = {} hands in {:.1}s", walls, walls * 4, t.elapsed().as_secs_f64());
            print_table(&stats);
        }
        Cmd::Evolve { generations, walls, sigma, seed, start, log } => {
            let mut champ = parse_styles(&start).remove(0);
            champ.name = "gen0".into();
            let mut rng = Rng::new(seed);
            if let Some(dir) = log.parent() {
                std::fs::create_dir_all(dir).ok();
            }
            let mut f = std::fs::File::create(&log).expect("create log");
            writeln!(f, "generation,name,speed,value,defence,claim,flush,pongs,pts_per_hand,stderr").unwrap();
            for gen in 1..=generations {
                let entrants: [Style; 4] = std::array::from_fn(|i| {
                    if i == 0 {
                        champ.clone()
                    } else {
                        champ.mutate(&mut rng, sigma, &format!("gen{}_m{}", gen, i))
                    }
                });
                let stats = selfplay::duplicate_tournament(&entrants, walls, seed.wrapping_add(gen as u64 * 1000));
                // Challenger must beat the champion by more than ~2 standard errors to take over.
                let champ_score = stats[0].mean();
                let (bi, best) = stats
                    .iter()
                    .enumerate()
                    .skip(1)
                    .max_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap())
                    .unwrap();
                let margin = 2.0 * (best.stderr().powi(2) + stats[0].stderr().powi(2)).sqrt();
                let promoted = best.mean() - champ_score > margin;
                println!(
                    "gen {:>3}: champion {:+.3}, best challenger {} {:+.3} (margin {:.3}) {}",
                    gen,
                    champ_score,
                    best.name,
                    best.mean(),
                    margin,
                    if promoted { "-> PROMOTED" } else { "" }
                );
                for (i, s) in stats.iter().enumerate() {
                    writeln!(f, "{},{},{:.4},{:.4}", gen, entrants[i].to_csv(), s.mean(), s.stderr()).unwrap();
                }
                if promoted {
                    champ = entrants[bi].clone();
                }
            }
            println!("final champion: {}", champ.to_csv());
        }
        Cmd::Score { hand, win, melds, seat, prevailing, tsumo } => {
            let tiles = parse_tiles(&hand).expect("bad hand");
            let bonus: Vec<Tile> = tiles.iter().copied().filter(|&t| is_bonus(t)).collect();
            let conc = counts_of(&tiles);
            let melds = parse_melds(&melds);
            let w = parse_tiles(&win).expect("bad win tile")[0];
            let ctx = WinContext { seat_wind: seat, prevailing_wind: prevailing, self_draw: tsumo, win_tile: w, ..Default::default() };
            match scoring::score_hand(&conc, &melds, &bonus, &ctx) {
                None => println!("Not a winning hand."),
                Some(s) => {
                    for (p, t) in &s.items {
                        println!("  {:<34} {} tai", p.name(), t);
                    }
                    println!("Total: {} tai (raw {}){}", s.tai, s.raw_tai, if s.tai < scoring::MIN_TAI { "  -- below minimum, cannot win" } else { "" });
                    if s.tai >= 1 {
                        if tsumo {
                            println!("Self-draw: each other player pays {}", 2 * scoring::unit_points(s.tai));
                        } else {
                            println!("Discard win: discarder pays {}", 4 * scoring::unit_points(s.tai));
                        }
                    }
                }
            }
        }
        Cmd::Scenarios { dir, bots, verbose, trials } => {
            let list = dsy_engine::scenario::load_dir(&dir).unwrap_or_else(|e| panic!("{}", e));
            let mut invalid = 0;
            for (file, sc) in &list {
                if let Err(e) = sc.validate() {
                    println!("INVALID {}:{} -> {}", file, sc.name, e);
                    invalid += 1;
                }
            }
            println!("{} scenarios loaded, {} invalid. Pass rate over {} runs per bot.", list.len(), invalid, trials);
            if invalid > 0 {
                std::process::exit(1);
            }
            let names: Vec<&str> = bots.split(',').map(|x| x.trim()).collect();
            println!("\n{:<30}{}", "scenario", names.iter().map(|n| format!("{:>11}", n)).collect::<String>());
            let mut totals = vec![0f64; names.len()];
            let mut by_tag: std::collections::BTreeMap<String, Vec<(f64, usize)>> = Default::default();
            for (_, sc) in &list {
                let mut row = format!("{:<30}", sc.name);
                let mut notes = Vec::new();
                for (bi, n) in names.iter().enumerate() {
                    let g = sc.build().unwrap();
                    let mut passes = 0;
                    let mut first_err = None;
                    for seed in 1..=trials {
                        let mut bot = dsy_engine::bots::make_bot(n, seed).unwrap_or_else(|| panic!("unknown bot {}", n));
                        let a = bot.act(&dsy_engine::obs::View::new(&g, sc.hero));
                        match sc.check(&g, a) {
                            Ok(()) => passes += 1,
                            Err(e) => {
                                first_err.get_or_insert(e);
                            }
                        }
                    }
                    let rate = passes as f64 / trials as f64;
                    totals[bi] += rate;
                    if let Some(e) = first_err {
                        notes.push(format!("    {}: {}", n, e));
                    }
                    for t in &sc.tags {
                        let e = by_tag.entry(t.clone()).or_insert_with(|| vec![(0.0, 0); names.len()]);
                        e[bi].0 += rate;
                        e[bi].1 += 1;
                    }
                    row.push_str(&format!("{:>10.0}%", 100.0 * rate));
                }
                println!("{}", row);
                if verbose {
                    for n in notes {
                        println!("{}", n);
                    }
                }
            }
            let n = list.len() as f64;
            println!("{:<30}{}", "OVERALL", totals.iter().map(|t| format!("{:>10.0}%", 100.0 * t / n)).collect::<String>());
            for (tag, v) in by_tag {
                println!("{:<30}{}", format!("  [{}]", tag), v.iter().map(|(a, b)| format!("{:>10.0}%", 100.0 * a / *b as f64)).collect::<String>());
            }
        }
        Cmd::Evaluate { games, seed, styles, fixed_lineup, worlds, oracle, out } => {
            evaluate_games(&games, seed, &styles, fixed_lineup, worlds, oracle, &out);
        }
        Cmd::Trace { game, hands, seed, shard_size, styles, fixed_lineup, out } => {
            let cfg = SelfPlayConfig {
                hands: 0,
                seed,
                styles: parse_styles(&styles),
                random_lineup: !fixed_lineup,
                shard_size,
                out_dir: PathBuf::new(),
                compress: false,
                games: 0,
            };
            let games: Vec<serde_json::Value> = match &game {
                Some(list) => list
                    .split(',')
                    .flat_map(|g| trace::trace_game(&cfg, g.trim().parse::<u64>().expect("game numbers") - 1))
                    .collect(),
                None => hands
                    .split(',')
                    .map(|h| trace::trace_hand(&cfg, h.trim().parse().expect("hand ids are numbers")))
                    .collect(),
            };
            let ids = &games;
            let doc = serde_json::json!({
                "format": 1,
                "obs_version": dsy_engine::obs::OBS_VERSION,
                "obs_len": dsy_engine::obs::OBS_LEN,
                "run": {"seed": seed, "shard_size": shard_size, "random_lineup": !fixed_lineup},
                "games": games,
            });
            std::fs::write(&out, serde_json::to_string(&doc).unwrap()).expect("write trace");
            println!("wrote {} hand(s) to {}", ids.len(), out.display());
        }
        Cmd::Bench { hands } => {
            let t = Instant::now();
            let stats = selfplay::duplicate_tournament(
                &std::array::from_fn(|i| Style::archetypes()[i].clone()),
                hands / 4,
                1,
            );
            let secs = t.elapsed().as_secs_f64();
            println!("{} hands in {:.2}s = {:.0} hands/s on {} threads", hands, secs, hands as f64 / secs, rayon::current_num_threads());
            print_table(&stats);
        }
    }
}
