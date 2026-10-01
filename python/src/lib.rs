//! PyO3 bindings. Arrays cross the boundary as bytes; `dasanyuan/__init__.py`
//! wraps them as numpy arrays.

use dsy_engine::bots::{Bot, HeuristicBot, Style};
use dsy_engine::game::{Config, Game};
use dsy_engine::obs::{self, View, N_ACTIONS, OBS_LEN, ORACLE_LEN, SHANTEN_LEN, WAITS_LEN};
use dsy_engine::scenario::{self, Scenario};
use dsy_engine::selfplay::{self, SelfPlayConfig};
use dsy_engine::tile::hand_string;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::path::PathBuf;

/// One hand of mahjong as a turn-based multi-agent environment.
#[pyclass(module = "dasanyuan._core")]
struct Env {
    game: Game,
    bots: HashMap<(u8, String), HeuristicBot>,
    seed: u64,
}

#[pymethods]
impl Env {
    #[new]
    #[pyo3(signature = (seed=0, dealer=0, prevailing=0))]
    fn new(seed: u64, dealer: u8, prevailing: u8) -> Self {
        let cfg = Config { dealer, prevailing_wind: prevailing, ..Default::default() };
        Env { game: Game::new(cfg, seed), bots: HashMap::new(), seed }
    }

    /// Start a new hand.
    #[pyo3(signature = (seed, dealer=0, prevailing=0))]
    fn reset(&mut self, seed: u64, dealer: u8, prevailing: u8) {
        let cfg = Config { dealer, prevailing_wind: prevailing, ..Default::default() };
        self.game = Game::new(cfg, seed);
        self.seed = seed;
        self.bots.clear();
    }

    /// Seat that must act next, or -1 if the hand is over.
    fn to_act(&self) -> i32 {
        self.game.to_act().map_or(-1, |s| s as i32)
    }

    fn is_over(&self) -> bool {
        self.game.is_over()
    }

    fn obs_bytes<'py>(&self, py: Python<'py>, seat: u8) -> Bound<'py, PyBytes> {
        let mut buf = vec![0u8; OBS_LEN];
        obs::encode_obs(&self.game, seat, &mut buf);
        PyBytes::new_bound(py, &buf)
    }

    fn oracle_bytes<'py>(&self, py: Python<'py>, seat: u8) -> Bound<'py, PyBytes> {
        let mut buf = vec![0u8; ORACLE_LEN];
        obs::encode_oracle(&self.game, seat, &mut buf);
        PyBytes::new_bound(py, &buf)
    }

    /// Exact labels (hidden information): (waits [3*34] u8 tai, shanten [4] i8 as bytes).
    fn labels_bytes<'py>(&self, py: Python<'py>, seat: u8) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let mut w = vec![0u8; WAITS_LEN];
        let mut sh = vec![0i8; SHANTEN_LEN];
        obs::encode_labels(&self.game, seat, &mut w, &mut sh);
        let shb: Vec<u8> = sh.iter().map(|&x| x as u8).collect();
        (PyBytes::new_bound(py, &w), PyBytes::new_bound(py, &shb))
    }

    fn mask_bytes<'py>(&self, py: Python<'py>, seat: u8) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &obs::legal_mask(&self.game, seat))
    }

    /// Apply an action index (see ACTIONS in the Python package).
    fn step(&mut self, seat: u8, action: usize) -> PyResult<()> {
        let a = obs::action_from_index(action).ok_or_else(|| PyValueError::new_err("bad action index"))?;
        if self.game.to_act() != Some(seat) {
            return Err(PyValueError::new_err(format!("seat {} is not to act", seat)));
        }
        if !self.game.legal_actions(seat).contains(&a) {
            return Err(PyValueError::new_err(format!("illegal action {:?}", a)));
        }
        self.game.apply(seat, a);
        Ok(())
    }

    /// Action index a heuristic bot of `style` would choose for `seat` now.
    #[pyo3(signature = (seat, style="balanced"))]
    fn bot_action(&mut self, seat: u8, style: &str) -> PyResult<usize> {
        let key = (seat, style.to_string());
        if !self.bots.contains_key(&key) {
            let st = Style::by_name(style).ok_or_else(|| PyValueError::new_err(format!("unknown style {}", style)))?;
            self.bots.insert(key.clone(), HeuristicBot::new(st, self.seed ^ (seat as u64 + 1) * 7919));
        }
        let bot = self.bots.get_mut(&key).unwrap();
        Ok(obs::action_index(bot.act(&View::new(&self.game, seat))))
    }

    /// Points change for each seat (all zero until the hand ends or on a draw).
    fn deltas(&self) -> [i32; 4] {
        self.game.result.as_ref().map_or([0; 4], |r| r.deltas)
    }

    /// Result dict: winner, discarder, self_draw, tai, patterns, deltas.
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        if let Some(r) = &self.game.result {
            d.set_item("winner", r.winner.map(|w| w as i32))?;
            d.set_item("discarder", r.discarder.map(|w| w as i32))?;
            d.set_item("self_draw", r.self_draw)?;
            d.set_item("deltas", r.deltas.to_vec())?;
            if let Some(s) = &r.score {
                d.set_item("tai", s.tai)?;
                let pats: Vec<(String, u8)> = s.items.iter().map(|(p, t)| (p.name().to_string(), *t)).collect();
                d.set_item("patterns", pats)?;
            }
        }
        Ok(d)
    }

    /// Readable hand for debugging, e.g. "123m456p789s11z f1".
    fn hand_str(&self, seat: u8) -> String {
        let p = &self.game.players[seat as usize];
        let mut tiles = Vec::new();
        for k in 0..34u8 {
            for _ in 0..p.hand[k as usize] {
                tiles.push(k);
            }
        }
        tiles.extend(p.bonus.iter().copied());
        hand_string(&tiles)
    }

    fn draws_left(&self) -> usize {
        self.game.draws_left()
    }
}

/// The scenario bank: hand-built positions with checkable expectations.
#[pyclass(module = "dasanyuan._core")]
struct ScenarioBank {
    items: Vec<(String, Scenario)>,
}

#[pymethods]
impl ScenarioBank {
    #[new]
    fn new(dir: String) -> PyResult<Self> {
        let items = scenario::load_dir(std::path::Path::new(&dir)).map_err(PyValueError::new_err)?;
        for (f, sc) in &items {
            sc.validate().map_err(|e| PyValueError::new_err(format!("{}:{}: {}", f, sc.name, e)))?;
        }
        Ok(ScenarioBank { items })
    }
    fn __len__(&self) -> usize {
        self.items.len()
    }
    /// (file, name, description, tags, hero seat)
    fn info(&self, i: usize) -> PyResult<(String, String, String, Vec<String>, u8)> {
        let (f, s) = self.items.get(i).ok_or_else(|| PyValueError::new_err("index out of range"))?;
        Ok((f.clone(), s.name.clone(), s.desc.clone(), s.tags.clone(), s.hero))
    }
    /// A fresh Env positioned at scenario `i`; the hero is the seat to act.
    fn env(&self, i: usize) -> PyResult<Env> {
        let (_, s) = self.items.get(i).ok_or_else(|| PyValueError::new_err("index out of range"))?;
        let game = s.build().map_err(PyValueError::new_err)?;
        Ok(Env { game, bots: HashMap::new(), seed: i as u64 })
    }
    /// Check an action index for scenario `i`: (passed, message).
    fn check(&self, i: usize, action: usize) -> PyResult<(bool, String)> {
        let (_, s) = self.items.get(i).ok_or_else(|| PyValueError::new_err("index out of range"))?;
        let game = s.build().map_err(PyValueError::new_err)?;
        let a = obs::action_from_index(action).ok_or_else(|| PyValueError::new_err("bad action index"))?;
        Ok(match s.check(&game, a) {
            Ok(()) => (true, String::new()),
            Err(e) => (false, e),
        })
    }
}

#[pyfunction]
fn obs_len() -> usize {
    OBS_LEN
}
#[pyfunction]
fn n_actions() -> usize {
    N_ACTIONS
}
#[pyfunction]
fn oracle_len() -> usize {
    ORACLE_LEN
}

/// Generate self-play shards (runs on all cores, releases the GIL).
#[pyfunction]
#[pyo3(signature = (out_dir, hands=10000, seed=1, styles=vec!["fast".to_string(), "high_tai".to_string(), "defensive".to_string(), "balanced".to_string()], random_lineup=true, shard_size=2000, compress=true))]
fn run_selfplay(
    py: Python<'_>,
    out_dir: String,
    hands: u64,
    seed: u64,
    styles: Vec<String>,
    random_lineup: bool,
    shard_size: u64,
    compress: bool,
) -> PyResult<(u64, usize)> {
    let styles: Vec<Style> = styles
        .iter()
        .map(|s| Style::by_name(s).or_else(|| Style::from_csv(s)).ok_or_else(|| PyValueError::new_err(format!("bad style {}", s))))
        .collect::<PyResult<_>>()?;
    let cfg = SelfPlayConfig { hands, seed, styles, random_lineup, shard_size, out_dir: PathBuf::from(out_dir), compress };
    let s = py.allow_threads(|| selfplay::run_selfplay(&cfg)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((s.hands, s.decisions))
}

/// Duplicate tournament between 4 styles (names or CSV lines). Returns a list of dicts.
#[pyfunction]
#[pyo3(signature = (styles, walls=2000, seed=7))]
fn tournament<'py>(py: Python<'py>, styles: Vec<String>, walls: u64, seed: u64) -> PyResult<Vec<Bound<'py, PyDict>>> {
    if styles.len() != 4 {
        return Err(PyValueError::new_err("need exactly 4 styles"));
    }
    let st: Vec<Style> = styles
        .iter()
        .map(|s| Style::by_name(s).or_else(|| Style::from_csv(s)).ok_or_else(|| PyValueError::new_err(format!("bad style {}", s))))
        .collect::<PyResult<_>>()?;
    let arr: [Style; 4] = std::array::from_fn(|i| st[i].clone());
    let stats = py.allow_threads(|| selfplay::duplicate_tournament(&arr, walls, seed));
    stats
        .iter()
        .map(|s| {
            let d = PyDict::new_bound(py);
            d.set_item("name", &s.name)?;
            d.set_item("hands", s.hands)?;
            d.set_item("points_per_hand", s.mean())?;
            d.set_item("stderr", s.stderr())?;
            d.set_item("wins", s.wins)?;
            d.set_item("self_draws", s.self_draws)?;
            d.set_item("deal_ins", s.deal_ins)?;
            Ok(d)
        })
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<ScenarioBank>()?;
    m.add_function(wrap_pyfunction!(obs_len, m)?)?;
    m.add_function(wrap_pyfunction!(n_actions, m)?)?;
    m.add_function(wrap_pyfunction!(oracle_len, m)?)?;
    m.add_function(wrap_pyfunction!(run_selfplay, m)?)?;
    m.add_function(wrap_pyfunction!(tournament, m)?)?;
    Ok(())
}
