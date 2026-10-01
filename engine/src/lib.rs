//! da-san-yuan engine: Singapore 4-player mahjong rules, heuristic bots and
//! self-play data generation. See RULES.md for the encoded ruleset.

pub mod bots;
pub mod game;
pub mod obs;
pub mod rng;
pub mod scenario;
pub mod scoring;
pub mod selfplay;
pub mod shanten;
pub mod tile;

pub use game::{Action, Config, Discard, Game, HandResult, Phase, Player};
