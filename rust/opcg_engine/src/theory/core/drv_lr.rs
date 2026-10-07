//! 段 6（作成中）
use super::ev::R;
use super::game::Game;
use super::obj::V;
use super::rows::Cfg;
use super::state::Core;

pub fn game(_c: &mut Core, _g: &Game, _cfg: &Cfg, _p: &V) -> R<V> {
    Err("まだ移していない".into())
}
