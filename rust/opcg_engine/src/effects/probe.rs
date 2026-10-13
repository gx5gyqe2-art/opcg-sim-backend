//! 効果プローブ（`audit::effect_probe`）だけが立てる旗。
//!
//! 新カード監査（`tests/scripts/card_effect_audit.py`）は、汎用盤面で能力を発動して
//! 「本文の句が一つずつ実行経路に乗るか」を見る。汎用盤面ではリーダー特徴などの条件が
//! 満たせず、手札や場のカードも特徴を持たないフィラーなので、そのままでは条件付きの句や
//! 「特徴《X》を持つカードを捨てる」コストの先が実行されない。そこでプローブの間だけ:
//!
//! - **条件を固定する**（[`forced_condition`]）: `check_condition` が評価せずに真／偽を返す
//! - **対象の絞り込みを緩める**（[`relax_targets`]）: `get_target_cards` がカード種別だけで
//!   候補を選ぶ（特徴・名前・色・属性・コスト・パワー・レスト状態は見ない）
//!
//! どちらも thread-local で、[`ProbeGuard`] の drop で必ず下りる。対局・探索・golden は
//! 一度も立てない（既定は「旗なし」＝通常の評価）。

use std::cell::Cell;

thread_local! {
    static FORCE_CONDITION: Cell<Option<bool>> = const { Cell::new(None) };
    static RELAX_TARGETS: Cell<bool> = const { Cell::new(false) };
}

/// プローブ中なら条件の固定値、そうでなければ `None`。
pub fn forced_condition() -> Option<bool> {
    FORCE_CONDITION.with(Cell::get)
}

/// プローブ中で、対象の絞り込みを緩めるなら `true`。
pub fn relax_targets() -> bool {
    RELAX_TARGETS.with(Cell::get)
}

/// 旗を立て、drop で必ず下ろす（途中でエラーが返っても残らない）。
pub struct ProbeGuard;

impl ProbeGuard {
    pub fn set(force_condition: Option<bool>, relax: bool) -> Self {
        FORCE_CONDITION.with(|f| f.set(force_condition));
        RELAX_TARGETS.with(|f| f.set(relax));
        ProbeGuard
    }
}

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        FORCE_CONDITION.with(|f| f.set(None));
        RELAX_TARGETS.with(|f| f.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_guard_clears_both_flags_on_drop() {
        assert_eq!(forced_condition(), None);
        assert!(!relax_targets());
        {
            let _g = ProbeGuard::set(Some(true), true);
            assert_eq!(forced_condition(), Some(true));
            assert!(relax_targets());
        }
        assert_eq!(forced_condition(), None);
        assert!(!relax_targets());
    }
}
