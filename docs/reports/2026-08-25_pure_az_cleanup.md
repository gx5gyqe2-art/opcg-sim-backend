# 純正AZ化: 弱いネット時代の補償層の削除（2026-08-25）

ユーザ決定（2026-08-25・CLAUDE.md「開発判断の前提」）: 純正 AlphaZero ループに向けた
コードベースの浄化として、弱いネットを読み出し側で補っていた層を**コード・設定・テストごと**
削除する。出荷品質の一時低下（coach_gate 基準値の低下等）は許容。

## 削除したもの

| 層 | 実体（削除） |
|---|---|
| プラン読み出し | `cpu_learned._plan_step`・decide のプランブロック・`SERVE_PLAN_READOUT`・`LearnedEngine(plan_readout=)`・`_turn_plans`/`_plan_sticky_id`・`tests/test_turn_plan.py`/`test_plan_box.py`/`test_plan_sticky.py` |
| 受け方針箱（P6-c） | `opcg_sim/src/learned/guard.py` 全体・`_guard_policy_for`・`SERVE_GUARD_POLICY`・`guard_policy=`・`tests/test_guard_policy.py` |
| 戦闘窓の読み出し/入口コミット | `_battle_window_choice`/`_battle_window_plan`/`_battle_commit_step`・`SERVE_BATTLE_READOUT`/`SERVE_BATTLE_COMMIT`・`battle_readout=`/`battle_commit=`・`_battle_plans`・`tests/test_battle_readout.py`/`test_battle_commit.py`（置き換え＝下記「窓の根畳み」） |
| root の LCB 乗り換え | `_select_root_group`（二重ゲート）・`SERVE_ROOT_SWITCH_MIN_FRAC/MIN_GAP`・`root_frac=`/`root_gap=`・`tests/test_learned_root_readout.py`。読み出しは**グループ合算後の argmax(N)** に簡約（`_merge_root_stats`＝行動の同一性は**残す**） |
| aux 粘り項 | `_aux_tie_scale`・`SERVE_AUX_TIEBREAK`/`AUX_TIE_DECAY`/`AUX_SAT_START`・`aux_tiebreak=`（`_value_fn` は素の predict に簡約）・`tests/test_learned_aux_tiebreak.py` |
| 終局値の深さ減衰 | `mcts.py` の TERM_DECAY/TERM_FLOOR/`_term_scale`（terminal は素の ±1）・`tests/test_mcts_terminal_decay.py` |
| ターン静止 | `SERVE_TURN_QUIESCE`・`turn_quiesce=`/`turn_value_fn=` の TreeMCTS/LearnedEngine 配線（`resolve_turn_inplace` は計器用に mcts.py へ残す・`TURN_QUIESCE_MAX_PLIES` は mcts.py へ移設）・`tests/test_mcts_turn_quiesce.py` |
| 旧ドン箱 | `SERVE_DON_BOX`・adapter の don_box 合成ブロック・`don_box=`・`tests/test_don_box.py`（`don_box_candidates` 関数と `don_box_first_primitive` は残置＝後者はアタック箱の実対局出力が使用。アタック箱 P2 が上位互換） |
| 計器フラグ | `arena_resume` の `--cand-box/--cand-tree-box/--cand-don-margin/--cand-don-box/--cand-plan-readout/--cand-plan-box/--cand-guard-policy`・`coach_gate --chall-boxes` の kwargs 縮小・`candidate_screen` の commit 表示・`search_averse_probe` のアブレーション腕（argmaxN/減衰off/auxoff） |

config の PLAN_*（PLAN_WORLDS/PLAN_PROPOSALS/PLAN_TEMP/PLAN_MIN_SPREAD/PLAN_STRUCT_*）は
`plan.py` 内へ移設。`plan.py` 自体は**計器専用**（`plan_dom_gen`/`plan_lethal_gen`/`plan_cf2_gen`
等の教師/計器が import）として残す。

## 残したもの（不変）

箱化一式（`SERVE_MACRO_MOVES`/`SERVE_DEFENSE_BOX`/`TREE_BOX_BATTLE`/`TREE_BOX_DIALOG`・
各候補合成・`defense_box_prune`）・静止探索（`SERVE_QUIESCE`）・`_merge_root_stats`・
PIMC/CRN/sticky world・`BOX_BRANCH_BUDGET`/`BOX_RESOLVE_DEPTH`・prune_futile/don_margin・
exit ヘッド機構（`predict_exit`/`_exit_value_fn`/`_battle_value_fn`）・N0 スパイク
（`tests/scripts/n0_spike.py`）・教師/計器スクリプト群。

## 「窓の根畳み」統一の意味論

decide に統一の高速経路 `LearnedEngine._window_choice` を1つ実装した:
**窓**（in_battle、または box_dialog 有効時の in_dialog）では、決定化1世界の合法手を
`resolved_branch_values`（戦闘窓＝`_battle_value_fn()`・対話窓＝本体 value・
window_pred=in_dialog）で採点し argmax の1手を返す。

これは木の `_expand` の箱畳み（TREE_BOX_BATTLE/TREE_BOX_DIALOG）が root ノードで行う計算と
**同一の意味論**である: 畳まれた root は単一辺になり訪問を配る意味が無いため、探索を回さず
直接その1手を返す＝**探索の迂回ではなく木の root 畳みの高速版**。旧
`_battle_window_choice`/`_dialog_window_choice`/入口コミットはこの1関数に置き換え、
トレースの readout ラベルは "window_resolved" に統一した。

## ロールバック

git 履歴を参照（本コミット直前＝origin/claude/cpu-spec-improvements-yw91jd の 612da0e）。
削除した各層は当時の config フラグ込みで履歴から復元できる。
