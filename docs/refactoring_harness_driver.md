# リファクタリング詳細設計⑥: CPU検証ハーネスの共通対局ドライバ化＋使い捨てスクリプト整理

CPU 強化フェーズでは思考トレース（`cpu_replay`）と強度・正しさの計器（arena / lethal / 単調性）を
高頻度で使い、新しい計器も増える。本設計は検証基盤の**拡張性**（新計器の追加コストを observer 1 個へ）と
**保守性**（エンジン変更への追従点を 7 箇所→1 箇所へ）を上げる。エンジン本体（`opcg_sim/`）には触れない。

## 1. 現状分析（実測・2026-07-04）

### 1.1 対局ループが 7 箇所に複製されている

「`start_game` → `winner is None` の間 `get_pending_request` → actor 解決 → decide →
`apply_battle_action`/`apply_game_action` → インバリアント検査」の同型ループ:

| 箇所 | 方策の形 | 観測 | 異常時 |
|---|---|---|---|
| `harness/cpu_selfplay.py::run_one_game` | 単一 policy（random/ai）＋事前 `get_legal_actions` | step 行 JSONL・trace_tail | `InvariantError` を raise |
| `harness/cpu_replay.py::run_replay` | 席別難易度・`decide_guarded(trace=)` | decision＋step 行 JSONL・`stop_after_decisions` | raise |
| `harness/cpu_arena.py::play_game` | 席別 decider（info_policy/CRN rng/pimc/budget/search/coeffs） | 勝敗サマリのみ | raise |
| `harness/cpu_arena.py::regret_trace` | MAIN_ACTION のみ `decide_with_regret` へ差替 | regret 集計 | break（検査なし） |
| `harness/cpu_arena.py::realize_trace` | 同上（`out=` で chosen_deep 採取） | value-realization gap 集計 | break |
| `scripts/lethal_audit.py::audit_game` | 席共通 decider | ターン頭の盤面リーサル記録 | None 返し（局を捨てる） |
| `scripts/lethal_regret.py::_gen_positions` | 席共通 | 意思決定点の局面収穫 | break |

エンジンのループ接点（pending request 種別・action_api の適用規約など）が変わると最大 7 箇所の追従が必要。
新計器を 1 つ作るたびにループを再実装しており、これが拡張の実コスト。

### 1.2 `cpu_selfplay.py` が事実上の共有ライブラリ

`_load_db`（プライベート命名のまま）・`build_deck`・`InvariantError`・`DEFAULT_MAX_STEPS` を
`cpu_replay` / `cpu_arena` / `arena_parallel` / `lethal_audit` / `lethal_regret` / `monotonicity_sweep` が
import。「自己対戦ランナー」という名のスクリプトがライブラリ責務を兼ねている。

### 1.3 決着済み調査の使い捨てスクリプトが残留

検証①（Gen2 解剖・`docs/reports/cpu_gen2_*_20260703.md`）は closure レポートで**完**。
その一回限りの調査スクリプト 8 本が `tests/scripts/` に残る（参照は凍結レポートのみ・
`_hunt_op16119.py` は参照ゼロ）。

### 1.4 触らないもの（対象外の明示）

- **`cpu_ai.decide(trace=...)` の opt-in トレース設計**は健全（`trace=None` の本番はオーバーヘッドゼロ・
  探索本体に不干渉）。ドライバはこれを**そのまま使う**。トレースのスキーマ `opcg-replay/v1` も不変。
- RL 系ループ（`opcg_game.OPCGGame` / `az_*` / `p3_*`）は別抽象（Game インターフェース）で既に統一済み＝対象外。
- pytest 用ヘルパ（`leader_test_helpers` / `engine_helpers`）は局面直組み用で対局ループを持たない＝対象外。

## 2. 設計

### 2.1 新モジュール `tests/harness/game_driver.py`

共有部品の移設（G-2）:

- `load_db()`（旧 `cpu_selfplay._load_db`・公開名へ改名）/ `build_deck` / `InvariantError` / `DEFAULT_MAX_STEPS`
- `cpu_selfplay` には同名の再エクスポートを残し既存 import を壊さない（G-4 で呼び元を書き換えて撤去）。

### 2.2 統一対局ループ `run_game`

```python
def run_game(
    seed, db, *,
    seats,                      # {"p1": decide_fn, "p2": decide_fn}
                                #   decide_fn(manager, actor, pending, moves) -> move
    deck_builder=build_deck,    # (db, owner_id) -> (leader, cards)。lethal_audit の実デッキ/全リーダー差替点
    observers=(),               # Observer の列（§2.3）
    max_steps=DEFAULT_MAX_STEPS,
    legal_moves="check",        # "check"=事前 get_legal_actions＋NO_LEGAL_MOVE 検査（selfplay/replay 系）
                                # "skip" =decide 内で解決（arena 系・既存の乱数消費順を保存するため）
    invariants="raise",         # "raise"=違反で InvariantError（selfplay/replay/play_game）
                                # "skip" =検査しない（regret/realize 系）
                                # "abort"=違反/例外で None を返し局を捨てる（lethal_audit 系）
    stop_after_decisions=None,  # cpu_replay の有界化
) -> Optional[GameResult]       # seed/winner/steps/turns/リーダー/decisions
```

席の生成は `make_seat(difficulty, *, info_policy=..., policy_rng=None, pimc_worlds=1,
budget=None, search=None, coeffs=None, trace_sink=None)` に一本化する
（`cpu_arena._make_decider` を昇格し、`cpu_selfplay._make_policy` の random/ai を吸収。
`trace_sink`（callable）指定時のみ `decide_guarded(trace=tr)` で採取して渡す）。

### 2.3 Observer プロトコル（観測専用・進行へ不干渉）

```python
class Observer:          # duck-typing。必要なコールバックだけ実装する
    def on_start(self, ctx): ...
    def on_decision_point(self, ctx): ...      # decide 前（ターン頭リーサル判定・局面収穫）
    def on_decision(self, ctx, trace): ...     # decide 後（思考トレース・decision 記録）
    def on_step(self, ctx, move, events): ...  # apply 後（盤面ステップ行）
    def on_end(self, ctx, result): ...
```

- `ctx` は `(manager, step, turn, phase, actor, pending)` の読み取り窓。**observer は manager を
  変更しない**（規約。決定論契約 §2.5 の前提）。
- **決定のインターセプトは observer でやらない**: `decide_with_regret` 系（regret/realize）は
  「MAIN_ACTION のみ差し替える seat（decider ラッパ）」＋集計 observer で実現する。
  観測=observer / 決定=seat の責務分離を守る。

### 2.4 既存ハーネスの写像（公開 API・出力スキーマは全て不変）

| 既存 | seats | observers | options |
|---|---|---|---|
| `run_one_game` | `make_seat`（random/ai） | StepEmitter(JSONL)＋TraceTail | `legal_moves="check"` |
| `run_replay` | `make_seat(trace_sink=…)`×席別難易度 | DecisionEmitter＋StepEmitter | `"check"`・`stop_after_decisions` |
| `play_game` | `make_seat`×席別フル指定 | なし | `legal_moves="skip"` |
| `regret_trace` / `realize_trace` | regret 差替 seat | Regret/RealizeCollector | `"skip"`・`invariants="skip"` |
| `audit_game` | `make_seat` | TurnHeadLethalObserver | `"skip"`・`invariants="abort"`・`deck_builder=実デッキ` |
| `_gen_positions` | `make_seat` | PositionHarvester | `"skip"`・`invariants="skip"` |

新計器の追加＝observer を 1 つ書いて `run_game` に渡すだけ。

### 2.5 決定論契約（最重要・破ってはならない）

同一 seed → 完全同一対局が本スイートの生命線（凍結ベースライン Elo・リプレイ種・挙動ベースラインが依存）。

1. 乱数は従来どおり **global `random`** に集約（`make_seat` の `policy_rng` 分離＝CRN は既存機能の移設のみ）。
2. **乱数消費点の順序を既存経路と一致**させる: `random.seed(seed)` → `deck_builder`×2 → `start_game` →
   （CRN rng 派生）→ 各 decide。`get_legal_actions` の呼出有無も既存に合わせる（`legal_moves` オプションが
   その保存装置。「乱数を消費しないはず」という仮定には依拠しない）。
3. トレース採取は `cpu_ai` 側の設計（trace 専用クローン・探索不干渉）に閉じる。
4. 破ったら気づけるよう、**等価性ゲート（§2.6）を移行より先に敷く**。

### 2.6 等価性ゲート（各移行 PR の合格条件）

移行前に旧実装の出力を採取（scratchpad・リポジトリ非コミット）し、移行後と比較する:

- `cpu_replay`: seed∈{7,11,23} × hard/normal で JSONL **バイト一致**（手記述は card_id 基準＝既存設計で安定）。
- `cpu_arena.play_game`: seed 0..19 の `(winner, steps, turns)` 一致（CRN あり/なし各系列）。
- `cpu_selfplay.run_one_game`: seed 0..9 のサマリ一致（policy=random/ai 両方）。
- `regret_trace` / `realize_trace`: seed 3 系列で集計 dict 完全一致。
- 既存 pytest 緑: `test_cpu_selfplay` / `test_cpu_replay` / `test_cpu_arena` / `test_cpu_puzzles` /
  `test_cpu_eval_v2` / `test_cpu_search_override` ほか全スイート。
- 恒久ゲートとして `tests/test_game_driver.py` を追加（同一 seed 2 回実行の再現一致＋observer 不干渉
  ＝observer 有無で結果不変。旧実装スナップショットには依存しない＝旧コード削除後も生きる形）。

### 2.7 スクリプト整理

**削除（G-1・検証①完につき）** — 参照は凍結レポートのみ・git 履歴で復元可:
`_dissect_gen2_alldecks.py` / `_dissect_gen2_blackbeard.py` / `_fairdet_probe.py` /
`_hunt_op16119.py`（参照ゼロ） / `_mine_gen2_selections.py` / `_verify_gen2_choices.py` /
`_reproduce_op16080_waste.py` / `_reproduce_op16119_lifeadd.py`

**統合（G-4）**: `depth_arena.py` / `thinktime_arena.py`（各 ~70 行・`arena_parallel.paired_play` の
同型ラッパ）→ `arena_parallel.py` の CLI オプション（`--challenger-search` / `--challenger-budget`）へ吸収。

**維持（現役）**: `bench_decide.py`（PyPy 回帰ベンチ・⑤E-6 ゲートでも使用）/ `_profile_decide.py` /
`_e2e_worker_smoke.py` / `p3_*` / `rl_*`（RL パイロット進行中）/ 診断・監査系
（`effect_diagnostics` / `coverage_report` / `text_execution_audit` / `sample_audit` /
`monotonicity_sweep` / `lethal_*` / `gate_b_diag` / `*_spec_probe` / `compare_parsers`）。

## 3. PR 分割と検証ゲート

| PR | 内容 | ゲート |
|---|---|---|
| G-1 | 使い捨て 8 本削除（コード変更なし・他と独立） | 全テスト green（削除物はテスト非参照を確認済み） |
| G-2 | `game_driver.py` 新設＝共有部品移設＋`cpu_selfplay` 再エクスポート | 全テスト**無変更で** green（互換の証明） |
| G-3 | `run_game`/Observer/`make_seat` 導入＋`cpu_replay`・`cpu_selfplay` 移行＋`test_game_driver.py` | §2.6（replay バイト一致・selfplay サマリ一致）＋全テスト |
| G-4 | `cpu_arena` 3 ループ＋`lethal_audit`/`lethal_regret` 移行、depth/thinktime 統合、再エクスポート撤去 | §2.6（arena/regret/realize 一致）＋全テスト＋文書追従 |

文書追従（G-4 内）: `TEST_SPEC.md` §3.1/§3.2 に driver 構成を追記、`docs/README.md` 索引更新。
`CLAUDE.md` の tests/ 配置規約は不変（`harness/` にモジュールが 1 つ増えるだけ）。

## 4. ロードマップとの関係

- ⑤E（tests/ 再編）の完了済みレイアウトの上に乗る。①〜④とファイル衝突なし（tests/ 側のみの変更）。
- 独立ワークストリームとしていつでも着手可。ただし **CPU 探索・make/unmake 系の機能開発とは
  `harness/cpu_*` を共有**するため、単一オーナー制（ロードマップ計画原則 1）に従い同時進行させない。

## 5. 期待効果

- 新計器の追加コスト = observer 1 個（対局ループの再実装ゼロ）。
- エンジンのループ接点変更時の追従が 7 箇所 → 1 箇所。
- `tests/scripts/` 8 本削除＋2 本統合、`cpu_arena` ~200 行・lethal 系 ~80 行の重複解消（正味 ~500 行減）。
- 思考トレース検証（CPU 強化の主目的)は `run_replay` の外形不変のまま、トレース粒度の拡張が
  observer / `trace_sink` の差し替えで済むようになる。
