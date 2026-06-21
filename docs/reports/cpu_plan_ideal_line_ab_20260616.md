# 計測報告: 理想ライン（J値スケジュール）A/B（Phase 3）

- 日付: 2026-06-16
- 種別: 報告（点・スナップショット）
- 対象: `cpu_self_plan.delta_schedule` / `cpu_ai._plan_progress`（理想ライン・J値スケジュール遵守度）
- 関連: 設計メモ `reports/cpu_plan_ideal_line_design_20260616.md`、仕様 `SPEC.md §2.5.5`

## 目的
Phase 1（自デッキ構成からの理想 J値差スケジュール自動導出）＋ Phase 2（相手リーダー由来のマッチアップ
傾き補正）の**純効果**を、検証基盤（`tests/cpu_arena.py`）の凍結ベースライン Elo で測る。

## 方法
- 同一シードのペア A/B。ON＝現状（`build_plan` が `delta_schedule` を導出）。OFF＝`_derive_delta_schedule`
  を `()` に差し替え＝Phase 0 相当（従来の手札＋場リソース差採点）。席を交互入替して先手有利を相殺。
- 2 種の計測:
  1. **vs easy**（`normal` 挑戦者 × 固定 easy）: 絶対強度の単調指標。ただし相手が貪欲 1-ply で弱く、
     中長期計画の価値が出にくい。
  2. **ON vs OFF 直接対決**（両者 `normal`・ON 方策 × OFF 方策）: 「スケジュールがそもそも良い方策か」を
     直接測る。
- 注: アリーナは自己対戦経路のため `opp_profile=None`＝**Phase 2 マッチアップ補正は不活性**。本計測が見るのは
  実質 Phase 1（自デッキ由来スケジュール）の効果。

## 結果

| 計測 | サンプル | OFF | ON | 差分 |
|---|---|---|---|---|
| normal vs easy | 24 局（seed0=100） | 14/24・wr0.583・Elo+58 | 13/24・wr0.542・Elo+29 | **−29 Elo**（ペア反転 3/24） |
| ON vs OFF 直接対決 | 20 局（seed0=200） | 9/20 | 11/20・wr0.550・Elo+35 | **+35 Elo** |

## 解釈
- いずれも **ノイズ域**（24/20 局では勝率標準偏差 ≈ ±0.10〜0.11。±30 Elo を 0 と区別できない）。
- 方向は vs easy はわずかにマイナス、直接対決はわずかにプラス。
- 構造監査・全テスト（981）pass、挙動ベースライン不変（プランは CPU 評価のみに作用しカード挙動は不変）。

## 現状
- 係数は `_J_SCHED_W=200`、傾き `_SCHED_SLOPE_*`、マッチアップ `_MATCHUP_*`。
  実装は理論整合（J値）・フェア（公開情報のみ）・`plan=None`/`delta_schedule` 空で現行完全同値、というガードを満たす。
- 係数チューニングに関する計測上の制約:
  - 数百局規模の Elo（アリーナは `normal` ≈ 1 手/秒で 1 局 ≈ 数十秒〜分。要バッチ/並列）。
  - もしくは決定単位の安価な指標（`regret` 拡張）で局所効果を切り分ける。
  - Phase 2 マッチアップ補正は対人テンプレ前提のため、アリーナ自己対戦では測れない。テンプレ相手を
    擬似配置する評価系が必要。

## 再現
使い捨て計測スクリプトはコミットしない（本報告に方法を記録）。要旨:
- vs easy A/B: `cpu_arena.arena("normal","easy",N)` を ON と「`_derive_delta_schedule→()`」の 2 アームで実行。
- 直接対決: 両者 `normal`・`decide_guarded(plan=on)` 対 `decide_guarded(plan=replace(on, delta_schedule=()))`、
  席ローテで ON 勝率を集計。`_J_SCHED_W` は環境変数で上書きしスイープ可能。
