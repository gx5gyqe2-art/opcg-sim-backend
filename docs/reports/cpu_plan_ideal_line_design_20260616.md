# 設計メモ: 自デッキ「理想ライン」自動導出プラン（J値統合）

- 日付: 2026-06-16
- 種別: 設計メモ（点・スナップショット。実装前の提案。実装後の正本は `docs/SPEC.md §2.5.5` へ吸収する）
- 対象: `cpu_self_plan.py`（`build_plan`/`PlanProfile`）／`cpu_ai.py`（`evaluate`/`_plan_progress`）
- 方針決定: **A（構成からのヒューリスティック自動導出）** ＋ **J値をプラン思考へ統合**

---

## 0. 要約

現状のプラン（§2.5.5）は自デッキを aggro/midrange/control に**3分類して評価重みを補正する静的バイアス**にとどまる。
本メモは、これを **「このデッキはどのターンに何を達成すべきか」の理想ライン（マイルストーン・スケジュール）** へ
拡張する設計を示す。理想ラインは**デッキ構成から自動導出**（人手の台本は書かない＝A）し、進捗の採点軸を
**J値（白＝デッキ残＋トラッシュ）の差分先行度**で表現する。相手リーダー由来の `OpponentProfile` を導出に
混ぜてマッチアップ補正する。手札的に理想手が打てない場合の「次善手」は**現行探索の性質で自動的に成立**する
（プランは手を強制せず評価にバイアスするだけ）ため、新規実装は不要。

フェア性・回帰の原則は現行踏襲: 参照は**自デッキ構成と公開情報のみ**、`plan=None` では一切作動せず
**現行挙動と完全同値**。

---

## 1. 背景と現状ギャップ

| 観点 | 現状（§2.5.5） | 本メモの拡張 |
|---|---|---|
| プランの正体 | アーキタイプ別の**重み乗数**（静的バイアス） | **ターン別の到達目標**（理想ライン） |
| 時間軸 | `clock_rate`（1ターン想定打点）の**連続値1本** | **J値差分の理想曲線**＝ターンごとの期待スケジュール |
| 進捗採点 | `_plan_progress`: 逆算リーサル＋（ライフ先行 or リソース差） | 同項を **J値差分の先行/遅延**へ一般化 |
| 相手対応 | `defense_factor`/`aggro_lean` を相手側評価へ反映 | **理想ラインの導出自体**を相手リーダーで補正 |
| フォールバック | （概念なし。探索が自然に次善を選ぶ） | 明示はしない＝探索の性質を**維持**するのが要件 |

現状の配管は既に整っている。`build_plan(masters, leader=None)` は `leader` 引数を**受けるが未使用**（将来の
リーダー別補正用の穴）、`plan` は `evaluate`→`_plan_progress` まで配線済みで「勝利状態からの逆算サブゴール」
という概念も実在する。本メモはこの穴にデータを流す増築であり、新概念の発明ではない。

---

## 2. J値を軸にした「理想ライン」の定義

### 2.1 なぜ J値で語るか
評価関数は既に J値理論ベース（黒リソースの重み付き和を主軸）。ただし `evaluate` は**明示の J値項を置かない**
（黒リソースと二重計上になるため）。一方で **J値そのものは盤面から実測できる**:

- 自分の J値 ≈ `len(self.deck) + len(self.trash)`（いずれも枚数は公開情報）
- 相手の J値 ≈ `len(opp.deck) + len(opp.trash)`（同上。中身は読まない＝フェア）

プランの進捗採点は「黒リソースの絶対量」ではなく**「J値差分が予定通り開いているか（スケジュール遵守度）」**
という直交した軸なので、評価本体（黒リソース和）との二重計上を避けつつ J値を自然に織り込める。
勝ち筋を J値で言い換えると:

> 自分の黒リソース（手札→場→打点）を効率よく回し、**相手の J値を押し上げ（ライフ削り＝相手 +1[J]、
> ブロッカー/カウンター強要＝相手の手札消費）**、自分の J値は温存する。理想ラインは
> **「ターン t までに (相手J値 − 自分J値) を Δ_t 以上に開く」** というサブゴール列。

### 2.2 理想ラインの自動導出（A）
`build_plan` で自デッキ構成（`masters`）と相手 `OpponentProfile` から、3つの曲線/値を機械生成する。
人手の台本は持たない。

1. **テンポ曲線 `tempo_curve[t]`（自分の黒展開）**
   コストヒストグラム＋ドン成長（≒ターン t で使用可能ドン = t）から、「ターン t に無駄なくドンを使い切る
   理想展開コスト総量」を期待値で出す。低コスト寄り構成ほど早ターンで立ち上がる曲線。

2. **Jクロック `j_clock`（相手J値の押し上げ率/ターン）**
   現行 `clock_rate`（ライフ近似）を **相手J値上昇率**へ一般化。アタッカー密度・平均打点・`aggro_lean`
   から「1ターンに相手J値を平均いくつ押し上げられるか（ライフ削り＋手札消費の期待値）」を出す。

3. **想定フィニッシュターン `finish_turn`**
   相手初期ライフと `j_clock` の交点。`lethal_mult` のマイルストーン（止めの形）を狙うべき目標ターン。

これらから理想スケジュール `delta_schedule[t] = 期待 (相手J値 − 自分J値)` を生成し、`_plan_progress` の
マイルストーン項を「実測 J値差分 − `delta_schedule[t]`」の先行/遅延で採点する（`milestone_mult` を流用）。

> いずれの係数も初期値は**要チューニング**。導出式は §5 に骨子を置く。

### 2.3 相手リーダー対応（A の範囲で）
`build_plan(masters, leader, opp_profile)` に `OpponentProfile`（テンプレ由来・§2.5.4）を渡し、
**理想ラインの導出自体を補正**する（質的な台本ではなく数値補正＝A）:

- 相手 `aggro_lean` 高（速い相手）→ `finish_turn` を前倒し／序盤に「ブロッカー確保」マイルストーンを挿入し
  `life_mult` 寄りへ。レース負け前に決める or 受けを固める。
- 相手 `blocker_ratio`/`removal_ratio` 高（重い・除去多）→ `finish_turn` を後ろ倒し、`tempo_curve` の
  横展開（複数体）を厚く評価。1体除去で崩れない盤面を志向。

参照は相手**リーダー紐付けのテンプレ構成のみ**（実手札・実デッキは読まない）＝現行のフェア原則を維持。

---

## 3. フォールバック（次善手）＝実装不要・現行性質の維持

プランは**評価にバイアスをかけるだけで手を強制しない**。探索は常に `get_legal_actions` の合法手を列挙し、
プランは「理想ラインに近づく手にご褒美」を与えるのみ。よって**手札的に理想手が打てない局面では、探索が
自然に“理想ライン目標スコアが最も高く残る別の手＝次善手”を選ぶ**。

→ 設計要件は「**台本を強制する分岐を入れない**」こと。理想ラインはあくまで `_plan_progress` の加点信号に
とどめ、特定手の合法手フィルタや強制選択は導入しない（現行のフォールバック性質を壊さないため）。

---

## 4. データ構造・配線（変更点）

### 4.1 `PlanProfile` 拡張（`cpu_self_plan.py`）
既存フィールドは不変。以下を**追加**（`plan=None`/`NEUTRAL` 時は全て無効化される値）:

- `delta_schedule: tuple[float, ...]`  … ターン別の理想 J値差分（`delta_schedule[min(t, len-1)]`）
- `j_clock: float`                     … 相手J値の想定上昇率/ターン（`clock_rate` の J値版・併存）
- `finish_turn: int`                   … 想定フィニッシュターン（`lethal_mult` 強調の起点）

`NEUTRAL` は `delta_schedule=()`／`j_clock=0`／`finish_turn=0` とし、**従来挙動と完全同値**を担保。

### 4.2 `build_plan` シグネチャ
`build_plan(masters, leader=None, opp_profile=None)` へ拡張（後方互換: 既存呼び出しは `opp_profile=None`
で従来通り）。`leader`/`opp_profile` 未指定時は §2.2 の自デッキ単独導出のみ。

### 4.3 `_plan_progress` 拡張（`cpu_ai.py`）
マイルストーン項を J値スケジュール採点へ一般化（`plan.delta_schedule` 空なら従来式へフォールバック）:

```
# 擬似コード（採点の骨子・係数は要チューニング）
if plan.delta_schedule:
    my_j  = len(me.deck)  + len(me.trash)
    opp_j = len(opp.deck) + len(opp.trash)
    target = plan.delta_schedule[min(manager.turn_count, len(plan.delta_schedule) - 1)]
    lead   = (opp_j - my_j) - target          # 予定より開いていれば +、遅れていれば −
    score += plan.milestone_mult * lead * _J_SCHED_W
else:
    # 従来: aggro=クロック先行 / control=リソース差 を aggro_lean でブレンド
    ...
```

逆算リーサル項は現行のまま（`discounted_reach` ベース）。`finish_turn` 近傍では `lethal_mult` の重みを
ターン接近に応じて強める補正を追加検討（過度な前のめりを避けるため弱めから）。

> 二重計上の注意: J値スケジュール項は「黒リソース絶対量」ではなく**予定差分からの乖離**を測る直交軸。
> ただし `W_DECK_DANGER`（自デッキ薄時の非線形ペナルティ）と符号干渉しうるので、`_J_SCHED_W` は小さめから
> 始め、デッキ枯渇局面では `delta_schedule` 採点を減衰させる（デッキを削る＝自J値低下が常に善ではない）。

### 4.4 API 配線（`app.py`）
`POST /api/game/create` で CPU の自デッキ＋（既存の）相手テンプレ `OpponentProfile` を `build_plan` に渡し
`CPU_GAMES[*].self_plan` に保持。`normal`/`hard` のみ・`easy` 非適用（現行と同じ）。

---

## 5. 導出式の骨子（初期案・要チューニング）

```
# 入力: masters（自デッキ）, opp_profile（相手テンプレ由来）
cost_hist  = ヒストグラム(cost)                       # 立ち上がりの速さ
atk_density = 攻撃的キャラ比率 × 平均打点              # 押し上げ力
aggro_lean = build_profile(masters).aggro_lean        # 既存集計の流用

# Jクロック: 1ターンの相手J値上昇期待（ライフ削り + 手札消費強要）
j_clock = base_push(atk_density) × (0.7 + 0.6×aggro_lean)
        × matchup(opp_profile.defense_factor, opp_profile.blocker_ratio)

# フィニッシュターン: 相手初期ライフ / Jクロック（マッチアップで前後）
finish_turn = round(opp_initial_life / max(j_clock, ε))
finish_turn += +1 if opp_profile.removal_ratio 高 else 0          # 重い相手は後ろ倒し
finish_turn += -1 if opp_profile.aggro_lean 高 else 0             # 速い相手は前倒し

# 理想 J値差分スケジュール（ターンごとの開くべき差）
delta_schedule[t] = clamp(j_clock × t − self_attrition(cost_hist, t), 0, …)
# self_attrition: 自分が攻めで失う黒/J（カウンター切り・相打ち）の期待。アグロほど大。
```

`base_push`/`matchup`/`self_attrition` の係数は、§7 の検証基盤（凍結ベースライン Elo／regret ログ）で
A/B しながら詰める。**まずは弱め（既存挙動からの逸脱小）に置き、回帰ゼロを確認してから強める**。

---

## 6. フェア性・回帰の担保

- 参照は **自デッキ構成 ＋ 公開情報（双方の deck 枚数・trash 枚数・場・ライフ枚数）のみ**。相手の実手札・
  実デッキの中身は読まない（`normal`）。`hard` は従来通り別途チート可だが本プランの導出は公開情報ベース。
- `plan=None` / `NEUTRAL`（`delta_schedule=()`）では新項は**一切作動せず現行挙動と完全同値**。既存の
  plan 単体テスト・挙動ベースライン（`full_card_audit.py`／`test_full_card_baseline.py`）不変を維持。
- `easy` 非適用（情報方針の3分化を維持）。

---

## 7. 段階導入とテスト計画

1. **Phase 1（最小）**: `PlanProfile` に `delta_schedule`/`j_clock`/`finish_turn` 追加（自デッキ単独導出）。
   `_plan_progress` のマイルストーンを J値スケジュール化。`opp_profile`/`leader` はまだ未使用。
   → ゲート: `OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s` 全 pass、`full_card_audit.py` 構造不変=0、
   ベースライン差分レビュー（`plan=None` 同値の確認）。挙動を意図的に変えたら `--regen`＋差分レビュー。
2. **Phase 2（相手補正）**: `build_plan(..., opp_profile)` でマッチアップ補正を導出へ注入。`app.py` 配線。
   → 検証基盤（凍結ベースライン Elo／regret ログ・`reports/cpu_precision_batch_20260616.md` の枠組み）で
   アリーナ A/B。`normal` の対 easy/対 hard 勝率と regret を計測し、退行が無いことを確認。
3. **Phase 3（チューニング）**: §5 係数を A/B で詰める。検証済みデッキの理想ライン挙動を
   `tests/test_verified_decks.py` にアサート追記。

---

## 8. 未解決の論点（実装着手前に確定したい）

- `delta_schedule` の長さ（何ターン分持つか）と、それ以降のクランプ方針。
- `_J_SCHED_W` の初期値と `W_DECK_DANGER`／`milestone_mult` との相互作用（過剰なデッキ削り回避）。
- `finish_turn` 接近時の `lethal_mult` 動的強調を入れるか（前のめり過ぎのリスク）。
- マッチアップ補正の強度（テンプレ精度に依存。弱めから）。
- 先攻/後攻でドン成長が1ずれる点を `tempo_curve` に織り込むか。

---

## 付録: 参照シンボル

- `cpu_self_plan.py`: `PlanProfile`/`build_plan`/`NEUTRAL`/`_PRESETS`/`_classify`
- `cpu_opponent_model.py`: `OpponentProfile`（`aggro_lean`/`defense_factor`/`blocker_ratio`/`removal_ratio`）
- `cpu_ai.py`: `evaluate(plan=...)`/`_plan_progress`/`_side_score`/`W_DECK_DANGER`/`clock_rate`/`milestone_mult`/`lethal_mult`
- 正本: `docs/SPEC.md §2.5.4`（相手モデル）／`§2.5.5`（勝ち筋プラン）／`§2.5.3`（精度向上バックログ）
