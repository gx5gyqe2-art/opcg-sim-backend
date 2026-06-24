# 設計＋実装: 人間ログ活用パイプライン（検証セット＋ライブ採取）（2026-06-24）

> 計測/設計スナップショット（改変禁止）。天井上げの本丸＝**人間 vs CPU 実戦ログの活用**の第1段。
> Phase 3 の GBDT 計測で「非線形価値の天井はモデル容量でなく**データ量が律速**」と確定したのを受け、
> 実戦分布のデータを採る配管と、それを測る検証セットを用意する。

## 背景（なぜ人間ログか）

`cpu_phase3a_fair_value_relearn_20260623.md`＝fair 再学習で線形 val_acc 0.725、
`cpu_gbdt_nonlinear_value_20260624.md`＝GBDT は fair 2110 行では過学習で線形に未達。
→ 価値の器（2層分離＋GBDT 推論）は完成済みで、**律速はデータ**。大量 fair 自己対戦はこの環境では
長時間。**人間ログ＝実戦分布のカバー**が最有力。WBS 将来項から「本丸」へ格上げ済み。

## 設計判定: オフライン・リプレイは却下 → サーバ側ライブ採取

採取 replay descriptor（seed/decks/leaders/actions）から**オフライン再生**して各ターン境界の盤面を
再構築する案を検討したが**却下**。根拠（実コード）:

- 記録は `cpu_ai._describe_move` を通る＝**`{action_type, card(card_id), targets}` のみ**。
- 効果選択手（`RESOLVE_EFFECT_SELECTION`）の payload は `selected_uuids`/`accepted` を持つのに
  `_describe_move` は `uuid`/`target_ids` しか拾わない＝**効果選択の中身が記録時に落ちる**。
- → 効果カードを含む局はリプレイが**ロッシー**で忠実再現できず、特徴がズレて検証セットを汚染する。

代わりに **`collect_value_data` が既にやっている「生 GameManager のターン境界で特徴抽出＋勝者ラベル」
を実対局でもサーバ側で行う**＝生盤面から直接計算で**忠実性100%**・card_id→uuid 問題も無し。

## 実装

- `core/cpu_value_data.py`（新規）: ターン境界サンプラの**単一情報源**。
  `turn_boundary_samples(manager)`＝両者視点 `{"f","p"}`（プレイヤー**名**基準＝カスタム名追従・
  `manager.winner` も名前で整合）・読み取り専用。`label_samples(samples, winner)`＝勝者視点 y=1。
  自己対戦（`collect_value_data`）と実対局（`app.py`）が同一観測点・同一ラベル規約を共有。
- `api/app.py`: `_capture_value_samples` を3適用サイト後に挿入（human main / human battle / CPU loop）。
  **`cpu_trace` 時のみ**作動（既存リプレイ記録と同じ opt-in ゲート）＝未指定の本番対局には一切の
  オーバーヘッド・挙動変化なし。例外安全（hot path を壊さない）。replay endpoint が終局時に勝者で
  ラベル確定し `value_samples={f,y}` を descriptor に同梱。**フロント采取は replay 全体を運ぶので追加配線不要**。
  - 堅牢化（セルフレビュー）: 終局勝者を `meta["_winner"]` に保持＝WS 切断後 `delayed_cleanup` が
    GAMES から manager を退避しても replay でラベル確定でき、終了後采取のデータ消失を防ぐ。
- `tests/eval_value_on_set.py`（新規）= **検証セット(a)**: 学習済みモデルを**外部セット**で採点
  （acc/logloss/Brier/ECE＋定数予測参照線＋キャリブレーション表）。読み取り専用・`value_model.json` 不変。
  推論は `cpu_value_model.predict_winprob(features, model=...)`（後方互換で `model` 引数追加）を経由＝単一情報源。
- `tests/human_log_ingest.py`（新規）: 采取 JSON（エンベロープ階層を問わず）→ `value_samples` → JSONL。
- `tests/collect_value_data.py`: 共有サンプラへ置換（重複排除・挙動同値）。

## 実証

- 全経路スモーク: `collect_value_data`（リファクタ後）→ 采取エンベロープ模擬 → `human_log_ingest` →
  `eval_value_on_set`。同梱モデルを新規自己対戦ホールドアウト 82 行で採点＝
  **acc 0.634 / logloss 0.637（定数 0.693 を改善）/ ECE 0.141**＝「当てているが較正は不完全」を定量化。
  （この自己対戦セットは配管実証用のスタンドイン。人間采取ログが来たら同ハーネスで実戦汎化を測る。）
- ゲート: `test_value_capture.py`（サンプラ純粋性/非破壊/名前基準・未trace無影響・境界蓄積・
  replay ラベル確定・manager 退避サバイバル・取り込みの不正行除外）、`test_eval_value_on_set.py`
  （指標の既知値・明示モデル経路の同梱一致＝単一情報源・読み取り専用/決定論）。

## 次（このスナップショット時点で未実施）

- 実機デプロイ後、`cpu_trace=true` で人間 vs CPU を采取 → `human_log_ingest` で JSONL 化 →
  `eval_value_on_set` で **(a) 実戦汎化の本物指標**を取得。
- (b) 弱点採掘: 人間が逆転勝ちした局＝value-realization gap の現場抽出（采取 eventLog/value_samples から）。
- (A) 価値データ補完: 人間到達盤面を自己対戦データに重み付け混合して再学習＝GBDT の天井をデータで破る。
- 留保: 人間データは希少（自己対戦の補完）・棋力ばらつきは強局に絞る・天井化回避（最終的な人間超えは
  自己対戦が主役・AlphaZero は人間データを捨てた）。
