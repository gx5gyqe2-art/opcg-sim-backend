# 計画: 実対局リプレイ検証（リプレイヤ＋ラウンドトリップ）

実アプリ CPU 対局（`cpu_trace=true`）の記録から対局を**再構築・再生**し、盤面・勝敗・CPU 思考トレースが
一致することを検証する仕組みを作る。既定 CPU が learned(Gen2) になった今、本番既定の実対局が seed から
丸ごと再現できることを機械保証するのが目的。**本書は計画（未実装）**。実装完了後に TEST_SPEC §3.2 へ吸収する。

## 1. 現状（調査済み・2026-07-04）

- **記録側は実装済み＋テスト済み**: `services/replay.py` が `cpu_trace` 対局のアクション（`_replay_record_action`）と
  CPU 思考トレース（`decisions`）をメモリに蓄積。`GET /api/game/{id}/replay` が記述子＋trace を返す。
  `test_api.py::test_replay_capture_and_fetch` が「撮れる・取り出せる」を固定。
- **再生側は不在**: 記述子（seed＋decks＋first_player＋actions）を食って対局を再現し一致を確かめる
  リプレイヤは**実装もテストも無い**。`cpu_replay.py --descriptor` は合成対局専用（seed/リーダー/難易度のみ・
  actions/decks を消費しない）。
- **記述子スキーマが2系統**: 合成＝`opcg-replay/v1`（`cpu_replay.make_descriptor`＝seed/leaders/difficulty）／
  実対局＝`REPLAY_SCHEMA`（`routers.game_replay`＝seed/first_player/difficulty/cpu_player_id/leaders/**decks**/**actions**）。
- **記録テストが hard 固定**: `test_replay_capture_and_fetch` は `cpu_difficulty="hard"`。**本番既定 learned は記録テスト未カバー**。

### 1.1 再現に必要な材料（記述子が持っているもの）

| 材料 | 記述子キー | 復元手段 |
|---|---|---|
| 乱数系列（シャッフル・コイントス・CPU探索） | `seed` | `random.seed(seed)`（learned も PR-D2 で global random 由来＝再現可） |
| 先攻 | `first_player` | `manager.start_game(first_player)` |
| デッキ（両者・card_id 列） | `decks` | `CardInstance(card_db.get_card(cid), owner)` の列で再構築 |
| 難易度 | `difficulty` | learned / hard を席へ結線（`game_driver.make_seat`） |
| 人間の操作列 | `actions`（`src:"human"`） | 順に適用（← ここが最難関・§3） |
| 期待値（照合対象） | `decisions`＋`_winner` | 再生結果と比較 |

## 2. 目標と検証の形（ラウンドトリップ）

### 2.0 主要ユースケース: 人間が気づいた異常局面の調査

**本計画の第一動機**。プレイヤーが CPU 対局中に「この手はおかしい」と感じたとき、その局面の CPU 思考を
再現して原因を確認できるようにする。前提として本番フロントは全 CPU 対局で `cpu_trace:true` を送っており
（`opcg-sim-frontend/src/api/client.ts`）、seed 固定＋思考トレースが記録され、UI の「ログ採取ボタン」で
`GET /api/game/{id}/replay`（記述子＋`decisions`）を取得できる。

現状の到達点と本計画で埋める差:

| 調査ステップ | 今できるか | 手段 / 埋める PR |
|---|---|---|
| 対局が記録される | ✅ できる | 本番 `cpu_trace:true` 常時（※揮発性＝採取ボタンで撮る必要・§6-6） |
| 「その手の候補と選定理由」を読む | ✅ できる（再生不要） | ライブ `decisions`（learned=MCTS訪問%/Q・L1第二意見／hard=候補/regret/J値成分） |
| **読み筋（read_ahead＝先読みPV）まで深掘り** | ❌ 今は不可 | ライブは軽量版で read_ahead 省略。**記述子から再生**して full トレース再取得＝R1 |
| **修正して同じ局面で再テスト／回帰ケース化** | ❌ 今は不可 | リプレイヤ（R1）＋ラウンドトリップ（R2/R3）。崩れ局面を `test_cpu_puzzles` へ落とす |
| 既定 learned(Gen2) の実対局記録の担保 | ⚠ 未検証 | 記録テストが hard 固定＝R3 で learned を追加 |

→ **R1 の設計要件に落ちること**: (a) 記述子から full トレース（read_ahead 込み）を再取得する
`--full` 相当の再生モードを持つ、(b) 採取ボタンで得た JSON をそのまま食える入力形式にする、
(c) 再生した崩れ局面を回帰ケース（決定論パズル）へ書き出せる。

### 2.1 検証の形（ラウンドトリップ）

```
録画: create(cpu_trace,seed) → 人間操作＋CPU手番を進める → 記述子D＋trace T を得る
再生: D から対局を再構築 → 人間操作は D.actions を注入・CPU手番は decide で再計算
検証: 勝敗・最終盤面・CPU 思考トレース（card_id 基準）が録画 T と一致する
```

- **CPU 手番は「記録の再適用」ではなく「再 decide」**する。同一 seed から再計算した手が録画と一致すること自体が
  決定論の証明＝検証の主眼（learned は PR-D2 で seed 再現、hard は既存の決定論）。
- **人間手番は D.actions を注入**する（seed から生成できないため）。
- 調査用途（§2.0）は「再生して full トレースを読む」＝一致検証の**副産物**。ラウンドトリップが通る＝
  再生が録画と同一局面を辿る保証があって初めて、再取得した read_ahead を信頼して読める。

## 3. 最難関: 人間アクションの再適用（ラベル→適用可能手）

`_replay_record_action` はアクションを `_describe_move` で **card_id/ラベル基準**（`{action_type, card, targets}`・
uuid 非依存）に落として記録する。再生時はこれを**再構築した局面の実 uuid を持つ合法手へ逆写像**する必要がある。

**課題:**
1. **同名カードの曖昧性**: 手札に同 card_id が複数あるとき、どの uuid かラベルだけでは一意に決まらない。
2. **effect 選択の payload 欠落**: `RESOLVE_EFFECT_SELECTION` 等の選択インデックス/対象は `_describe_move` が
   `card/targets` しか拾わず、選択の中身が lossy な可能性。
3. **target の逆写像**: `targets`（ラベル列）も同名複数で曖昧。

**対策の選択肢（設計判断が要る・§6 で確定）:**
- (A) **決定論タイブレークの逆引きリゾルバ**: 合法手を列挙し、`action_type`＋card_id（＋target card_id 集合）で
  マッチ。複数該当は「合法手列挙順の先頭」等の**文書化されたタイブレーク**で一意化。録画と再生で列挙順が同一
  （決定論）なら安全。まず effect 選択の無い単純手（PLAY/ATTACK/TURN_END/KEEP_HAND）で成立させる。
- (B) **記録の高解像度化**: `_replay_record_action` に**曖昧性解消の最小情報**（合法手 index か、選択 payload の
  card_id 化）を足す。記録フォーマットのバージョンを上げる（`REPLAY_SCHEMA` 更新＝契約影響・§5）。
- 推奨: **(A) を基本、(A) で一意化できない手種のみ (B) で最小限補強**。実装は「解ける手種」から始め、
  解けない手種は replay を stop してテスト対象外に明示（サイレント誤再生を出さない）。

## 4. 実装計画（PR 分割）

| PR | 内容 | ゲート |
|---|---|---|
| **R0（任意・先行）** | 記録の曖昧性を実測: 既存 traced 対局で「ラベルだけでは一意化できない人間手」の発生率を計測し、(A)/(B) の線引きを確定 | 計測レポート（`docs/reports/`） |
| **R1** | リプレイヤ中核 `replay_from_descriptor(descriptor) -> {winner, decisions, board}`: デッキ復元＋`start_game(first_player)`＋`random.seed`＋人間手注入（§3 リゾルバ）＋CPU 再 decide。合成用 `cpu_replay` と実対局用でループ本体は `game_driver.run_game` を共有（席＝learned/hard、人間席＝注入リゾルバ、observer＝trace 収集）。**§2.0 の調査用途**を満たす: (a) `--full`＝read_ahead 込みで再取得、(b) 採取ボタン JSON をそのまま入力、(c) 崩れ局面を決定論パズルへ書き出し | 単体（デッキ復元一致・単純手の注入が合法）＋採取 JSON の食い込み |
| **R2** | ラウンドトリップ・テスト（**hard**）: create(cpu_trace,seed) で短対局を録画→`replay_from_descriptor`→勝敗・decisions(card_id 基準)一致を assert。`tests/test_replay_roundtrip.py`（CI 内・有界 seed） | 一致 assert＋全テスト |
| **R3** | **learned** 対応＋ラウンドトリップ（learned）＋記録テスト（`test_replay_capture_and_fetch`）に learned ケース追加。PR-D2 の seed 再現が「実対局丸ごと再現」まで通ることを固定 | learned 一致 assert |
| **R4（任意）** | スキーマ統一: 合成 `opcg-replay/v1` と実対局 `REPLAY_SCHEMA` のリプレイヤ共通化（`cpu_replay --descriptor` が両方を再生）。契約更新は §5 | contract 再生成＋差分レビュー |

**実装順は R0→R1→R2→R3**。R2（hard）で骨組みを固めてから R3（learned）を乗せる。

## 5. 契約・記録フォーマットへの影響

- (A) のみで済めば **契約変更なし**（`decisions`/`actions` の読み取りのみ）。
- (B)（記録の高解像度化）を採る場合は `REPLAY_SCHEMA` を更新し、`shared_constants.json`/`schemas.py` に
  リプレイ関連キーがあるか確認 → `export_contract` 再生成＋`contract/` 同一コミット（CLAUDE.md の API 契約規約）。
  ※現状 difficulty は契約に無いが、actions/decisions のスキーマ化状況は R0/R1 着手時に確認する。

## 6. 未解決の設計判断（着手前に確定したい）

1. **人間手の曖昧性対策**: (A) 逆引き＋タイブレーク中心か、(B) 記録高解像度化をどこまで許すか（契約更新の是非）。
2. **effect 選択（`RESOLVE_EFFECT_SELECTION`）の再現粒度**: 現行記録で足りるか、選択 payload の card_id 化が要るか。
3. **不一致時の扱い**: 再 decide した CPU 手が録画と食い違ったら「テスト失敗（回帰）」か「記録が古い（許容）」か。
   → 原則テスト失敗（決定論契約の破れ）。ただし net 更新等で learned の手が変わる場合は録画を撮り直す運用にする。
4. **CI 負荷**: learned のラウンドトリップは MCTS で重い。低 sims＋短対局＋少 seed で有界化（決定論検証が目的で強さ無関係）。
5. **範囲**: まずは「単純手のみの短対局」を確実に再現。effect 対話の多い対局は R3 以降 or 対象外を明示。

## 7. 前提（既に満たされている土台）

- `game_driver.run_game`＋`make_seat`（設計⑥・PR-D3 で learned 席あり）＝再生ループは新規実装せず席と observer を差すだけ。
- learned の seed 再現（PR-D2）＝実対局丸ごと再現の必要条件は充足済み。
- 記録側（`services/replay.py`＋`/replay`）＝録画は既存。本計画は**再生側と検証**を足す。
