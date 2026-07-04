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

**確定した方針（R0 実測に基づく・`docs/reports/cpu_replay_ambiguity_r0_20260704.md`）:**
R0 で曖昧率は実デッキで 3.5〜4.5%・fan-out 小（ほぼ 2手）・**effect 選択は 0/515（完全一意）** と判明。よって:
- **(A) 決定論タイブレーク逆引きを主とする**: 合法手を列挙し `action_type`＋card_id（＋target card_id 集合）で
  マッチ。複数該当は**合法手の決定論列挙順の先頭**を採る（録画・再生で列挙順が同一＝安全）。
- **手札複製（PLAY/SELECT_COUNTER）は挙動等価**（同一カード＝結果盤面同一）＝(A) で完全に解ける。
- **場複製（ATTACK/ATTACH_DON/SELECT_BLOCKER）の一部**だけが真の分岐リスク（付与ドン/パワー差）。これは
  **round-trip（R2/R3）で検出**し、残差が無視できなければその手種**だけ** (B)-lite（記録に個体弁別子を追加・
  `REPLAY_SCHEMA` 更新・§5）へ escalate する。effect 選択は現行記録のまま（0% 実測）。

## 4. 実装計画（PR 分割）

| PR | 内容 | ゲート |
|---|---|---|
| **R0 ✅完了** | 記録の曖昧性を実測（`replay_ambiguity_probe.py`・実デッキ）→ **曖昧率 3.5〜4.5%・fan-out 小・effect 選択は 0/515**。(A) 主で確定。詳細 `docs/reports/cpu_replay_ambiguity_r0_20260704.md` | ✅ 計測レポート |
| **R1 ✅実装済** | リプレイヤ中核 `replay_runner.replay_from_descriptor`: デッキ復元（`build_deck_from_ids`・pre-shuffle 順確認済）＋`random.seed`＋人間手注入（R0 確定の (A) 決定論タイブレーク逆引き `resolve_recorded_action`）＋CPU 再 decide。ループは `game_driver.run_game` 共有（人間席＝注入・CPU席＝learned/hard）。分岐は crash させず記録（`reproduced`/`misses`） | ✅ **実デッキ 10 seed で 10/10 完全一致**（勝敗+手数+ターン・人間手30〜50/局を tie-break で復元） |
| **R2 ✅実装済（hard）** | ラウンドトリップ・テスト `tests/test_replay_roundtrip.py`: 録画（人間=private rng・global random 非消費）→再生→勝敗・手数・ターン一致＋miss=0 を assert（実デッキ・有界 seed）。リゾルバ単体も検証 | ✅ hard で一致 assert（learned は R3） |
| **R1 副産物（engine 修正）** | `cpu_ai._find_card` が **stage/temp_zone** を探索せず、ACTIVATE_MAIN 等の手記述が card_id に解決できず uuid のまま漏れて再現不能だった欠落を修正（round-trip が検出）。修正で 8/10→**10/10** | ✅ trace テスト無退行（test_cpu_replay/learned 16 passed）・ruff clean |
| **R3 ✅実装済（コア）** | **learned** ラウンドトリップ（Gen2・実デッキで再現・sims 低で高速）＋**coin toss 再現**（`run_game(first_player=…)`＝実対局は CPU＝常に "random" を seed から再現・round-trip 3/3）＋API 記録テストに **learned ケース追加**（`test_replay_capture_learned`＝既定 Gen2 の記録担保）。`run_game` の first_player は既定 None で既存挙動不変（実測確認） | ✅ hard/learned/coin-toss 一致 assert＋API learned 記録 |
| **R3 実結線 ✅実装済** | API 記述子（`REPLAY_SCHEMA`）の **end-to-end 実結線**: routers create(cpu_trace,first_player=random)+step で実録画→`/replay`→`replay_from_descriptor(first_player="random")` で **CPU 意思決定列が録画と一致**。coin toss/デッキ復元/人間手注入/CPU 再 decide が実 API 記述子で整合。`cpu_player_id` が名前("P2")でも席("p2")でも解決（名前照合で人間手抽出） | ✅ `test_api::test_replay_api_descriptor_end_to_end` |
| **R3 残（少）** | **RESOLVE_EFFECT_SELECTION の記録欠落**（learned が踏む少数・§6-2）を (B)-lite（選択内容を記録に載せる）で閉じるか判断。effect 対話の多い実対局の完全再現に必要なら着手 | — |
| **R4（任意）** | スキーマ統一: 合成 `opcg-replay/v1` と実対局 `REPLAY_SCHEMA` のリプレイヤ共通化（`cpu_replay --descriptor` が両方を再生）。契約更新は §5 | contract 再生成＋差分レビュー |

**実装順は R0→R1→R2→R3**。R2（hard）で骨組みを固めてから R3（learned）を乗せる。

## 5. 契約・記録フォーマットへの影響

- (A) のみで済めば **契約変更なし**（`decisions`/`actions` の読み取りのみ）。
- (B)（記録の高解像度化）を採る場合は `REPLAY_SCHEMA` を更新し、`shared_constants.json`/`schemas.py` に
  リプレイ関連キーがあるか確認 → `export_contract` 再生成＋`contract/` 同一コミット（CLAUDE.md の API 契約規約）。
  ※現状 difficulty は契約に無いが、actions/decisions のスキーマ化状況は R0/R1 着手時に確認する。

## 6. 設計判断

1. ~~**人間手の曖昧性対策**~~ → **R0 で確定**: (A) 逆引き＋決定論タイブレーク主。曖昧率 3.5〜4.5%・fan-out 小。
   場複製の残差だけ round-trip で検出し必要時のみ (B)-lite（§3）。
2. **effect 選択の再現粒度（R0 → R3 で更新）**: R0（hard/random）では `RESOLVE_EFFECT_SELECTION` 0/515＝
   足りると見えたが、**R3 の learned 対局で稀に分岐**（learned は hard/random が踏まない effect 選択状態に
   到達し、選択内容が `_describe_move` に載らず bare `{RESOLVE_EFFECT_SELECTION}` で一意化できない）。
   → 大半は現行記録で足りるが、learned 完全再現には `RESOLVE_EFFECT_SELECTION` の選択内容を記録に載せる
   (B)-lite が要る（残作業・§R3 残）。round-trip がこの分岐を検出する（サイレント誤再生は出ない）。
3. **不一致時の扱い（要確認）**: 再 decide した CPU 手が録画と食い違ったら「テスト失敗（回帰）」か「記録が古い（許容）」か。
   → 原則テスト失敗（決定論契約の破れ）。ただし net 更新等で learned の手が変わる場合は録画を撮り直す運用にする。
4. **CI 負荷**: learned のラウンドトリップは MCTS で重い。低 sims＋短対局＋少 seed で有界化（決定論検証が目的で強さ無関係）。
5. **範囲**: まずは「単純手のみの短対局」を確実に再現。effect 対話の多い対局は R3 以降 or 対象外を明示。

## 7. 前提（既に満たされている土台）

- `game_driver.run_game`＋`make_seat`（設計⑥・PR-D3 で learned 席あり）＝再生ループは新規実装せず席と observer を差すだけ。
- learned の seed 再現（PR-D2）＝実対局丸ごと再現の必要条件は充足済み。
- 記録側（`services/replay.py`＋`/replay`）＝録画は既存。本計画は**再生側と検証**を足す。
