# 計画書 — ルールモード CPU 対戦 ＆ 効果検証ハーネス

本書は、ルールモードに **CPU（AI）対戦機能** を追加し、同じ仕組みを **CPU 対 CPU の効果検証
ハーネス** としても使えるようにするための計画書である。実装着手前の合意・設計基盤として位置づける。

- 関連: システム仕様書 [`docs/SPEC.md`](SPEC.md)、テスト仕様書 [`docs/TEST_SPEC.md`](TEST_SPEC.md)、
  パーサ設計 [`docs/parser_v2.md`](parser_v2.md)、リーダー個別仕様 [`docs/leader_specs/`](leader_specs/README.md)。
- フロントエンド側の計画: `opcg-sim-frontend/docs/CPU_BATTLE_PLAN.md`。
- 開発ブランチ: 両リポジトリとも `claude/rule-mode-cpu-battle-g2z8vi`。

---

## 0. 目的とゴール

1. **遊ぶ機能**: ルールモードで人間（p1）が CPU（p2）と対戦できる。CPU は「先読み付きの強い AI」を目指す。
2. **効果検証ハーネス**: CPU 対 CPU を **決定論的・再現可能・自動異常検出付き** で実行し、その実行結果を
   追ってカード効果の正しさを検証できる。既存の監査基盤（`golden_cases` / `text_execution_audit` /
   `engine_helpers` / `get_debug_snapshot` / `action_events` / 構造化 `log_event`）と同じ規約・哲学に統合する。

この2つは独立に育てられる設計とする（弱い AI でも長時間の自己対戦で効果は踏めるため、
検証品質と AI の強さを分離する）。

---

## 1. 設計方針

**AI はバックエンドに置く。** ルールエンジン（`GameManager`）・効果解決・`pending_request` /
`active_interaction`・勝敗判定がすべてサーバ側にあり、先読み（状態を複製してシミュレート）するには
`GameManager` のロジックを直接使えるバックエンドが有利。フロントは「人間=p1」を操作し、CPU(p2) の手は
`/api/game/cpu/step` を **ポーリング** して 1 手ずつ受け取る（ステップ逐次の演出）。

対局形態のマトリクスに 3 つ目を追加する。

```
              ソロ            オンライン対戦        CPU対戦 (新規)
ルール    RealGame('both')  RuleLobby + ws       RealGame('p1', vsCpu) + /api/game/cpu/*
```

---

## 2. バックエンド設計

### 2.1 アクション適用ロジックの抽出（前提リファクタ）

現在 `/api/game/action`・`/api/game/battle`（`opcg_sim/api/app.py`）のエンドポイント内に
ディスパッチ処理が直書きされている。これを純粋関数へ切り出す。

- `apply_game_action(manager, player, action_type, payload) -> action_events`
- `apply_battle_action(manager, player, action_type, card_uuid) -> action_events`

→ 人間のエンドポイント・CPU ドライバ・自己対戦ランナーが **同一コードパス** を通る。
これがないと AI シミュレーション・自己対戦とルール本番の挙動が乖離するため最初に実施する。

### 2.2 状態複製・シミュレーション基盤（先読みの核）

- `GameManager.clone()`: `copy.deepcopy` ベース。WebSocket 等の非データ参照を持たないことを確認
  （持つ場合は除外）。`action_events` 等の一時状態はリセットする。
- **sim 専用の対話自動解決器** `auto_resolve_for_sim(manager, player)`: 先読み中に
  `active_interaction` / `pending_request` が立った場合に機械的に確定する（対象=ヒューリスティック
  最良 or 先頭、CONFIRM=自分有利なら使う、ARRANGE/COST は規定値）。既存 `_auto_resolve_replacement`
  を参考に拡張する。
- **隠れ情報の扱い（公平性）**: AI は相手手札・裏向きライフの中身を **見ない** 前提でクローンを
  「マスク」する（相手手札は枚数のみ、カウンターは確率モデル）。チート防止を設計原則として明記する。

> 重要な分離: `auto_resolve_for_sim` は **クローン上の先読み専用**。本番（実対局・自己対戦）の対話は
> AI の意思決定器が解決し、その選択を必ずログする（§4.5）。本番の未解決中断は握り潰さない。

### 2.3 評価関数（ヒューリスティック）

`evaluate(manager, me) -> float`。盤面優劣スコア。要素例:

- ライフ差（最重要）、盤面キャラの総パワー / 枚数差、手札枚数差、アクティブ DON 差、
  リーダー / キャラの KO 耐性・ブロッカー有無、相手リーダーへの打点期待値。
- カード固有の重みは将来調整可能なテーブルへ分離する。

### 2.4 探索（強い AI の本体）

完全なミニマックスは相手手札が隠れ＋効果分岐で爆発するため、実務的に
**「自分のターンのアクション列に対するビームサーチ＋浅い先読み」** を採用する。

1. **攻撃ターン**: 現在状態から合法手（プレイ / DON 付与 / 起動効果 / アタック / エンド）を列挙
   → 各手を `clone()` 上で適用 → `evaluate` → 上位 *k* 本をビーム保持 → エンドまで展開し、
   最良の **手順** を確定。
2. **防御判断（人間のアタック時）**: ブロッカー候補 / カウンター候補ごとに `resolve_attack` まで
   シミュレートし、被害が最小の選択を採用。
3. **対話解決（本番）**: `pending_request` が CPU 宛なら、対象選択も `evaluate` 比較で最良を選ぶ。
4. **マリガン**: 初期手札の評価（低コスト帯・キーカード有無）で keep / mulligan を判定。

合法手列挙は `get_legal_actions(player)` ヘルパーを新設する（手札の支払可能カード、アクティブな
攻撃者、有効な攻撃対象、起動可能効果）。生成手は `_validate_action` を通ることをテストで保証する。

### 2.5 難易度

| 難易度 | 思考ロジック |
|---|---|
| `easy`   | ランダム合法手 |
| `normal` | 貪欲 1 手（`evaluate` 最良の単手） |
| `hard`   | ビーム＋浅い先読み（§2.4） |

### 2.6 CPU 対局の配線とポーリング API

- **生成**: `POST /api/game/create` に `vs_cpu: true` / `cpu_difficulty` / `cpu_deck` を追加
  （または `/api/game/cpu/create`）。`GAMES[game_id]` に `cpu_player_id='p2'`・難易度をメタ保持する。
- **逐次ステップ**: `POST /api/game/cpu/step { game_id }`
  - 現在の本番状態で AI が「次の 1 手」を決定し、§2.1 の関数で適用して返す。
  - レスポンス = 通常の `build_game_result_hybrid` ＋ `{ cpu_acted, cpu_event, waiting_for }`。
    `waiting_for ∈ { 'human', 'cpu', 'human_decision', 'game_over' }`。
  - 「CPU が行動すべき状況」= CPU の手番 ／ `pending_request.player_id == cpu` ／
    `active_battle` が CPU の防御待ち。そうでなければ `cpu_acted=false, waiting_for='human'` を返し、
    フロントはポーリングを停止する。
  - ステートレス（毎回再計画）とし、desync に強くする。

---

## 3. フロントエンド設計（概要）

詳細は `opcg-sim-frontend/docs/CPU_BATTLE_PLAN.md`。要点のみ:

- **メニュー**（`ui/GameStart.tsx`）: ルールモード第 3 階層に「CPU 対戦」を追加（ソロ / オンライン → 3 択）。
  選択後に難易度セレクトと自分 / CPU のデッキ選択。
- **RealGame**（`screens/RealGame.tsx`）: 人間=p1 固定の新モード `vsCpu`。表示はオンラインと同等
  （自陣を下に固定・相手手札裏向き・手番ゲート）を再利用するが、通信は REST `/api/game/*` ＋ポーリング
  （WS 不使用）。
- **ポーリング駆動**（`game/actions.ts`）: 「CPU が動くべき状態」を検知したら `/api/game/cpu/step` を
  一定間隔で `waiting_for !== 'cpu'` になるまで呼び、各ステップの `action_events` をトースト / ログに反映して
  1 手ずつアニメ表示する。CPU 思考中は人間操作をロックする。
- **client / 型 / 配線**: `api/client.ts` に `createGame` の CPU オプションと `cpuStep(gameId)`、
  `api/types.ts` に `waiting_for` 等、`App.tsx` に `ruleCpu` 経路を追加。

---

## 4. 効果検証ハーネス（CPU 対 CPU）

CPU 対 CPU を「遊ぶ機能」と同時に **決定論的・自動異常検出付きの効果検証ツール** とする。
既存の whack-a-mole → burn-down 哲学に統合する。

### 4.1 決定論・再現性（必須要件）

- 全乱数を単一の seed 付き RNG に集約（シャッフル・AI のタイブレーク）。`--seed N` で完全再現。
- **リプレイ可能なアクションログ**: 適用した `(player, action_type, payload)` を順序付きで記録。
  同 seed ＋同ログで盤面が 1 ステップ単位で再現する。バグ報告 =「seed ＋ログ＋停止ステップ」で完結。

### 4.2 ヘッドレス自己対戦ランナー（既存 `tests/` 規約準拠）

`tests/cpu_selfplay.py`（`import conftest` ＋ `OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py ...` 形式）。

- オプション例: `--deck imu nami`（特定デッキ）/ `--leader OPxx`（特定リーダー）/
  `--games 100 --seed 0` / `--card OP11-041`（そのカードを強制投入して効果を踏ませる。
  `text_execution_audit --card` と同じ思想）/ `--max-turns N`。
- 出力 = **機械可読トレース（JSONL）**: 1 行 = 1 ステップ
  `{step, turn, phase, player, action, events, snapshot_diff, flags}`。`grep` / `diff` で異常箇所に直行できる。

### 4.3 対局中インバリアント／異常検出フック（核心）

各ステップ後に **実行時不変条件** をチェックし、破れたら **即停止＋完全リプロ出力**（fail-fast）。
`text_execution_audit` の実行時フラグを「ゲーム進行中」に常時作動させる。

- 既存思想の流用: `SUSPEND_LEAK`（手番をまたいで未解決の `active_interaction` /
  `pending_request` / temp_zone が残る）、`HIDDEN_LEAK`（隠しゾーンの中身露出）。
- 新規不変条件:
  - 場のキャラ ≤ 5（`FIELD_LIMIT`）
  - DON 総数保存
  - パワー非負
  - UUID ユニーク・ゾーン間の重複 / 消失なし
  - ライフ枚数とゾーンの整合
  - `get_legal_actions` が常に空でない（詰み / スタック検出）
  - 無限ループ検出（同状態反復・ステップ上限）

これにより「効果が静かに失敗する（`ActionType.OTHER` 等）」「中断が解決されない」を進行から自動炙り出しする。

### 4.4 期待挙動オラクルとの接続

- 可能な範囲で `golden_cases` / `leader_specs` を **期待仕様** として参照し、自己対戦で踏んだ効果の
  summary と突き合わせる。一致しない / 未検証はトレースに `anomaly` としてタグ付けする。
- 完全自動判定が難しい効果は「異常候補」として列挙するに留め、人 / AI レビュー前提とする（既存ツールと同運用）。

### 4.5 AI の自動解決がバグを隠さないこと（設計原則）

- §2.2 の `auto_resolve_for_sim` は **クローン上の先読み専用**。
- 本番（実対局・自己対戦）の対話は AI の意思決定器が解決し、その選択を **必ずログ** する。
- 本番状態の未解決中断・リークは握り潰さず §4.3 で必ず表面化する。
  ＝ CPU が「とりあえず動く」ことで効果バグを覆い隠す事態を構造的に防ぐ。

---

## 5. テスト・検証

- **CPU 対 CPU スモーク**: seed 固定で完走（クラッシュ / 無限ループ無し・必ず決着）。
- `clone()` が本番状態を破壊しない不変条件テスト。
- `get_legal_actions` の合法性（生成手が `_validate_action` を通る）テスト。
- 防御判断・マリガンの単体テスト。
- インバリアント検出フック自体のテスト（既知の破れを意図的に注入して検出されること）。
- 既存ゲート: `OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider`、
  `OPCG_LOG_SILENT=1 python tests/full_card_audit.py`。
- フロント: `npx tsc -b` / `npx eslint .` / `npx vite build`。

---

## 6. リスクと対策

| リスク | 対策 |
|---|---|
| `deepcopy` のコスト（探索で多数クローン） | ビーム幅・探索深さを難易度で制限。必要なら差分適用 / undo 方式へ最適化（段階的） |
| 効果の対話が先読みを複雑化 | sim 専用の自動解決器でデフォルト確定。複雑効果は保守的評価にフォールバック |
| 隠れ情報のチート懸念 | クローンを相手視点でマスク。AI は公開情報＋確率モデルのみ参照 |
| 無限ループ（AI がエンドしない / 対話ループ） | ステップ数上限・必ずエンドに収束するガード・タイムアウト |
| ルール挙動と sim の乖離 | §2.1 の共通コードパス徹底＋ CPU 対 CPU 完走テスト |
| AI が効果バグを覆い隠す | §4.5 の分離＋ §4.3 の fail-fast インバリアント |

---

## 7. 段階的進め方（PR 分割）

| PR | 内容 | 主な成果物 |
|---|---|---|
| **PR1: 基盤＋自己対戦ランナー** | §2.1 リファクタ・`clone()`・`get_legal_actions`・**決定論ランナー（§4.1/4.2）＋インバリアント検出（§4.3）** | これだけで効果検証ツールとして機能し、以降の回帰を即検出できる |
| **PR2: AI（backend）** | 評価関数（§2.3）・探索（§2.4）・難易度（§2.5）・`/api/game/cpu/step`・create フラグ（§2.6） | 強い CPU |
| **PR3: frontend** | メニュー・RealGame `vsCpu`・ポーリング駆動・client / 型（§3） | プレイ可能な CPU 対戦 |
| **PR4: 仕上げ** | 難易度チューニング・オラクル接続（§4.4）・`docs/SPEC.md`（両リポジトリ）と feature 分類の更新 | 完成 |

各 PR で §5 のゲートを通す。**PR1 に決定論的自己対戦ランナーとインバリアント検出まで含める**ことで、
強い AI を作る前に効果検証基盤を先に立ち上げる。
