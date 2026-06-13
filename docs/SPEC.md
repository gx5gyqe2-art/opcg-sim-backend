# システム仕様書 — opcg-sim-backend

本書は `opcg-sim-backend`（FastAPI + 独自ルールエンジン）の**システム仕様書**である。
対になる **テスト仕様書** は [`docs/TEST_SPEC.md`](TEST_SPEC.md)、カード効果パーサの設計詳細は
[`docs/parser_v2.md`](parser_v2.md)、リーダー個別仕様は [`docs/leader_specs/`](leader_specs/README.md)。
フロントエンドの仕様は `opcg-sim-frontend/docs/SPEC.md`。

---

## 0. 全体アーキテクチャ

```
[フロントエンド React/Pixi] ── REST(/api/...) ──┐
                            └─ WebSocket(/ws/...) ┤
                                                  ▼
                         [FastAPI: opcg_sim/api/app.py]
        ┌───────────────────────────────┬───────────────────────────────┐
        ▼                               ▼                               ▼
  ルールモード(GameManager)      フリーモード(SandboxManager)      デッキ/カードDB
  公式ルール自動進行              手動操作(ルール強制なし)          (Firestore / JSON)
  GAMES[game_id]                  SANDBOX_GAMES[game_id]
  + RULE_ROOMS(オンライン対戦)    + ルーム/WS(/ws/sandbox)
```

本システムは2つの対局モードを持つ。

| モード | エンジン | エンドポイント | 特徴 |
|---|---|---|---|
| **ルールモード** | `GameManager`（`core/gamestate.py`） | `/api/game/*`（REST）＋ `/api/rule/*` ＋ `/api/game/cpu/step` ＋ `/ws/game/{id}` | 公式ルールを自動進行。ソロ（ホットシート）／オンライン対戦／**CPU 対戦**に対応 |
| **フリーモード** | `SandboxManager`（`core/sandbox.py`） | `/api/sandbox/*` ＋ `/ws/sandbox/{id}` | ルール強制なしの自由操作。ソロ／オンライン対戦に対応 |

カード効果は `GameManager`（ルールモード）でのみ解決される。フリーモードは盤面操作のみ。

ルールモードのアクション適用ロジックは `core/action_api.py`（`apply_game_action`/`apply_battle_action`）に
集約され、HTTP エンドポイント・CPU 対戦ドライバ・自己対戦ランナーが**同一コアパス**を通る。
CPU（AI）・効果検証ハーネスの詳細は [`docs/CPU_BATTLE_PLAN.md`](CPU_BATTLE_PLAN.md)。

---

## 1. コアゲームルール仕様（ルールモード）

`GameManager`（`opcg_sim/src/core/gamestate.py`）が公式ワンピースカードゲームのルールに沿って
進行する。公式ルールに準拠する主要項目を以下に定める（実装箇所は file:line で示す）。

### 1.1 ターン構造
`switch_turn → refresh_phase → draw_phase → don_phase → main_phase → end_turn` の順で進む（`gamestate.py` 各 `*_phase`）。

- **リフレッシュ**: 自分のキャラ/リーダー/ステージをアクティブ化し、レストドンと付与ドンをアクティブのドンプールへ戻す（`refresh_all`）。`FREEZE` フラグ持ちはレスト維持。
- **ドロー**: `turn_count > 1` のときのみ1枚ドロー。**先攻1ターン目はドローしない**（公式準拠、`draw_phase`）。
- **ドン!!**: `turn_count == 1` なら1枚、以降は2枚をアクティブで追加。**先攻1ターン目は1枚**（公式準拠、`don_phase`）。ドン!!デッキ上限は**10枚**（`Player.__init__`）。
- **エンド**: ターン終了時誘発（`TURN_END`／非ターンプレイヤーの `OPP_TURN_END`）と遅延効果（`pending_end_of_turn`）を解決し、手番交代（`end_turn`／`_fire_turn_end_triggers`）。

### 1.2 ゲーム開始・マリガン・リソース
- **先攻決定**: `start_game(first_player)`。未指定なら p1 先攻（オンライン対戦ではホスト=p1=先攻）。
- **ライフ初期化**: リーダーの `master.life` 枚をデッキ上から配置（`place_life`）。初期手札は**5枚**（`draw_initial_hand`）。
- **マリガン**: 手札5枚を全てデッキに戻してシャッフルし5枚引き直す**全交換・1回限り**（`do_mulligan`／`keep_hand`／`_check_mulligan_complete`）。
- **付与ドン!!**: キャラ/リーダーに付与すると**自分のターン中のみ** +1000/枚（`CardInstance.get_power(is_my_turn)`）。相手ターン中は加算しない。
- **手札上限**: 公式に手札上限は無い。本システムも上限を設けない（仕様どおり）。

### 1.3 戦闘（バトル）
`declare_attack → (ブロッカー) → (カウンター) → resolve_attack` の順（`gamestate.py:declare_attack`／`handle_block`／`apply_counter`／`resolve_attack`）。

- **最初のターンの攻撃禁止**: 先攻・後攻ともに**自分の最初のターンはリーダー・キャラのいずれもアタックできない**（公式準拠）。ターンは先攻=`turn_count 1`／後攻=`turn_count 2` と交互に進むため、`declare_attack` は `turn_count <= 2` のアタック宣言を弾き、`get_legal_actions` も同条件で攻撃手を列挙しない。
- **アタック宣言**: アクティブなキャラ/リーダーのみ宣言可。宣言で攻撃元はレストする。`ATTACK_DISABLE`／`CANNOT_REST` 等の制限を尊重。
- **攻撃対象**: 相手の**レスト状態のキャラ**か**リーダー**（リーダーは常に対象可。`ATTACK_ACTIVE` 保有時はアクティブキャラも対象可）。自己制限 `CANNOT_ATTACK_LEADER` を尊重。
- **召喚酔い／速攻**（§1.4）。
- **ブロッカー**: アクティブで【ブロッカー】を持つキャラのみブロック可。ブロックでレストし攻撃対象を肩代わり。`ON_BLOCK` 能力を解決。
- **カウンター**: 防御側が手札のカウンター値／【カウンター】イベントで防御。
- **ダメージ解決**: 攻撃側パワー ≥ 対象パワー(+カウンター) で命中。
  - 対リーダー: ライフ上から手札へ（【ダブルアタック】は2枚、【バニッシュ】はトラッシュへ）。ライフ0で攻撃が通れば敗北。
  - 対キャラ: KO（除去保護 `BATTLE_KO`／置換を尊重、`ON_KO` 解決）。
- **【トリガー】**: ライフが手札に加わる際に任意発動（`_pending_triggers` + `CONFIRM_TRIGGER`）。
- **勝敗**: ライフ0で攻撃が通る／デッキ切れ（山札0でドロー）で敗北（`check_victory`、デッキアウト勝利置換に対応）。

### 1.4 召喚酔い／速攻（FIELD: 登場ターンの攻撃制限）
- **仕様**: キャラクターは**登場したターンは攻撃できない**。ただし【速攻】を持つキャラは登場ターンでも攻撃できる。リーダーは登場の概念が無いため召喚酔いの対象外（ただし最初のターンの攻撃禁止は別途 §1.3 で適用）。
- **実装**: 登場時に `CardInstance.is_newly_played=True`（`play_card_action`／効果 `PLAY_CARD`）。自分のリフレッシュで `reset_turn_status` により解除。`declare_attack` で
  `master.type==CHARACTER and is_newly_played and not has_keyword("速攻")` を弾く。
  `has_keyword` は付与/timed キーワードも含めて判定する。

### 1.5 場のキャラクター5体上限（強制トラッシュ）
- **仕様**: キャラクターエリアは最大**5体**。6体目を登場させた場合、自分のキャラ1体（**新規登場分を含む**）を選んでトラッシュし5体に戻す（公式準拠）。ステージ（`owner.stage`）・ドン!!は対象外。
- **実装**: 定数 `FIELD_LIMIT=5`。登場2経路（手札 `play_card_action`／効果 `PLAY_CARD`）の ON_PLAY 解決後と、`resolve_interaction` 末尾で `_enforce_field_limit(owner)` を呼ぶ。超過時は `_suspend_for_field_overflow` が中断要求 `active_interaction.action_type="FIELD_OVERFLOW_TRASH"`（候補=自分の全キャラ、`min=max=超過数`、`can_skip=False`）を立てる。
  - フロント向けには `get_pending_request` が `SEARCH_AND_SELECT` にマップ（既存の選択UIを再利用、**フロント変更不要**）。
  - 選択は `resolve_interaction` の `FIELD_OVERFLOW_TRASH` ブランチで処理し、選んだカードをトラッシュ。**KOではない**ため `ON_KO` 誘発は起こさない。
  - ON_PLAY が対話を起こす場合は `if not self.active_interaction:` ガードで中断のネストを避け、対話完了後に末尾チェックで拾う（1プレイヤーずつ逐次化）。

---

## 2. オンライン対戦アーキテクチャ（ルールモード）

ルールモードのオンライン対人戦は、フリーモード(sandbox)のルーム制を踏襲しつつ、対局進行に
本物のルールエンジン(`GameManager`)を使う。状態同期は WebSocket で**全情報を配信し、相手手札の
非表示などの表示制御はフロント側で行う**方針（フリーモードと同水準）。

### 2.1 ルーム（ロビー）
`app.py` の `RULE_ROOMS: Dict[str, dict]` レジストリ。各ルーム = `{game_id, room_name, created_at,
status(WAITING/PLAYING/FINISHED), ready{p1,p2}, decks{p1,p2}, deck_preview{p1,p2}}`。
対局開始後の `GameManager` 本体は `GAMES[game_id]` に格納し、進行は既存の `/api/game/*` を共用する。

| エンドポイント | 役割 |
|---|---|
| `POST /api/rule/create` | ルーム作成（WAITING）。`game_id` とロビー状態を返す |
| `GET /api/rule/list` | 募集中ルーム一覧（game_id/room_name/接続数/status/ready） |
| `POST /api/rule/action` | ロビー操作：`SET_DECK`（デッキ選択＝当該プレイヤー ready）／`START`（両者 ready で対局生成）／`KICK_PLAYER` |
| `WS /ws/game/{game_id}` | 対局/ルーム状態の購読。接続時に現在状態を送信 |

### 2.2 状態配信
- `GameConnectionManager`（`game_ws_manager`）が `game_id` ごとの接続を保持し、全接続へ同一ペイロードをブロードキャストする（視点別シリアライズはしない）。
- `build_rule_message(game_id)` がペイロードを生成：`{type:"STATE_UPDATE", game_id, room_name, status, ready_states, deck_preview, ...}`。`PLAYING/FINISHED` 時は `build_game_result_hybrid` の結果（`success/game_state/pending_request/action_events`）を内包。`WAITING` 時は `game_state=None`。
- `broadcast_rule_state(game_id)` を、既存 `/api/game/action`・`/api/game/battle` の成功時と、`/api/rule/action` で呼ぶ（**ルーム対局のみ**。非ルーム＝ソロ対局には影響しない）。`manager.winner` が立てば `status=FINISHED`。
- 切断時は `GameConnectionManager` が猶予期間後に `RULE_ROOMS`/`GAMES` を掃除する。

### 2.3 開始フロー
1. ロビーでルーム作成/参加 → クライアントが `/ws/game/{id}` を購読。
2. 各プレイヤーが `SET_DECK`（ready が立つ）。
3. ホスト(p1)が `START` → 両者デッキで `GameManager` 生成・`start_game()`・`GAMES[game_id]` 格納・`status=PLAYING`・ブロードキャスト。
4. 対局は `/api/game/action`・`/api/game/battle`（player_id を伴う）で進め、毎アクションで全接続へ配信。

### 2.4 情報秘匿の方針
相手手札・裏向きライフの識別情報（name/card_id/text）は**サーバから配信される**。秘匿は
フロント側で行う（相手側の手札を裏向き描画する等。`frontend/docs/SPEC.md` 参照）。これはフリー
モードの既存挙動と同水準の設計判断。サーバ側秘匿が必要になった場合は WS 接続にプレイヤー識別を
持たせ、視点別シリアライズ＋伏せカードの redaction を行う拡張余地がある。

---

## 2.5 CPU 対戦（ルールモード・ソロ）

人間（p1）が CPU（p2）と対戦する。AI はバックエンドの `core/cpu_ai.py` に置き、
`GameManager.clone()` 上で各合法手を試して評価する 1-ply 先読みで意思決定する。

- **生成**: `POST /api/game/create` に `vs_cpu:true` / `cpu_difficulty`（easy/normal/hard）/
  `cpu_deck`。CPU メタは `app.py` の `CPU_GAMES` に保持（`{cpu_player_id, difficulty}`）。
- **逐次進行**: `POST /api/game/cpu/step {game_id}` が CPU の次の 1 手を `action_api` 経由で適用し、
  `{cpu_acted, cpu_event, waiting_for}` を返す（`waiting_for`: `cpu`=継続/`human`/`human_decision`/`game_over`）。
  フロントは `waiting_for!='cpu'` までポーリングして 1 手ずつ演出する。
- **暴走防止**: ターン内の手総数キャップと起動効果/ドン付与の繰り返しキャップで CPU ターンが必ず終わる。
- 合法手は `GameManager.get_legal_actions`、効果対話の既定解決は `default_interaction_payload`。
- 詳細・効果検証ハーネス（CPU 対 CPU 自己対戦）は [`docs/CPU_BATTLE_PLAN.md`](CPU_BATTLE_PLAN.md)。

---

## 3. カード効果システム

カードの日本語テキストを解析し中間表現(IR)へ落とし、対局中に解決する。設計詳細は
[`docs/parser_v2.md`](parser_v2.md)。

### 3.1 効果処理パイプライン
```
カードDB(日本語) ─ loader._create_card_master/make_parser ─▶ EffectParserV2
  ・構造分解(レガシー流用) + 原子句のみ rules で解釈 + 未対応はレガシーへフォールバック
  ─▶ Ability(IR: trigger/condition/cost/effect)
  ─▶ resolver.py(EffectResolver) : AST を実行スタックで処理（対象選択/任意確認は中断/再開）
  ─▶ gamestate.py(apply_action_to_engine / continuous / 除去保護 / 誘発キュー) ─▶ 盤面更新
```
`EffectParserV2` は `EffectParser`(レガシー) を継承し `_parse_atomic_action()` のみ上書き。トリガー
判定・コスト分離・逐次/分岐/選択肢の構造分解はレガシーを使う。原子句は `default_registry.apply(ctx)`
でルール優先解釈し、不一致はレガシーへフォールバックして `unmatched`／`fallback_other` に記録する。

### 3.2 中間表現(IR)
`models/effect_types.py`。`Ability`（trigger/condition/cost/effect）を頂点に、効果ツリーは
`GameAction`/`Sequence`/`Branch`/`Choice` の組合せ。`GameAction.sub_effect`=置換効果の置換アクション、
`GameAction.face_up`=ライフへの向き、`Ability.cost_optional`=任意コスト。

### 3.3 継続効果（期間付き効果）
`effects/continuous.py` の `ContinuousEffectManager`。

- `CardInstance` の `timed_power`/`timed_cost`/`timed_flags`/`timed_keywords` に反映。これらは
  `reset_turn_status()` でクリアされない（`power_buff`/`cost_buff`/`flags`/`current_keywords` とは別）。
- kind: `POWER`/`COST`/`FLAG`/`KEYWORD`。Duration: `THIS_BATTLE`/`THIS_TURN`/`UNTIL_NEXT_TURN_END`/`PERMANENT`。
- 失効は `expire(event)` をバトル終了(`resolve_attack`)・ターン終了(`end_turn`)で呼ぶ。場を離れる際は `move_card` が `drop_for(uuid)`。
- **効果無効化**は `FLAG "EFFECTS_DISABLED"`（timed_flags）。参照側は `CardInstance.is_effect_negated`（`ability_disabled` または timed_flags の `EFFECTS_DISABLED`）。能力発動ガード・キーワード判定・除去保護の走査がこれを見る。

### 3.4 除去保護（PREVENT_LEAVE）と置換効果（REPLACE_EFFECT）
`gamestate._active_protection(card, status)`／`_active_replacement(card, status)`。除去が起こる瞬間に
対象の PASSIVE を走査し条件をその場で評価する（フラグをラッチしない）。

- 保護 `PREVENT_LEAVE`: `LEAVE`（あらゆる除去）／`EFFECT_KO`（KO 限定＝手札戻し等の非KO除去には効かない）／`BATTLE_KO`。除去ディスパッチは KO に `("LEAVE","EFFECT_KO")`、非KO除去に `("LEAVE",)` を照合。
- 置換 `REPLACE_EFFECT`: 「代わりに〜」。`_can_satisfy_node` を満たせば `sub_effect` を実行し本来の除去をスキップ。`sub_effect` の実行・実行可能性判定の source は**離れるカード**（条件/ターン1回は能力保持カード）。「代わりに（そのカードを）ライフに加える」等が離れるカード自身を対象に取れる（OP11-101）。
- 置換 `sub_effect` の中断は `_auto_resolve_replacement` が同期解決（任意=accept、対象=自動選択）。`active_interaction` は単一スロット設計。

### 3.5 誘発・対話・コスト
- 誘発は `_pending_triggers` キュー経由。`move_card` がライフ離脱を `ON_LIFE_DECREASE` として積み、API境界/対話完了/戦闘・効果ダメージ末尾でドレイン（二重計上しない）。
- 【トリガー】（ライフ公開）は `CONFIRM_TRIGGER` で確認してから解決。複数枚はキューで保持し中断跨ぎで消えない。
- 【ドン‼×N】= `Condition(HAS_DON, value=N, GE)`。`source_card.attached_don` を見る。
- 任意コスト能力は `Ability.cost_optional`。自動誘発は発動前に `CONFIRM_OPTIONAL` で確認（`ACTIVATE_MAIN` は対象外）。
- 遅延「ターン終了時、」は `GameAction.delay="TURN_END"` → `pending_end_of_turn` → `end_turn` で解決。

### 3.6 対象解決・値解析（要点）
- `parse_target` は主語修飾（特徴/コスト上限/枚数）を保全。「相手が選び」等は `chooser` へ。期間/タイミング句の「(次の)相手の…」は player 判定から除外。
- 「持ち主の〜」系除去（手札に戻す/デッキの下に置く/ライフに加える）で**側無指定**の対象は自分・相手の両方（`Player.ALL`）。`get_target_cards` は `ALL` 候補を**「相手→自分」順**に並べ、既定選択（CPU/自己対戦/監査）は相手キャラを選ぶ（UI は両側選択可）。「自分の/相手の」明示と「この…」自己参照は除外。
- 隠しゾーン（ライフ/デッキ）の対象は上から自動取得（情報リーク防止）。明示公開選択は `TargetQuery.flags` の `"REVEAL_SELECT"` で対話へ。
- 「他の／このキャラ以外」→ `EXCLUDE_SOURCE`。coreference「そのキャラ」は選択結果を `saved_targets` 参照。
- 自己制限（self_cannot）は `player.restrictions` に記録し各地点で enforce（`SELF_RESTRICTION_KEYS`）。
- 値: 全角符号・丸数字コスト・「N以上/以下/からM」「ちょうどN」「N枚になるように(DOWN_TO_N)」「N枚につき(PREV_ACTION_COUNT/COUNT_QUERY)」を解析。

---

## 4. ファイルマップ（本番コード）

| パス | 役割 |
|---|---|
| `opcg_sim/api/app.py` | FastAPI。REST/WS エンドポイント、`GAMES`/`SANDBOX_GAMES`/`RULE_ROOMS`/`CPU_GAMES`、`/api/game/cpu/step`、`build_game_result_hybrid`、`build_rule_message`/`broadcast_rule_state`、`GameConnectionManager` |
| `opcg_sim/api/schemas.py` | レスポンス/リクエストの Pydantic スキーマ |
| `opcg_sim/src/core/gamestate.py` | ルールエンジン本体（ターン/戦闘/召喚酔い/5体上限/効果解決/除去保護/誘発キュー、`clone`/`get_legal_actions`/`default_interaction_payload`） |
| `opcg_sim/src/core/action_api.py` | アクション適用の共通コアパス（`apply_game_action`/`apply_battle_action`）。HTTP/CPU/自己対戦が共用 |
| `opcg_sim/src/core/cpu_ai.py` | CPU(AI) 意思決定（`evaluate`/`decide`/`decide_guarded`）。1-ply 先読み・難易度・暴走防止 |
| `opcg_sim/src/core/invariants.py` | 対局中インバリアント検出（自己対戦/テストの各ステップ後に呼ぶ） |
| `opcg_sim/src/core/sandbox.py` | フリーモードの盤面マネージャ |
| `opcg_sim/src/core/effects/parser.py` / `parser_v2.py` | レガシー/V2 パーサ |
| `opcg_sim/src/core/effects/rules/base.py` / `atoms.py` | ルール基盤／原子アクションルール群 |
| `opcg_sim/src/core/effects/continuous.py` | 継続効果マネージャ |
| `opcg_sim/src/core/effects/matcher.py` | 対象指定の解析(`parse_target`)・実体化(`get_target_cards`) |
| `opcg_sim/src/core/effects/resolver.py` | IR の実行 |
| `opcg_sim/src/models/effect_types.py` | IR 定義（Ability/GameAction/TargetQuery/Condition…） |
| `opcg_sim/src/models/models.py` | CardMaster/CardInstance（`is_newly_played`、`timed_*`、`has_keyword()`、`is_effect_negated`、`get_power`） |
| `opcg_sim/src/models/enums.py` | ActionType/TriggerType/Zone/Phase/CardType/ConditionType… |
| `opcg_sim/src/utils/loader.py` | カードDB/デッキ読込・`make_parser()`・キーワード抽出（`_STATIC_KEYWORDS`） |
| `shared_constants.json` | フロントと共有する定数（PLAYER_KEYS/CARD_PROPERTIES/c_to_s_interface 等） |

---

## 5. 運用（環境変数）

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_PARSER` | `v2` | `legacy` でレガシーパーサに切替（再デプロイ不要）。V2 読込失敗時も自動退避 |
| `OPCG_LOG_SILENT` | （未設定） | `1` で stdout ログ抑止（テスト/診断用） |

---

## 6. 実装上の不変条件・注意点

- **本番パスは loader 経由**。効果定義は EffectParserV2 の自動解析に一本化（旧 catalog.py の手動オーバーライドは廃止）。
- **テキスト正規化**: パーサは NFC、loader の DataCleaner は NFKC を使う箇所がある。全角/半角・`!!`/`‼`(U+203C)・各種マイナス記号の揺れに両対応する。
- **`timed_*`（power/cost/flags/keywords）は `reset_turn_status` でクリアしない**。期間付き COST/KEYWORD は `timed_cost`/`timed_keywords` に載せる（直接 cost_buff/current_keywords に加えると passive 再計算で消える）。
- **`_apply_passive_effects` は cost_buff/current_keywords を毎回リセット**（power_buff/flags はしない）。`active_interaction` 中は何もしない。`refresh_passive_state()` を API アクション境界で呼ぶ（中断中・再帰中は no-op）。
- **CardMaster は frozen dataclass**。abilities は生成時に確定する。
- **全カード挙動ベースライン `full_card_baseline.json`** は現状挙動の凍結。挙動を変えたら差分をレビューして `tests/full_card_audit.py --regen` で更新する（テスト手順は TEST_SPEC §品質ゲート）。
- **スコープ付き相手効果無効は `Player.negate_onplay_until`**（現状【登場時】(ON_PLAY)のみ）。
- **`parser._parse_to_node` の split_pattern が Sequence 分割境界を定義**する（`。`/`その後、`/連用形 `(?<=置き)、` 等）。

---

## 7. 関連ドキュメント
- テスト仕様: [`docs/TEST_SPEC.md`](TEST_SPEC.md)
- パーサ設計詳細: [`docs/parser_v2.md`](parser_v2.md)
- リーダー個別仕様・既知差異: [`docs/leader_specs/`](leader_specs/README.md)（差異一覧は [`ISSUES.md`](leader_specs/ISSUES.md)）
- フロントエンド仕様: `opcg-sim-frontend/docs/SPEC.md`
