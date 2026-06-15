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
集約され、HTTP エンドポイント・CPU 対戦ドライバ・自己対戦ランナーが**同一コアパス**を通る。これが
ないと AI シミュレーション・自己対戦とルール本番の挙動が乖離するため、適用ロジックは必ずこの関数を
経由する。CPU（AI）対戦の設計は §2.5、効果検証ハーネス（CPU 対 CPU 自己対戦）は
[`docs/TEST_SPEC.md`](TEST_SPEC.md) §3.1 を参照。

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
| `GET /api/game/state?game_id=` | 現在状態を**読み取り専用**で返す（`build_rule_message` と同形）。WS 取りこぼし時の再同期フォールバック。冪等・副作用なし |

### 2.2 状態配信
- `GameConnectionManager`（`game_ws_manager`）が `game_id` ごとの接続を保持し、全接続へ同一ペイロードをブロードキャストする（視点別シリアライズはしない）。
- `build_rule_message(game_id)` がペイロードを生成：`{type:"STATE_UPDATE", game_id, room_name, status, ready_states, deck_preview, ...}`。`PLAYING/FINISHED` 時は `build_game_result_hybrid` の結果（`success/game_state/pending_request/action_events`）を内包。`WAITING` 時は `game_state=None`。
- `broadcast_rule_state(game_id)` を、既存 `/api/game/action`・`/api/game/battle` の成功時と、`/api/rule/action` で呼ぶ（**ルーム対局のみ**。非ルーム＝ソロ対局には影響しない）。`manager.winner` が立てば `status=FINISHED`。
- 切断時は `GameConnectionManager` が猶予期間後に `RULE_ROOMS`/`GAMES` を掃除する。
- **取りこぼし耐性**: 対局進行は相手へ WS ブロードキャストでのみ伝わるため、モバイルのバックグラウンド化・通信瞬断で1回でも取りこぼすと、片側が古い「相手待ち」状態のまま自力復帰できず停止して見える（特にカウンター解決後に攻撃側の手番が戻らない）。フロントは**相手待ちの間だけ** `GET /api/game/state` を軽量ポーリング（約3秒）して最新状態へ再同期する（読み取り専用・冪等。自分の操作待ち中・決着後はポーリングしない）。再接続時の現在状態送信（`GameConnectionManager.connect`）と二重の安全網。

### 2.3 開始フロー
1. ロビーでルーム作成/参加 → クライアントが `/ws/game/{id}` を購読。
2. 各プレイヤーが `SET_DECK`（ready が立つ）。
3. ホスト(p1)が `START` → 両者デッキで `GameManager` 生成・`start_game()`・`GAMES[game_id]` 格納・`status=PLAYING`・ブロードキャスト。
4. 対局は `/api/game/action`・`/api/game/battle`（player_id を伴う）で進め、毎アクションで全接続へ配信。

### 2.4 情報秘匿の方針
相手手札・裏向きライフの識別情報（name/card_id/text）は**サーバから配信される**。秘匿は
フロント側で行う（相手側の手札を裏向き描画する等。`frontend/docs/SPEC.md` 参照）。これはフリー
モードの既存挙動と同水準の設計判断。

---

## 2.5 CPU 対戦・AI（ルールモード・ソロ）

人間（p1）が CPU（p2）と対戦する。AI はバックエンドの `core/cpu_ai.py` に置く。ルールエンジン・
効果解決・勝敗判定がすべてサーバ側にあるため、先読み（状態を複製してシミュレート）も含めてサーバで
完結する。フロントは人間=p1 を操作し、CPU の手はポーリングで 1 手ずつ受け取る。

### 2.5.1 配線・逐次進行
- **生成**: `POST /api/game/create` に `vs_cpu:true` / `cpu_difficulty`（easy/normal/hard）/
  `cpu_deck`。CPU メタは `app.py` の `CPU_GAMES` に保持（`{cpu_player_id, difficulty}`）。
- **逐次進行**: `POST /api/game/cpu/step {game_id}` が CPU の次の 1 手を `action_api`（§0 の共通
  コアパス）経由で適用し、`{cpu_acted, cpu_event, waiting_for}` を返す
  （`waiting_for`: `cpu`=継続 / `human` / `human_decision` / `game_over`）。フロントは
  `waiting_for!='cpu'` までポーリングして 1 手ずつ演出する。毎回再計画するステートレス設計で desync に強い。

### 2.5.2 AI 設計（`cpu_ai.py`）
- **状態複製**: `GameManager.clone()`（`copy.deepcopy` ベース。WebSocket 等の非データ参照は持たない）。
  本体（self）は一切変化させず、`action_events` 等の一時状態はリセットする。
- **合法手列挙**: `GameManager.get_legal_actions(player)`（支払可能な手札・アクティブな攻撃者・有効な
  攻撃対象・起動可能効果）。生成手は `_validate_action` を通る。
- **評価関数** `evaluate(manager, me, see_opp_hand=True)→float`: **J値理論**（J値＝白＝デッキ残＋
  トラッシュ／黒＝手札・ライフ・場・ステージ＝リソース。自分の J値を下げ相手の J値を上げるゲーム）に
  整合する形で黒リソースを加重する。具体的には **ライフ（最重要・非線形＝薄いほど 1 枚の限界価値が高い／
  45[J] ラインの危険）**・**手札（枚数＋カウンター値＝防御に回せる資源）**・**場（存在価値・有効パワー・
  ブロッカー＝最終防御・攻め圧）**・アクティブ DON 差を加重する。KO／カウンター誘発／
  ハンデス等の「相手 +1[J]」は相手側の枚数・パワー低下として差分に自然に反映される。`see_opp_hand`
  フラグで**相手手札を「中身（カウンター値）まで読む（full）」か「枚数のみ（public）」か**を切り替える
  （難易度の情報方針＝後述）。自分の手札は常に full。評価は以下の機微を織り込む（無意味手の抑制と
  J=0 境界の織り込み）:
  - **戦闘の閾値性（有効パワー）**: アタックは「攻撃側パワー ≥ 防御側パワー」で連撃成立なので、パワーは
    **対面の最硬防御（リーダー/場キャラの最大パワー）を上回るまで**が価値を持ち、超過分は強く減衰させる
    （`_effective_power`／係数 `W_POWER_OVERCAP`）。これにより**届かない／過剰なドン付与は静的にはほぼ
    無加点**となり、実際に戦闘結果（ライフ -1／KO）を変える付与だけが多 ply 探索の差分として価値化される。
  - **白（J）の決定境界**: 自デッキ残がデッキ切れ（J=0・ドロー不能＝敗北）へ近づくほど非線形に減点する
    （`DECK_DANGER`／`W_DECK_DANGER`）＝相手を削り切る／自滅ドローを避ける動機。黒は白の相補なので
    素点は据え置き、境界の非線形分のみを足す。
  - **攻め圧は実際に攻撃できる体に限定**: 自ターンの召喚酔い（速攻なし）キャラは今ターン攻撃できないので
    攻め圧（`W_ATTACKER`）を加点しない＝意味のない小型展開で攻め圧を水増ししない（相手ターン視点では
    将来圧として加点し過小評価を避ける）。
  - **「何もしない」を一級の選択肢に**: `decide` はターン終了（パス）を常に比較し、行動が `_ACT_MARGIN`
    を超えて盤面を改善するときのみ採用する＝無意味なキャラ展開・不利アタック・効かないドン付与を採らない。
- **探索**: `decide` が難易度に応じて手を選ぶ。`_search` は **α-β ＋ ビームの多 ply 先読み**で、自分の
  連続手を読み、ターン終了後は相手の応答（ブロック／カウンター／相手ターン）まで min ノードとして辿る。
  探索木内で `winner` に到達する手順を **ply 割引付き**で最高評価とすることで **最短リーサル（削り切り）を
  認識**する。`decide_guarded` が暴走防止ガード（ターン内の手総数キャップ・起動 / ドン付与の繰り返し
  キャップ）でターンの収束を保証する（per-turn 制限の付け忘れカードはこのガードで顕在化）。ルート手は
  1-ply で事前選別し上位 `HARD_ROOT_BEAM` 手だけを深掘りする（深掘りは手ごとに均等予算
  `HARD_PER_MOVE_BUDGET`・深さ `HARD_DEPTH`・各ノード幅 `HARD_BEAM`）。これにより採点の公平性を
  保ちつつ 1 手のレイテンシを実用域（平均 ~1 秒）に保つ。
- **難易度＝情報方針の 3 分化**（API キー `easy/normal/hard` は維持し、挙動を再定義）:
  | UI | キー | 方策 | 相手情報 |
  |---|---|---|---|
  | かんたん | `easy` | 正直な 1-ply 貪欲（ミスなし） | **公開のみ**（相手手札は枚数だけ） |
  | ふつう | `normal` | 多 ply 先読み（リーダー推測の相手モデル） | **公開のみ**＋テンプレ由来の想定（§2.5.4。実手札は読まない） |
  | つよい | `hard` | 多 ply 先読み（フルクローン・最強） | **full**（クローン上の相手手札＝隠れ情報も読む） |
  `easy` は探索せず 1-ply の即時最良手。`normal`/`hard` は多 ply。`normal` は相手 min ノードで**相手の
  隠れ手札に依存する手（手札からの登場・カウンター）を使わない保守モデル**で読む（テンプレ供給時は
  §2.5.4 の想定手で補強）。`hard` のみ相手手札を読む。
- **sim 専用の対話自動解決** `default_interaction_payload`: 先読み中に `active_interaction` /
  `pending_request` が立った場合の機械的な既定確定（対象=ヒューリスティック最良 or 先頭、CONFIRM=
  有利なら使う 等）。**これはクローン上の先読み専用**であり、本番（実対局・自己対戦）の未解決中断は
  握り潰さず [`docs/TEST_SPEC.md`](TEST_SPEC.md) §3.1 のインバリアントで表面化させる（AI が「とりあえず
  動く」ことで効果バグを覆い隠さないための分離）。
- **公平性**: `easy/normal` は隠れ情報（相手手札の中身・裏向きライフ）を見ない（`see_opp_hand=False`＋
  相手 min ノードの手札依存手を除外）＝チート防止。`hard` はユーザ選択により「最強」を優先し、相手応答
  シミュレーションにクローン上の相手手札も用いる別方針（§6 の視点マスクは `hard` 探索には適用しない）。

### 2.5.3 精度向上（実装済み／継続テーマ）
J値理論ベースの評価関数・多 ply 先読み（α-β ＋ ビーム）・最短リーサル認識を実装済み（§2.5.2）。さらなる
強化の継続テーマ:

- **探索の深化**: `HARD_DEPTH`／`HARD_BEAM` の拡大、反復深化・置換表（メモ化）・move ordering の改良。
  `decide_guarded` の収束保証は維持する。
- **評価関数の高度化**: 重みの学習／チューニング、効果連鎖（チェイン）・盤面テンポ・相手のリーサル
  （被削り切り）認識の織り込み。
- **局面別ヒューリスティック**: 攻め／受けの切替、ライフ・ドン・手札リソースのトレードオフ評価。
- **検証**: 改善は §3.1 の CPU 自己対戦ハーネスで決定論・インバリアントを保ちつつ、難易度間の
  勝率（弱 < 強）で精度を回帰確認する。

効果検証ハーネス（CPU 対 CPU 自己対戦・決定論・インバリアント検出）は
[`docs/TEST_SPEC.md`](TEST_SPEC.md) §3.1 を参照。

### 2.5.4 リーダー推測の相手モデル（`normal`・テンプレートデッキ）
`normal`（ふつう）は隠れ情報を読まずに賢く振る舞うため、**相手（人間）のリーダーから「相手はどんな
デッキをどう回すか」を推測する相手モデル**を用いる。

- **テンプレートデッキ**: `leader_id → 代表デッキ（50枚）` を**デッキと同形**で保存する（Firestore
  `cpu_templates` コレクション。`POST /api/cpu_template`・`/list`・`/get`・`DELETE`、shape は `decks`
  と同じ `{id, name, leader_id, card_uuids, don_uuids}`）。フロントは `DeckBuilder` を流用した登録画面で
  入力する（リポジトリ同梱の既定テンプレ＋ユーザ追加）。
- **相手モデルの構築**: 対局開始時、**人間のリーダー**で `cpu_templates` を引当て、テンプレ構成から
  静的な集計（想定カウンター密度／ブロッカー数／除去密度／パワーカーブ／主要脅威）を作る（観測ベリーフ
  更新は行わない＝静的）。テンプレ未登録のリーダーは `easy` 相当の公開情報のみ（保守モデル）へフォール
  バックする。
- **使い所**: 先読みの相手 min ノードで、相手の実手札の代わりに**テンプレ由来の代表的最善応答**（想定
  カウンター予算までの受け・想定ブロッカー）を仮定して読む。これにより相手手札を見ずに「このリーダー
  ならこう守る／攻める」を織り込む。**フェア性**＝相手の実デッキ・実手札は参照せず、リーダーに紐づく
  テンプレ（メタ知識）のみを使う。

> 実装フェーズ: Phase1＝評価の情報モード切替＋ `easy`/`normal`（保守モデル）/`hard` の確定（実装済み）。
> Phase2＝`cpu_templates` レジストリ＋テンプレ由来モデルの供給。Phase3（frontend）＝登録画面・難易度
> 説明文の更新。

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
| `opcg_sim/src/core/cpu_ai.py` | CPU(AI) 意思決定（`evaluate`/`decide`/`_search`/`decide_guarded`）。J値評価・情報方針3分化（easy=正直1-ply・公開／normal=多ply・リーダー推測／hard=多ply・チート）・α-β＋ビーム・最短リーサル認識・暴走防止 |
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
- **「手札のこのカードは、〈条件〉、コスト±N」は手札での自己コスト増減**。パーサ `_try_hand_self_cost` が
  対象＝手札のこのカード自身（`zone=HAND`/`ref_id="self"`/flag `SELF_IN_HAND`）の COST_REDUCTION（PASSIVE）に組む。
  場の PASSIVE 走査では手札カードを評価しないため、`_apply_passive_effects` の Step4（`_apply_hand_self_cost`）が
  手札カードの当該能力の条件を評価し `cost_buff` を加算する（ウタ ST23-001 / サッチ OP16-005 ほか計13枚）。
- **「元々のパワーN」指定は印刷時パワー（`master.power`）で対象を絞る**。matcher が `ORIGINAL_POWER` フラグを
  立て、`get_power`（バフ込みの現在パワー）ではなく `master.power` と比較する（OP16-010 ナミュール等）。

### 6.1 既知の制約（エンジン・モデル化）

リーダー効果の差異は `leader_specs/ISSUES.md`（xfail で固定）。ここではテストで固定して
いない、エンジン側のモデル化上の既知制約を記す。

- **「お互いの〜」の同時両側処理**: 「お互いのライフの上から1枚をトラッシュに置く」
  （OP11-102）や「お互いは手札がN枚になるように捨てる」（OP05-058）のような**両プレイヤーへ
  同時に適用**すべき効果は、`matcher` が「お互い」を `Player.ALL` ＋ `BOTH_SIDES` フラグとして
  解析し、`resolver._resolve_targets` が各プレイヤーで候補・枚数を**個別に解決して結合**する
  （隠しゾーン自動取得・`DOWN_TO_N`・`select_mode=ALL` を各サイド独立に計算）。これにより
  片側のみ解決される問題は解消した。**選択を伴うサイド**（候補>必要枚数。お互いが手札を捨てる
  枚数選択 OP05-058 等）は、そのサイドのプレイヤーに **相手→自分の順で個別に選ばせる**
  （`_resolve_targets` が `_both_sides_pending` を立てて逐次 SELECT_TARGET 中断し、再開で各サイドの
  選択を結合する）。選択の余地が無いサイド（`select_mode=ALL`／位置確定の隠しゾーン=ライフ/デッキ／
  候補≤必要数。OP11-102 のライフ上トラッシュ等）は非中断で確定する。
- **置換 sub_effect のネスト中断（多段継続を含めて対話化済み）**: 置換（REPLACE_EFFECT）は除去解決の
  最中に走る入れ子の中断。中断は `active_interaction`（= `_interaction_stack` 先頭の互換プロパティ）で
  表現する。置換の内側選択（対象選択／任意確認）は**そのまま UI へ提示**して被保護側に選ばせ、
  `resume` で `sub_effect` を完了させる（`_active_replacement(..., can_suspend=True)`、現在は常に許可）。
  **失われる外側継続（多段 / multi-source）**は退避して内側中断の解決後に再開する:
  - **後続シーケンスの退避（B1）**: 除去アクションの後にこのリゾルバの実行スタックが残る場合、
    `_defer_resolver_stack` が後続（execution_stack/context/source）を `_deferred_continuations` へ
    退避し、`execution_stack` を空にして中断を提示する。
  - **複数対象の残対象退避（B2）**: 複数対象除去で先頭対象の置換が中断したら、`_defer_removal_targets`
    が未処理の残対象（uuid＋action＋value）を退避し、ループを抜ける。再開時に
    `apply_action_to_engine` を残対象で再実行する。
  - **再開**: 中断が解消された後（`resolve_interaction` 末尾）に `_resume_deferred_continuations` が
    退避フレームを LIFO で再開する（退避順は「残対象=append→後続シーケンス=insert(0)」なので
    pop() で残対象→後続の順に正しく再開）。`_deferred_continuations` は `clone()` の deepcopy で
    複製され、uuid 解決で再開するため CPU クローン安全。ヘッドレス/CPU の既定応答は従来の
    自動採用と同一結果のため挙動ベースラインは不変。

  **バトル KO 置換の任意確認（対話化済み）**: バトルでKOされる際の**任意**置換（「代わりに〜しても
  よい／できる」OP10-034 フランキー等）は、被KO側へ `CONFIRM_OPTIONAL` を提示して確認する
  （`_suspend_for_battle_ko_replacement` → resume）。**accept** で置換を実行し本来のKOをスキップ、
  **decline** で本来のKO（トラッシュ＋ON_KO）を実行し、いずれも `_finish_attack` で戦闘後処理を
  完了する。検出と実行は `_find_replacement`（適用可能な置換の検出のみ）/`_active_replacement`
  （実行）に分離した。ヘッドレス/CPU の既定応答（index0=accept）は従来の自動採用と一致するため
  挙動ベースラインは不変。

  **継続付与型の置換（配線済み）**: EB02-030「【カウンター】自分のキャラすべては、このターン中、
  バトルでKOされる場合、代わりに自分の手札1枚を捨てることができる」のように、**場に残らない
  発生源（イベント＝即トラッシュ）が「自分のキャラすべて」へ this-turn の置換を付与**するケースは、
  `master.abilities` の場上 protector 走査では拾えない。カウンター解決時に
  `_register_granted_replacements` が `Player.granted_replacements`（`{status, sub_effect,
  is_optional, expire_turn}`、`turn_count <= expire_turn` の遅延失効）へ退避し、`_find_replacement`
  が protector 走査の後にこれも参照する（被KOキャラが自分のキャラのときのみ）。付与する `sub` は
  共有ノードを汚さないようコピーし、「できる／〜してもよい」から `is_optional` を確定するため
  上記の任意確認（CONFIRM_OPTIONAL）で被KO側へ提示される。`granted_replacements` は `clone()` の
  deepcopy で複製され CPU クローン安全。

  > 旧 accepted limitation（バトル KO 置換の拒否・多段 multi-source 継続）および
  > EB02-030 の継続付与型置換の配線は、いずれも解消済み。

---

## 7. 関連ドキュメント
- テスト仕様: [`docs/TEST_SPEC.md`](TEST_SPEC.md)
- パーサ設計詳細: [`docs/parser_v2.md`](parser_v2.md)
- リーダー個別仕様・既知差異: [`docs/leader_specs/`](leader_specs/README.md)（差異一覧は [`ISSUES.md`](leader_specs/ISSUES.md)）
- フロントエンド仕様: `opcg-sim-frontend/docs/SPEC.md`
