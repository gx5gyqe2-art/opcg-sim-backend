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

> **更新（2026-06）**: 出荷 `hard`（α-β）は**情報方針フェア既定＋PIMC（決定化）**で運用（PIMC の詳細＝§2.5.8）。
> 本節の探索/評価の機構（clone・ビーム・settle 等）は現行も有効。
>
> **【2026-06-27 撤去】** 手書き J値評価（`_side_score` とその重み定数群）／評価v2フラグ（`OPCG_EVAL_V2`・
> `set_eval_v2_override`/`_eval_v2_active` の二重経路）／effect-value 実験（`_effect_weights`/`_build_effect_value_cache`/
> `_effect_impact_one`/`EFFECT_VALUE_WEIGHT` 等の既定 OFF 実験）／相手モデル `profile` による評価補正
> （`_own_life_knee`・`W_LIFE_AGGRO_K`・`_estimate_counter_buffer`＝B-1(b) カウンター緩衝ベリーフ・
> `defense_factor`/`aggro_lean` の eval 補正・`profile` 引数）を撤去。
>
> **【2026-06-28 撤去】** **学習価値サブシステムを全廃**（A/B で Elo 中立＝出荷係数は既に局所最適と実証）：
> 葉ブレンド `_value_blend`／`_VALUE_BLEND_SCALE`／`cpu_value_model`・`cpu_features`・`cpu_value_data`・
> `value_model.json`／API の価値サンプル採取／L1 SPSA チューナ（`eval_v2_spsa.py`）／#4 settle pressure
> （`V2_W_SETTLE_THREAT`/`settle_threat_penalty`）／マリガン方策／settle-PASS 計装。経緯と実測は
> `docs/reports/cpu_strength_plan_20260628.md` §I〜§K（**全レバーが出荷済み/失敗/既出/幽霊＝現アーキは上限近傍**）。
>
> **CPU 評価は L1（カード通貨）単一系統** ＝`evaluate_base` は `cpu_eval_v2.evaluate_v2(...)` を素返しし、
> `evaluate` は `evaluate_base` の別名（ブレンド層なし）。本節以下の「手書き J値評価」「`_side_score`」「ライフ
> concave・薄域膝（`W_LIFE`/`W_LIFE_LOW`/`W_LIFE_HIGH`）」「相手モデル profile 補正」を述べた設計テキストは
> **廃止前の設計記録（履歴）**。L1 で存続するのは `_power_cap`/`_effective_power`/`W_POWER_OVERCAP`/`DECK_DANGER`
> （いずれも `cpu_eval_v2` が import して使用）と PIMC・α-β/ビーム探索・手番計画/キャッシュ。

- **状態複製**: `GameManager.clone()`（`copy.deepcopy` ベース。WebSocket 等の非データ参照は持たない）。
  本体（self）は一切変化させず、`action_events` 等の一時状態はリセットする。**先読みの支配的コストは
  この clone**（プロファイルで ~96%）。`CardMaster`（不変＝共有）に加え `CardInstance`／`DonInstance` に
  **高速 `__deepcopy__`** を実装し、汎用 deepcopy の内省・再帰を排除（フィールドはスカラ＋プリミティブ要素の
  set/dict＋共有 master に限られるので、set/dict は浅コピーで独立な深コピーになる。想定外の可変属性のみ
  汎用 deepcopy へフォールバック）。clone が **~3 倍高速化**＝同レイテンシで探索ホライズンを深められる
  （下記 `HARD_HORIZON=3`）。回帰=`tests/test_clone_fast.py`（独立性・master 共有・複数参照の同一性）。
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
    無加点**となり、実際に戦闘結果（ライフ -1／KO）を変える付与だけが探索の差分として価値化される。
  - **白（J）の決定境界**: 自デッキ残がデッキ切れ（J=0・ドロー不能＝敗北）へ近づくほど非線形に減点する
    （`DECK_DANGER`／`W_DECK_DANGER`）＝相手を削り切る／自滅ドローを避ける動機。黒は白の相補なので
    素点は据え置き、境界の非線形分のみを足す。
  - **攻め圧は実際に攻撃できる体に限定**: 自ターンの召喚酔い（速攻なし）キャラは今ターン攻撃できないので
    攻め圧（`W_ATTACKER`）を加点しない＝意味のない小型展開で攻め圧を水増ししない（相手ターン視点では
    将来圧として加点し過小評価を避ける）。
  - **「何もしない」を一級の選択肢に**: `decide` はターン終了（パス）を常に比較し、行動が `_ACT_MARGIN`
    を超えて盤面を改善するときのみ採用する＝無意味なキャラ展開・不利アタック・効かないドン付与を採らない。
- **探索（ターン境界評価・α-β＋ビーム）**: `decide` が難易度に応じて手を選ぶ。`_search` は葉の評価点を
  **`start_turn` から `horizon` ターン進んだ MAIN_ACTION（一定の静止点）に固定**する。自ターン内
  （diff=0）の自分のメイン手は max、自分のアタックへの相手のブロック／カウンター応答は min。**全候補が
  同じ静止点で評価される**ため、手番パリティ／horizon による「何もしない（パス）が不当に低く評価され、
  やる事が無くても何かしてしまう」バイアスが消える（例: 5000未満でドン付与してリーダーに通らないアタック＝
  純損は、パスより低く出て畳まれる。意図的レスト生成のための非貫通アタックは後続効果で盤面が良くなるなら
  正当に残る）。探索木内で `winner` に到達する手順を **ply 割引付き**で最高評価とし **最短リーサルを認識**する。
  予算/ply 上限の打ち切りは `_settle_eval` で TURN_END／既定解決により**ターン境界へ整流してから評価**する
  （自ターン途中の甘い局面で評価せず＝horizon の抜け道を塞ぐ）。`decide_guarded` が手総数キャップ
  （`TURN_ACTION_CAP`＝終了保証の最終防壁）で収束を保証する。**【2026-06-27】** 旧 `REPEAT_CAP`（同一起動
  効果の繰り返し除外）は撤去＝起動効果はエンジンの正規ゲート（コスト充足・ターン使用回数）で自己制限する
  （自己レストコストの `cost_optional` 誤判定＋`ref_id='self'` 充足の食い違いを修正済み）。
  - **ホライズン**: ルート手は 1-ply で事前選別し、上位 `HARD_ROOT_BEAM` 手＋`TURN_END`（パスの基準線）
    のみを **`HARD_HORIZON`（既定=4）ターン先**まで深掘りする（深掘り集合のみ採用＝評価ホライズンを一致
    させ誤選択を防ぐ）。**horizon=1=B1**（相手ターン開始で評価・相手ターンへ潜らない）、**horizon=2=B2-lite**
    （相手のターンを丸ごと＝攻撃まで読み、自分の次ターン開始で評価）＝相手の反撃に対する守り（ブロッカー／
    カウンター温存）を min/max で読む。**horizon=3**（さらに自分の次ターン→相手の次ターン開始まで）で
    相手の反撃後の立て直しまで読む。**horizon=4**（さらに相手の次ターン→自分の次々ターン付近）で相手の
    再反撃まで読む。手ごと均等予算 `HARD_PER_MOVE_BUDGET`（=300）・各ノード幅 `HARD_BEAM`・総 ply 上限
    `HARD_MAX_PLY`（=52）。**horizon だけ上げても予算切れで読み切れない**（予算が実深さの律速）ため、
    horizon を上げるときは settle（予算切れ）率が同等になるよう予算も連動させる（horizon3→4 で 90→150）。
    予算は当初 150 だったが、実測（hard 自己対戦）で葉の到達深さが中央値 ~1 ターン・horizon=4 到達は ~4%
    と判明し、1 手レイテンシに約 2 倍の余力（思考手 mean ~210ms / 最大 ~465ms）があったため 150→300 へ
    増量（1 秒厳守のまま 4T 到達 4.4%→17.6%。Phase 1）。以降の深掘りは壁時計デッドライン化＋passive 最適化
    （Phase 2）で 1 秒以内のまま到達率を伸ばす。
    深さの拡張史: clone の ~3 倍高速化（`__deepcopy__`）で horizon 2→3（予算 36→90）、その後の make/unmake
    ＋同一性比較等で探索 ~4.2x 高速化した余力で **horizon 3→4（予算 90→150・maxply 40→52）**。後者は
    **A/B 自己対戦（horizon4 vs horizon3・both hard・席交互・独立2シード群 計60局）で 35/60＝58.3%＝+58 Elo**
    （両群 >50%・戦術退行なし）と実測検証して採用（2026-06）。予算はレイテンシ予算（切れても settle で境界
    評価）で、中盤 decide 実測 **~516ms**（高速化前 horizon=3 の ~1176ms より速く、かつ 1 ターン深い）。
    `easy` は素の 1-ply のまま（探索なし）。
  - **PyPy 探索オフロード（方式B・プロセス分離・実装済み 2026-06）**: 探索（`decide`）の実行だけを **PyPy
    ワーカープロセス**へ委譲し、JIT で高速化する（CPython 比 **~2.1x**・中盤 decide median 222.6ms→101.6ms・
    改変ゼロ・**挙動ビット一致**＝同一337step/280decide を再生）。ランタイム差し替えであって**方策・評価・カード
    挙動は一切変えない**（エンジン `src/**` は stdlib-only ゆえ PyPy でそのまま動く）。配信スタック
    （`pydantic-core`＝Rust／`grpcio`＝C 拡張）は PyPy 非互換のため **CPython 側に据え置き**、探索だけを分離する
    （Phase0 互換スパイクで A 単一プロセスは不成立と判定・`docs/reports/pypy_phase0_result_20260620.md`）。
    配線: `api/app.py` の cpu/step が `api/decide_client.py` 経由で **盤面を pickle 送信**（`GameManager` は
    `__getstate__` 不要で round-trip・~69KB／decide）、PyPy 側 `tools/decide_worker.py` が `cpu_ai.decide_guarded`
    を実行し `(move, trace, mem)` を返す。**`OPCG_PYPY_WORKER=1` で有効・未起動/IPC 失敗/0 でインプロセス実行へ
    自動フォールバック**（可用性・ロールバック）。決定性は RNG 状態を IPC で渡してタイブレークを一致させる。
    回帰=`tests/test_pypy_worker_parity.py`（pickle round-trip 同一手・profile/plan/mem 往復・ブリッジ=インプロセス
    同値）。実測・手順は `docs/reports/cpu_search_accel_pypy_20260620.md`／移行手順は同 `pypy_migration_runbook_20260620.md`。
    PyPy の ~2.1x は下記の算法系高速化（差分評価・LMR 等）と**乗算**で効く（horizon +1〜2 の足場）。
  - **探索高速化ロードマップ（horizon4＋へ・計画／未実装）**: 予算（=clone 回数）が実深さの律速で、その
    clone(deepcopy) が先読みコストの **~96%**（プロファイル）。さらに深くするには clone コストを下げるのが本筋。
    計測（2026-06-19・中盤6局面）では探索ノードの **転置率（手順違いで同一盤面の再出現）≈23%**（範囲15〜36%）。
    - **② インクリメンタル clone（make/unmake・ジャーナリング＋スナップショット照合）【PoC 実装済み**: 盤面を
      1 つだけ持ち「手を適用→再帰→巻き戻し(undo)」で per-node の deepcopy を排除する（最大レバー）。最大リスク
      「undo 漏れで静かに盤面破壊」は、**適用前の盤面スナップショットを保持し undo 後に完全等価を assert**
      （make/unmake 不変条件）して**テスト失敗に変換**し、実プレイ全手で undo の取りこぼしを炙り出す（フル等価
      比較 `deep_diff`＝デバッグ時のみ・本番 OFF）。実装は効果ごとの逆操作手書きでなく**ミューテーション・
      ジャーナリング**（`opcg_sim/src/core/journal.py`：`transaction()`／`JournaledList・Set・Dict`／
      `__setattr__` 旧値記録→逆順再生）で構造的に取りこぼしを防ぐ。**不活性時（transaction 外）は組み込み型・
      素の __setattr__ と完全同一**（グローバル 1 読みで素通り）＝通常プレイ無影響。
      **PoC 結果（2026-06）**: 基盤を `CardInstance/DonInstance/Player/GameManager/ContinuousEffectManager/
      EffectResolver` に配線し、状態コンテナを journaled 型化。`tests/test_journal.py` が「適用→巻き戻し→開始
      deepcopy と完全一致」を実プレイ全手で照合（**全 1028 pass・構造監査 0＝不活性時の挙動完全不変**）。
      ベンチ＝**clone 1.02ms → make/unmake 0.24ms = 4.3x**（per-node コピーコスト）。
      **boundary**: 「非中断＝resolver が parked でない静止点から適用する手」が対象（CPU 探索の根もこの静止点）。
      **中断（複数段効果解決の途中）を再開する手**は parked resolver 状態（`execution_stack`・continuation の共有／
      ネスト構造）を持ち越すため対象外＝clone へフォールバックする。
      **実探索への統合済み（ハイブリッド・2026-06）**: `cpu_ai.py` の 1-ply 採点（ビーム選別＝`_score_move_1ply`）と
      ビーム手の深掘り再帰（`_recurse_child`＝入れ子トランザクションで manager をその場適用→`_search` 再帰→巻き戻し）
      を make/unmake 化し、`_mu_safe`（active_interaction が None）な静止点のみ適用・中断再開は clone。`_USE_MAKE_UNMAKE`
      フラグで即時に従来挙動へ戻せる。**方策は clone 方式と完全同一**（内部最適化）＝`tests/test_cpu_make_unmake.py` が
      (1)decide 選択手一致・(2)`_scored_search` 深掘りスコア一致・(3)decide が manager を無変更（巻き戻し完全）を機械照合。
      **ベンチ＝hard decide 1312ms→367ms＝3.57x**（中盤5局面平均）。全1032pass・構造監査0・カード挙動ベースライン不変。
      スレッド安全性【threadlocal 化済み・2026-06】: journal 状態（`_active`/`_gen_counter`/`_mut_count`）は当初
      プロセス共有グローバルで「探索は単一スレッド」前提だった。だが本番ポンダリングが `asyncio.to_thread`（OS スレッド）
      でも探索を走らせ、**あるスレッドの transaction 中に別スレッドが live オブジェクトの属性を初回セット→その set が
      別スレッドの journal へ誤記録→rollback で live から属性が pop**＝間欠クラッシュ（`CardInstance has no attribute
      'master'`）を起こした。対策として journal 状態を **`threading.local`** にし（各スレッドの記録が互いに漏れない・
      単一スレッド時は従来同値）、外部参照（`journal._active` 等）は PEP 562 の `__getattr__` で後方互換、ホットパス
      （各 `__setattr__`・passive dirty-flag）は `journal._TL.active` を直接読む。`_ponder_plan` も live 盤面を
      メインスレッドで clone してから `to_thread` へ渡す（deepcopy 競合の防止）。並行回帰 `tests/test_journal_concurrency.py`
      （スレッド 2 本を Event で同期し、旧グローバル実装なら必ず壊れるシナリオで live 属性が消えないことを固定）。
      なお当初は止血としてポンダリングを Dockerfile で無効化していたが、**本 threadlocal 化で安全化済みのため再有効化済み**
      （現行 Dockerfile は `OPCG_PONDER=1`／`OPCG_PONDER_SPEC=1`＝⑥-a 先行計画・⑥-b 投機ポンダリングともに既定 ON）。
      ルート `_scored_search` も make/unmake 化済み（`_eval_root_move`・clone 版と完全同値）。**clone 除去の床に到達**:
      残クローンの内訳実測（hard decide）＝中断状態の再帰フォールバック ~90%（parked resolver 未 journaled）・
      ルート `_scored_search` ~5%（≈ decide の 0.7%・変換効果はノイズ内）。残コストは clone でなく **apply＋evaluate**。
      `_apply_modeled_counter` は SELECT_COUNTER＝中断でフォールバックのため変換無益。
      **B-1 エンジン最適化（clone でない部分・2026-06）**: ②後の実測で残コストは clone でなく apply＋evaluate と判明し、
      毎ノード走るエンジン本体を最適化した（方策不変・等価ゲート緑）。
      (1) **`CardInstance`/`DonInstance` を同一性比較（`@dataclass(eq=False)`）**: dataclass 既定の値ベース __eq__（全 ~25
      フィールドの逐次比較）が `card in zone`／`leader == card` を激重にし `_find_card_location` が探索の ~17% を占めていた。
      カードは固有実体（同一 uuid＝同一オブジェクト）なので同一性比較（ポインタ・id ハッシュで hashable 化）に戻す＝盤面内は
      オブジェクト同一性＝論理同一カード・盤面跨ぎは uuid で引く（`_find_card_by_uuid`）ため挙動不変。**hard decide 1.30x**。
      (2) **軽量 `pending_actor_action()`**: `get_pending_request` は毎回 selectable 構築・候補 to_dict・`uuid4()` を作るが
      探索は (player_id, action) しか見ない。軽量版を `_search` の手番/葉判定に使う（副作用の phase 正規化はフル版と一致・
      `test_pending_actor_action_matches_full` で機械照合）。
      **通算: hard decide ~1176ms→~278ms＝~4.2x**（make/unmake×eq×pending・全1033pass・構造監査0・カード挙動ベースライン不変）。
      (3) **`_apply_passive_effects` Step1 リセットの差分書き込み化（2026-06・~1.12x）**: PyPy オフロード後の再プロファイルで
      **残コストの支配項は eval ではなく「make/unmake の journaled setattr 量」＋「passive 再計算」と判明**（`evaluate` は
      cumulative ~10%）。両者は**同一の書き込み**＝`_apply_passive_effects` の Step1 が**毎ノード全カードのバフ
      （`cost_buff`/`passive_power`/`passive_power_override`/`current_keywords`/`passive_counter`）を無条件リセット**し、
      `__setattr__`→`record_attr` が探索の最大の self コストだった（実測 __setattr__ 357k 回・record_attr 397k 回）。
      無条件代入を**「値が変わるときだけ書く」ガード**に変更（大半のカードはバフ 0＝no-op 代入を消す）。最終状態は
      無条件代入と完全同一＝挙動・方策不変。実測 **__setattr__ 357k→85k（−76%）・record_attr 397k→125k（−69%）・
      rollback −55%・`_apply_passive_effects` cumulative −60%**＝controlled A/B で **hard decide ~1.12x**（同一手・全1038pass・
      構造監査0・ベースライン不変）。**PyPy オフロード（~2.1x）と乗算で対 CPython ~2.35x**。
      (4) **parked resolver の journaled 化＝中断再開手も make/unmake（2026-06・~1.33x）**: (3)後の再プロファイルで
      **最大の単一残項は deepcopy（残 clone フォールバック）**＝`_mu_safe` が `active_interaction is not None`（中断再開）で
      clone へ退避していた（全ノードの ~4.3% だが時間では cumulative ~26%）。中断状態を **journaled 化**して make/unmake へ移した:
      EffectResolver に journaled `__setattr__`、`context`・`saved_targets`・`saved_values`・`_grp_consumed`・`_both_sides`・
      `_confirmed_optionals` を JournaledDict/Set 化、`execution_stack`・`saved_stack`・退避スタック（`_deferred_continuations`）を
      JournaledList 化、誘発待ち item（`_pending_triggers`）を JournaledDict 化、`Player` のゾーン list 代入を JournaledList へ昇格
      （素 list 代入でも巻き戻せる安全網）。`_mu_safe` を常時 `make/unmake` に解放。**正しさゲート**＝`tests/test_journal.py` の
      **parked round-trip**（中断再開手で「適用→巻き戻し→開始 deepcopy と完全一致」を実プレイで機械照合）＋全1039pass（任意コスト
      確認など parked 判断の decide 一致を含む）。controlled A/B で **hard decide ~1.33x**（同一手）。
      (5) **ドレイン/整流/読み筋ループの軽量 pending 化（2026-06・~1.06x）**: 再プロファイルで `get_pending_request`
      （毎回 selectable 構築＋`uuid4()`）が探索の ~17%（12,302 回）と判明。`_drain_own_interactions`／`_settle_eval`／
      読み筋ループは **(pid, action) しか見ない**ので軽量 `pending_actor_action` で判定し、フル payload は実際に既定解決する
      ときだけ作る（呼数 12,302→7,018・−43%）。方策不変（`pending_actor_action` は get_pending_request と (pid,action)・
      副作用一致）。controlled A/B で **~1.06x**・全1039pass・構造監査0。
      > **API 配線（Phase 3 ①-b・2026-06）**: `/api/game/cpu/step` に計画キャッシュを配線（`OPCG_PLAN_CACHE=1`・
      > 本番体感最適化・既定 OFF＝従来挙動完全同値）。対局ごとの `meta["plan_cache"]={"queue":[...]}` を保持し、
      > `_cached_cpu_move` が (a) 次の計画手が現局面で**合法なら即返す**（探索なし・**ワーカー往復なし**＝即時 replay）、
      > (b) ミス/前提崩れなら PyPy ワーカーの **plan モード**（`decide_worker._handle` の 10要素目 `mode="plan"`／
      > `decide_client.plan_segment`）で `plan_turn` を実行しセグメントを再計画してキャッシュ。先頭手が不正なら通常
      > `decide` へフォールバック（合法性検証で常に安全）。トレース採取時は per-action を維持（読み筋記録）。これにより
      > **セグメント先頭=1回のワーカー計画→残りは CPython 即時 replay**＝待ちを1回に集約。検証 `tests/test_plan_cache.py`
      > （`plan_segment`==`plan_turn`・`_cached_cpu_move` の合法/replay）。残: A/B・⑥ ポンダリング（相手の番に前倒し）。
      > **不採用メモ（B）**: `_apply_passive_effects` Step2/3 の「常在トリガーを持つカードだけ走査」案は、既存の
      > `if not card.master.abilities` 空チェックで能力なしカードが既に安く弾かれており、per-card の判定関数呼び出しが
      > 逆にオーバーヘッド＝**ネット負**で撤回（再プロファイルで cumulative 悪化を確認）。
      **通算: PyPy(~2.1x) × passive 差分化(~1.12x) × parked journaled(~1.33x) × 軽量pending(~1.06x)**（局面依存）。
      残コストは clone でなく apply＋evaluate（rollback／_side_score／_apply_passive_effects）。
      (6) **`_apply_passive_effects` の継続効果再計算 dirty-flag（Phase2・2026-06・~1.07x）**: 再プロファイルで
      `_apply_passive_effects` が依然最大ホットスポット（cumulative ~26%・大半は継続効果 PASSIVE/YOUR_TURN の `resolve_ability`
      再計算）と判明。**実測で呼び出しの ~38.5% が「前回再計算から盤面入力が不変」＝冗長**（同一入力で同一結果を再計算）。
      `journal._mut_count`（**journaled な全変更＝`record_attr`／`JournaledList・Set・Dict._touch` ＋探索開始の top-level
      transaction 入場で +1**＝探索中の入力変化を漏れなく捕捉）を導入し、`_apply_passive_effects` は「前回再計算時の
      `_mut_count` と一致＝入力不変」なら再計算を**スキップ**する（バフ＝`cost_buff`/`passive_power`/`current_keywords` は前回値の
      まま正しい）。**探索中(make/unmake)のみ作動**＝正常プレイ（`_active is None`）は一切作動せず常に再計算＝従来挙動と完全同値。
      ロールバックはバフを正しく復元する＋`_mut_count ≥ _passive_mc` を保つため省略しても必要時は必ず再計算され安全
      （`_passive_mc` は `object.__setattr__` で更新＝journaled せず＝`deep_diff` 比較対象外）。**正しさゲート**＝`test_journal`
      の make/unmake round-trip（`deep_diff==None`）＋`test_cpu_make_unmake`（decide 選択手が clone 方式と一致＝方策不変）。
      実測 `_apply_passive_effects` cumulative **11.2s→9.5s**・hard 思考手 decide **mean 380ms→354ms（~1.07x）**・全1045pass・
      構造監査0・カード挙動ベースライン不変。残: top-level transaction 単位の保守的無効化のため冗長 38.5% を全ては取り切れて
      おらず、decide スコープ無効化への精緻化で上積み余地あり（要・追加の健全性検証）。併せて `deep_diff` を型不一致
      （dict-vs-非dict 等）でクラッシュせず差分文字列を返すよう堅牢化（テストユーティリティ・実差分は従来どおり検出）。
    - **③ 置換表（transposition table）= 実測で不採用（2026-06）**: ②後は**健全（完全一致キー）な転置率 ≤0.5%**（exact key
      ＝デッキ順／全カード状態／継続効果まで含むと手順違いでも byte 一致がほぼ起きない）に対し、健全な位置キー計算が
      **~3%/node** のオーバーヘッド＝**ネット負**。②が per-node clone を消したため「再探索を省く」価値自体が消えた
      （clone が 86% だった頃の前提が無効化）。下記は不採用の経緯記録。
    - **（不採用）③ 置換表（transposition table）**: 同一盤面の再探索を省くキャッシュ（**当時の粗いキーでの**転置率 ≈23% が
      省ける上限と見積もったが、健全キーでは ≤0.5% と判明）。
      **内容ベースの一意ハッシュ**（uuid は毎回変わるため不可・**デッキ順／継続効果／隠れ情報**まで漏れなく＝
      hard は相手手札も含む）→ `ハッシュ→(評価値, 深さ, 最善手)`。ノード入口で probe（同深さ以上ならカット）・
      出口で store。リスク＝ハッシュ取りこぼし／衝突による**誤った値の再利用**（②の照合と同種の網羅性問題）→
      状態網羅テスト＋衝突対策が必須。
      期待＝実効 ~1.2〜1.3 倍（上限23%−ハッシュ計算オーバーヘッド）。いずれも**挙動・カード挙動ベースライン
      不変が必須ゲート**（探索の内部最適化であって手の評価結果は変えない）。
    - **④ 着手順序の前回 PV 再利用（move ordering by principal variation／killer move）= 実装済み（2026-06・粒度a）**:
      ③置換表が「同一**盤面の値**を使い回す」案で**健全キーのコスト>節約（≤0.5%）でネット負**だったのに対し、本案は
      「同一/類似局面で**良かった手（の順番）**を使い回す」案＝**盤面キー不要**（覚えるのは move の signature だけ）で
      置換表の致命的コストを回避する。狙いは α-β の枝刈り効率向上: α-β は**良い手から先に試すほど早くカット**でき、
      探索ノード数が減る＝**同じ予算（clone 回数）でより深く読める**（horizon4＋へ）。**手の評価結果は変えず**（α-β の
      返り値は探索順序に依らず不変）、変わるのは探索順序だけ＝②make/unmake と同種の**純粋な内部最適化**。
      - **2 つの粒度**: (a) **decide 内（反復深化）**＝浅い探索で得た PV を深い探索の着手順序の先頭に置く。
        (b) **連続する decide 間（同一ターン内）**＝手 N の探索で得た PV は、盤面が 1 ply 進んだだけの手 N+1 でも
        PV の末尾がそのまま良手であることが多い＝前回 PV を次手の着手順序ヒントに引き継ぐ（毎手ゼロから探索する現状の
        ムダを削る。①計画キャッシュが「線形セグメントを前倒し」なのに対し本案は「探索の中身を安くする」＝相補）。
      - **実装ポイント**: `_search` の `children.sort(...)`（ビーム選別の並べ替え）と `_scored_search` の
        `order = sorted(prelim...)`（ルート深掘り集合の選別）に **PV/killer 手を先頭へ寄せるバイアス**を足す。
        PV は (player, ply) または盤面の軽量シグネチャ→best move の小辞書で保持（衝突しても**順序が変わるだけで誤った値の
        再利用は起きない**＝置換表のような健全性リスクが無い＝低リスク）。
      - **正しさゲート（必須・挙動不変）**: 選ぶ手・深掘りスコアが PV 並べ替えの有無で**完全一致**すること
        （`tests/test_cpu_make_unmake.py` と同型の等価照合を PV ON/OFF で実施）＝方策・カード挙動ベースライン不変。
        並べ替えで候補の**欠落/重複が起きない**ことも固定（候補集合は不変・順序のみ変化）。
      - **効果計測**: 探索ノード数の削減率（同一局面で PV ON/OFF のノード数比較）＋ **A/B 自己対戦で強さ非退行**
        （ノード削減で浮いた予算を深さに振った場合に ≥ 現状であること。退行が無ければ深く読める分だけ強くなる）。
        期待＝着手順序改善による α-β カット増＝ノード数 1.x 倍削減（局面依存）。②（安価なノード遷移）と相乗する。
      - **実装（2026-06・粒度a＝decide 内）**: `cpu_ai._USE_PV_ORDER`（既定 True・`_USE_MAKE_UNMAKE` と同型で即時 OFF 可）。
        `_search` に ply→直近カット手 signature 列の `killers` 辞書を通し、ビーム選別**後**に `_pv_reorder` で killer 手を
        先頭へ寄せ（集合不変・安定並べ替え）、`alpha>=beta` カット時に `_record_killer`（MRU・上限 `_KILLER_SLOTS=2`）で登録。
        killer は ply 単位の汎用ヒントなので深掘りルート手間で共有（`_scored_search` が辞書を生成）。**正しさ＝挙動不変**を
        `tests/test_cpu_pv_order.py` で機械照合: (1)`decide` 選択手が ON/OFF 一致（normal/hard・多 seed）・(2)`_scored_search`
        深掘りスコアが ON/OFF 完全一致（既定予算300＋増量予算2000）・(3)ノード数が ON≤OFF かつ合計で削減・(4)killer/reorder の
        単体不変条件（集合不変・MRU・上限）。**実測（中盤6局面・増量予算で純カット効果）= 探索ノード −2.4%（最大 −6.5%・悪化0）**。
        全1056pass・構造監査0・カード挙動ベースライン不変。予算固定（=本番）では節約ノードが深さ（settle 回避）に回る形で効く。
      - **粒度b（連続 decide 間の引き継ぎ・実装済み＝既定 OFF・2026-06）**: `decide`／`_scored_search` に `killer_state`
        を追加し、`decide_guarded` が `mem["killers"]`（ターン内で持ち越し・ターン跨ぎは破棄）を供給して **killer 表を
        連続する decide 間で再利用**する。配線は挙動不変（reorder＝集合不変＝予算非拘束なら値不変＝
        `tests/test_cpu_pv_order.py::test_pv_cross_decide_invariant` で手順一致を機械照合）。**だが実測で中立〜わずかに悪化**
        （中盤6局面・増量予算でノード **-0.4%**＝悪化／ply を 1 ずらす整流でも ±0%）。原因＝killer は局面固有のヒントで、
        decide をまたぐと探索ルート（盤面）が変わり同 ply の局面が一致しない（単一探索内の兄弟部分木のような構造類似が無い）。
        よって **`_USE_PV_CROSS_DECIDE=False`（既定 OFF）**＝粒度a のみ本番有効。将来 **PV 継続**（最善応手列を次手の同一局面
        ノードへ正しく対応付ける＝ply 整合の本質的解決）で正の利得が出れば有効化を再検討する（土台＝`killer_state` 配線は完了）。
  - **体感最適化（perceived latency・Phase 3・計画）**: 観点を per-call レイテンシ単体から「**ゲームを通して
    不快がない（体感）**」へ拡張する設計方針。相手の手を織り込む（「〜されたら〜する」＝contingency）ぶん
    計算は増えるため、**両面**（計算を前倒し/再利用して隠す＋相手分岐を広く読む）で攻める。
    - **前提＝raw 速度軸はほぼ頭打ち**: デプロイは PyPy（方式B・`OPCG_PYPY_WORKER=1` で**全 decide を PyPy ワーカーへ
      オフロード**・~2.1x・§0/Dockerfile）。エンジンは **stdlib-only/純Python**で PyPy の JIT が効くよう作ってある。
      ゆえに **Cython/C 化は不採用**（PyPy 非互換＝cpyext 経由でむしろ遅い）・PyPy カバレッジ拡大も全 decide が既に
      PyPy 済で残りは JIT 親和の純Python維持のみ。raw 速度の伸びしろは小さい＝**前倒し/再利用が主軸**。
    - **① 計画キャッシュ（contingency 部分木の抽出＆ルックアップ）**: 探索は既に **min/max で相手の応手を
      読んでいる**（「相手がこう来たら自分はこう」が探索木に含まれる）。現状は 1 手選ぶと木を捨て**毎手再探索**して
      いるが、最善の**部分木（相手応手分岐込み）を抽出してキャッシュ**し、相手が実際に動いたら該当枝をルックアップ
      ＝即応する。ターン開始で 1 回深く考え以降は即時。メインフェイズ（相手介入なし）は完全キャッシュ可、攻撃は
      相手のブロック/カウンターで前提が崩れた枝から再探索。**決定的結果の前倒し/再利用＝挙動不変**。下記 ④⑥ が乗る土台。
      **PoC 実装済み（`plan_turn`・2026-06）**: 相手の介入（ブロック/カウンター）が入るまで、または TURN_END までの
      自分の連続手番を**クローン上で `decide_guarded` を逐次適用して計画**する。同じ単一 rng ストリーム・同じ mem を渡せば、
      介入の無い区切り内では各手番の (盤面, rng, mem) が完全一致＝**per-action 流とビット等価**（同じ手列・同じ rng/mem
      消費。前倒しで全部計算→以降 replay という時間配分だけが違う）。実測（hard 自己対戦・seed0-1）で **90/90 セグメントが
      ビット等価・不一致 0**（`tests/test_plan_cache.py` で機械照合＝強さ・再現性を変えずに導入できる）。レイテンシ分布:
      セグメント長 mean 2.1・複数手(≥2)セグメント平均 3.1 手で **plan_turn 平均 ~1054ms（=待ちを 1 回に集約）／最大 ~4.3s**＝
      総計算は per-action と同じで**1 回の思考に前倒し**。長いセグメントの単発待ちは大きくなるため、前倒し分は
      ④（演出重ね）/⑥（ポンダリング）で隠す＝計画キャッシュはその土台。
      **配線（`decide_cached`・2026-06）**: 対局ごとに `cache={"queue":[...]}` を保持し、(a) 次の計画手が現局面で**合法なら
      即返す**（探索なし＝即時 replay）、(b) ミス/前提崩れなら `plan_turn` で再計画。**合法性検証で常に安全**＝rng がズレても
      キャッシュ手が不正なら通常 `decide_guarded` に落ちる（不正手は打たない）。ただし `plan_turn` のクローン適用と本番
      replay の実適用で**シャッフル効果等の rng 消費が前後しうる**（`random.shuffle` ＝デッキ操作。`gamestate` 2334 等）ため
      **決定性は保証しない＝テスト/自己対戦は `decide_guarded`（フラグ OFF 既定）・本番のみ `decide_cached`**（本番は決定性
      不要）。検証 `tests/test_plan_cache.py`（合法手のみ・対局完走・plan_turn 呼数 < decide 回数＝replay 実効）。
      **残**: API 層（`/cpu/step`・`CPU_GAMES`）と PyPy ワーカーへの cache 配線（保持＋シリアライズ）・A/B。
    - **② 自明手の即答パス＝実測で不採用（2026-06）**: 「1-ply で best が 2nd を大差（≈12000）で圧倒する自明局面は
      深掘りを省いて即答」を試作（`_scored_search`・`OPCG_TRIVIAL_FASTPATH` フラグ）。**A/B（hard・席交互24局）= ②ON
      14-10（58.3%）＝強さは中立（ノイズ範囲）**だが、**思考手レイテンシ ON 368ms vs OFF 364ms ＝利得ゼロ**で不採用。
      理由: 深掘りをスキップしても **prelim（全ルート手の 1-ply 事前採点＝`_eval_root_move`）は依然走る**ため節約に
      ならない（コストの大半は deepening でなく prelim＋eval）。中立・無利得・分岐の複雑性増のためコードは revert。
      `decide` は単一合法手は従来どおり即答する。下記は不採用の経緯記録。
    - **③ 相手 min 手番のビーム幅拡張（E1・実装済み・2026-06）**: 従来 `HARD_BEAM`(=3) を max/min 共通で適用
      （`children[:HARD_BEAM]`）していたのを、**min（相手）ノードだけ `HARD_OPP_BEAM`(=4) に拡げる**
      （`children[:(HARD_BEAM if is_max else HARD_OPP_BEAM)]`）。自分(max)は最善へ収束させたいので狭く、相手(min)の
      応手（ブロック/カウンター/除去応答/相手ターンの攻め手）を多く残して contingency（「相手がこう来たら」の枝）を
      厚くする。**A/B 自己対戦（asymmetric arena・席交互・hard）**: beam=4 vs 3 で **24 局 14/24＝58.3%＝+58 Elo**
      （別シード 8 局でも 5/8＝+89 Elo）＝広い相手ビームが明確に強いことを実測。**レイテンシ**: 値決めスイープで
      beam=5 は深掘り clone 増で 1 秒目標を超過（p95 ~1009ms / max ~1285ms）＝不採用、**beam=4 を採用**（クリーン計測で
      mean ~551ms / p95 ~1021ms / max ~1283ms / 1 秒超 ~7%＝目標 1 秒を稀に超える程度・体感は ④/⑤ で吸収予定として
      現状許容）。回帰 `tests/test_cpu_puzzles.py::test_e1_opp_beam_widens_min_node_independently_of_max`（min 幅が
      `HARD_OPP_BEAM` に従い max とは独立であることを決定論的に固定）。
    - **④ アニメーションと探索の重ね合わせ**: CPU の行動 N を演出している間に N+1 を計算＝思考を演出の裏に隠す
      （① があれば演出中はキャッシュ参照でほぼゼロ体感）。frontend/プロトコル連携。
    - **⑤「思考中」演出**: 速度に依らず意図的な間＋キャラ性で不快感を下げるクッション（安い・安全・常時有効）。
    - **⑥ ポンダリング（相手の番に ① を前倒し）— 設計（2026-06）**: 人間の手番中に ① の計画を
      先行構築し、相手が確定したらルックアップ＝CPU 手番が即時。①（計画キャッシュ）が PoC＋API 配線済みのため、⑥は
      「ゼロから」ではなく「**①の plan 計算を CPU 手番のポーリング時から、人間の手番処理時へ前倒し（lazy→eager）**」する
      拡張＝決定性も合法性安全網も①から継承する。以下は実装前の設計。
      - **足場（アーキ調査済み）**: API は FastAPI async・**1 手 1 HTTP のポーリング**（`/api/game/cpu/step`＝CPU 手を 1 つ
        返す `game_cpu_step`／人間手は `/api/game/action`・`/api/game/battle`）。①は `app._cached_cpu_move` が
        `CPU_GAMES[gid]["plan_cache"]={"queue":[...]}` を消費し、`_move_sig` の合法性ゲートで stale を安全に通常 `decide`
        へ落とす。重い探索は **PyPy ワーカー（別プロセス・Unix ソケット）** にオフロード済み＝`asyncio.create_task` から
        ワーカー往復すればイベントループを塞がず「人間の手番中に裏で計算」が成立する。
      - **⑥-a 先行計画（eager・実装済み・2026-06）**: 契機＝人間アクション（`/api/game/action`・`/api/game/battle`）の処理末尾で
        pending が CPU 手番へ移った瞬間（`_kick_ponder`）。そこで `asyncio.create_task(_ponder_plan(gid))` を起動し、
        `plan_segment` を `asyncio.to_thread` で PyPy ワーカー（別プロセス）へオフロードしてイベントループを塞がずに計画、結果を
        `meta["plan_cache"]["queue"]` へ充填する。frontend が次に `/cpu/step` を叩く頃には①の `_cached_cpu_move` が**ヒット
        ＝CPU 初手が即時 replay**。①との差分は**起動タイミングのみ**（計算内容・合法性ゲート・決定性方針は同一）。**並行/競合回避**:
        `game_cpu_step` は計算前に走行中の先行計画タスクを `await`（warm な queue を使い二重計算を避ける）／二重起動は
        `plan_cache["task"]` で防止／イベントループ外（同期テスト）では `create_task` 不可＝no-op。フラグ
        `OPCG_PONDER=1`（①の `OPCG_PLAN_CACHE=1` 配下・既定 OFF＝従来挙動完全同値）。検証
        `tests/test_plan_cache.py`（`_ponder_plan` が queue を温め後続 `_cached_cpu_move` が再計算なしでヒット／
        `_kick_ponder` のゲートとタスク起動→完走で queue 充填）。**残**: ⑥-b（投機）と体感 A/B（prewarm 有無で /cpu/step 応答時間）。
      - **⑥-b 投機ポンダリング（実装済み・bounded v1・2026-06）**: 契機＝人間の MAIN 継続中（`/api/game/action`・
        `/battle` 末尾で pending が人間の MAIN_ACTION のとき `_kick_speculate`）。人間 MAIN の分岐は巨大（PLAY×対象×ドン
        配分×攻撃順…）＝全列挙不可なので、**「人間が今 TURN_END したら」の 1 本に限定**して投機する: `_speculate_compute`
        が**クローン上で**人間の TURN_END を仮適用し、pending が素直に CPU 手番へ移れば CPU セグメントを `plan_segment`
        （ワーカー）で計画して `plan_cache["spec_queue"]` に保持（介在する人間決定があれば None＝投機しない）。clone は
        **メインスレッドで原子的**に取り（読み書き競合なし）、TURN_END 仮適用＋plan だけを `to_thread` へ逃がす。使い捨て
        clone・使い捨て mem＝**live 盤面/turn_mem 不変**。人間がさらに動いたら世代トークン `spec_gen` を進めて旧投機を
        **supersede**（1 ゲーム 1 タスク＝本数ゲート）。人間が実際に TURN_END したら `_kick_ponder` が `spec_queue` の先頭を
        実盤面の合法手と照合し、**合法なら queue へ昇格＝投機ヒット（CPU 初手の待ちすら消える）**／外れは `spec_misses` を
        計上して ⑥-a（実盤面の先行計画）へ。採否は①の合法性ゲートが最終担保＝**外れても事故らない**。`spec_hits`/`spec_misses`
        で**当たり率を計測**できる（本採用＝既定 ON 化はこの実測を見て判断）。フラグ `OPCG_PONDER_SPEC=1`（⑥-a の
        `OPCG_PONDER=1` 配下・既定 OFF＝従来挙動完全同値）。検証 `tests/test_plan_cache.py`（`_speculate_compute` が
        clone 上で CPU 計画を返し live 不変／`_kick_ponder` が合法な spec_queue を昇格し `spec_hits` 計上／`_kick_speculate`
        のゲートと live 非変異）。**残**: 複数候補盤面への拡張（出力ゲート前提）。
      - **本番有効化と実測（2026-06）**: 計測ハーネス（p1=人間プロキシ・p2=CPU で本番フローを asyncio 駆動）で hard 自己
        対戦を計測。harness 上は ⑥-a 単独で CPU セグメント先頭の 100% が warm・無駄打ちゼロと出たが、これは**ハーネスが
        TURN_END 後にポンダリングタスクの完了を await（＝実在しない思考時間を ⑥-a に与えていた）ための過大評価**だった。
        実プレイの体感では ⑥-b が効く: 人間は TURN_END した時点でもう打ち終わっており frontend の初回 `/cpu/step`
        ポーリングが**すぐ来る**＝⑥-a の前倒し（TURN_END 起点）は間に合わず `game_cpu_step` がタスクを await してフル待ちに
        なりがち。人間が実際に考えている **MAIN 中**に先回りする ⑥-b で初めて CPU 手番の待ちが消える（実機観察で確認）。
        トレードオフ＝⑥-b は人間アクションごとに投機を焼き直しワーカー負荷が増える（当たり率 ~30%）が、単一の CPU 対戦
        では許容。よって**本番は ①＋⑥-a＋⑥-b を ON**（`OPCG_PLAN_CACHE=1`/`OPCG_PONDER=1`/`OPCG_PONDER_SPEC=1` を
        Dockerfile に追加）。即ロールバックは各フラグを 0/未設定に。テスト/自己対戦は env 非依存（同期 decide）で決定性を維持。
      - **状態キー（`CPU_GAMES[gid]["plan_cache"]`）**: `queue`＝残り計画手／`task`＝進行中ポンダリングタスク（二重起動防止・
        キャンセル用）／`base_sig`＝計画を立てた局面の指紋（人間が動いたら queue を無効化判定）。
      - **並行・破棄**: 1 ゲーム 1 タスク。次の CPU 手番までに未完なら `/cpu/step` 側で**通常 decide にフォールバック**（待たない）。
        人間ターン中に状態が変わる契機（追加アクション）が来たら旧タスクをキャンセルし queue を無効化（`base_sig` 不一致）。
      - **決定性**: ①と同じく「**decide の決定的結果を前倒しで計算しておくだけ**」＝決める内容は不変。**本番のみ**作動し
        （テスト/自己対戦は同期 `decide_guarded`・フラグ既定 OFF）、rng 消費の前後ズレは①の合法性ゲートが吸収（不正手は打たない）。
      - **テスト戦略**: 単体＝`prewarm_plan` が queue を充填し後続 `_cached_cpu_move` がヒット（plan_segment 呼数 < decide）／
        並行＝タスク未完時に合法 decide へフォールバック／破棄＝`base_sig` 不一致で queue 無効化→再計画／体感＝prewarm 有無で
        「CPU 初手の `/cpu/step` 応答時間」を A/B（本番フラグ）。決定性経路（同期）は不変＝全既存テスト影響なし。
      - **フラグ**: `OPCG_PONDER=1`（①の `OPCG_PLAN_CACHE` 配下・既定 OFF）。①②④が固まってから ⑥-a→⑥-b の順で着手。
    - **決定性の維持（設計制約）**: 前倒し/キャッシュ/ポンダリングは「**decide が出す決定的結果を先に計算しておく
      だけ**」に留め、決める内容は変えない＝テスト・自己対戦の再現性は不変（壁時計デッドラインのような非決定化は
      採らない）。WBS「Phase 3: 体感最適化」に課題登録済み。
  - **効果ターゲット選択を探索分岐へ（`_selection_moves`）**: 効果の**単一対象選択**（KO／除去／バウンス／
    手札破壊／場溢れトラッシュ＝ `SEARCH_AND_SELECT`・最大1体）は、`_drain_own_interactions` で既定解決
    せず**候補ごとの手に展開して探索する**（`stop_at_select`）。これにより「相手のどのキャラを除去するか」
    という最もインパクトの大きい意思決定を α-β／ビームで読み切る（対局で効く選択を最適化）。任意選択は
    「選ばない」も一級候補。候補数は `HARD_SELECT_CAP` で安全に上限。多対象（最大2体以上）・min>1 は
    組合せ爆発回避のため既定解決へ委ねる（継続テーマ）。`easy` は素の 1-ply のまま（分岐なし）。
- **難易度＝1 系統**（`easy`/`normal` は **2026-06 に廃止**。`expert`(MCTS) も撤去。下表へ集約）:
  | UI | キー | 方策 | 相手情報 |
  |---|---|---|---|
  | つよい | `hard` | 単ターン先読み α-β＋ビーム＋**PIMC（決定化・§2.5.8）**。葉評価は **L1（カード通貨）単一系統＝`cpu_eval_v2.evaluate_v2`**（手書き J値評価は **2026-06-27 に撤去**／学習価値の葉ブレンドは **2026-06-28 に撤去**／自デッキ勝ち筋プラン§2.5.5 は全廃＝plan 非供給のフラット評価） | **公平**（相手の伏せ手札を覗かず K 通り決定化して平均＝§2.5.8。`cheat` は診断用に明示指定のみ） |

  > **廃止経緯（2026-06・ユーザ判断）**: `easy`（1-ply 貪欲）と `normal`（リーダー推測の相手モデル§2.5.4＋
  > 公開情報のみ先読み）を削除。一時は **α-β=`hard` と MCTS=`expert` の2系統**としていたが、
  > **`expert`(MCTS) は強さ・速度で α-β に劣ると判明し撤去（2026-06）**＝出荷 CPU は `hard`（α-β）の単一系統。
  > `decide` 系は hard 方針に一本化（公開 API の `profile` 引数も撤去）。`normal` 専用だった相手モデル
  > （§2.5.4 のリーダー推測・`/api/cpu_template` 画面）も併せて廃止（`cpu_opponent_model.build_profile` は
  > 自デッキ分類＝§2.5.5 で使うため残置）。
- **sim 専用の対話自動解決** `default_interaction_payload`: 先読み中に `active_interaction` /
  `pending_request` が立った場合の機械的な既定確定（対象=ヒューリスティック最良 or 先頭、CONFIRM=
  有利なら使う 等）。**これはクローン上の先読み専用**であり、本番（実対局・自己対戦）の未解決中断は
  握り潰さず [`docs/TEST_SPEC.md`](TEST_SPEC.md) §3.1 のインバリアントで表面化させる（AI が「とりあえず
  動く」ことで効果バグを覆い隠さないための分離）。
- **公平性（2026-06 更新）**: 出荷 `hard` は**フェア**＝相手手札の中身・裏向きライフを読まない
  （`info_policy=fair` 既定＋PIMC 決定化・§2.5.8）。`cheat`（旧 hard＝相手手札を
  読む）は凍結ベースライン/診断用に明示指定でのみ到達。`see_opp_hand=False`＋相手 min ノードの手札依存手を
  除外でフェアを担保（§6 の視点マスクは探索には適用しない）。

### 2.5.3 精度向上（実装済み／継続テーマ）
J値理論ベースの評価関数・ターン境界評価の α-β＋ビーム探索（B1 horizon=1／**B2-lite horizon=2＝守りの
深読み**）・最短リーサル認識を実装済み（§2.5.2）。さらなる強化の継続テーマ:

- **探索の高速化・深化**: 置換表（メモ化）・move ordering の改良で予算内の読みを深める
  （`HARD_PER_MOVE_BUDGET` はレイテンシ予算・切れても `_settle_eval` で境界評価）。横展開（深掘り対象
  `HARD_ROOT_BEAM`）や horizon の拡大もここで検討。`decide_guarded` の収束保証は維持する。
- **評価関数の高度化**: 重みの学習／チューニング、効果連鎖（チェイン）・盤面テンポ・相手のリーサル
  （被削り切り）認識の織り込み。
- **局面別ヒューリスティック**: 攻め／受けの切替、ライフ・ドン・手札リソースのトレードオフ評価。
- **検証**: 改善は §3.1 の CPU 自己対戦ハーネスで決定論・インバリアントを保ちつつ、難易度間の
  勝率（弱 < 強）で精度を回帰確認する。自己対戦＋インバリアントは自己参照的で特定症状（例: 余剰ドン
  温存）に信号が出ないため、**パズル/シナリオ回帰集**（正解手種が既知の局面・`tests/test_cpu_puzzles.py`）・
  **凍結ベースライン Elo**（固定参照相手への挑戦者勝率＝絶対強度・`tests/cpu_arena.py`）・**regret ログ**
  （崖エラーの安価な代理・`cpu_ai.decide_with_regret`）を併設する（下記「2026-06 外部レビュー収束」・全て実装済み）。

#### 改善バックログ（外部レビュー由来・着手用）
2026-06 の外部AIレビューで挙がった改善案を、**効果大×実装軽い順**にバッチ化したもの。各項目は
「対象 → やること → 重大度」。実装後は本節から該当項目を消し、§2.5.2/§2.5.6 等の本文へ吸収する。
WBS（`gx5gyqe2-art/WBS` の `projects/opcg-sim-backend.md`）と同期。
> **【2026-06-27 注記】** 本バックログは手書き J値評価（`_side_score` ＋重み定数群）時代の検討記録。
> CPU 評価はその後 **L1 単一系統（`cpu_eval_v2.evaluate_v2`）へ集約**され、`_side_score`/`W_LIFE`/`W_LIFE_LOW` 等は
> 撤去済み（§2.5.2 撤去バナー）。以下の `_side_score`/`W_*` 参照は当時の設計記録（履歴）。
> 裏取り済みの注記: ① min ノードは `children.sort(reverse=is_max)` で root最不利手をビームに残す実装済み
> （対象外）。② 逆算 reach の「false lethal」は本物のリーサルが探索の `winner` 到達で弾かれるため
> soft 精度改善（C-1）へ格下げ。③ **クロック（相手ライフ）は実装済みの最重量項**: `evaluate` は両側の
> `_side_score` 差分で、相手ライフも `W_LIFE`（＋薄域 `W_LIFE_LOW`）で最重量に評価される。よって
> **独立クロック項は二重計上のため不採用**（B-1 の「余剰ドン温存」はクロック未評価ではなく、後述の
> アイドルドン床のタイブレークが真因）。

**バッチA（軽量・高ROI・低リスク）**
- **A-1 アンブロッカブル／「効果で選ばれない」の評価【撤去 2026-06-27】**（`_threat_value`／
  `_is_unblockable`・§2.5.6）: 自デッキ勝ち筋プラン全廃に伴い削除（plan 供給時のみ作動の plan-gated 機能だった）。以下は履歴。
  アンブロッカブル【ブロック不可】を脅威項に加点（`W_KW_UNBLOCK=900`・両側対称・
  プラン供給時）。`keywords` に "ブロック不可" が載らないため自前キーワードのテキスト `【ブロック不可】(…)` で
  検出し付与句 `…を得る` と区別。**「効果で選ばれない」（対象保護）は現カードプール 0 枚＝表現未確定のため保留**
  （出現時にテキスト/フラグ検出で追加）。`tests/test_cpu_puzzles.py`（A-1）。
- **A-2 `_threat_value`・`_ACT_MARGIN` のアーキタイプ依存スケール【撤去 2026-06-27】**（`cpu_self_plan`／`cpu_ai`・
  §2.5.5/§2.5.6）: 自デッキ勝ち筋プラン全廃に伴い削除（脅威スケール・act-margin 乗数とも plan-gated）。以下は履歴。
  `PlanProfile` に係数 `threat_atk_mult`／`threat_def_mult`／`act_margin_mult` を追加し評価へ供給。
  `_threat_value(c, atk_mult, def_mult)` は攻撃的キーワード（ダブルアタック/速攻/バニッシュ/アンブロッカブル）を
  `threat_atk_mult`、防御的キーワード（効果耐性「KOされない」）を `threat_def_mult` でスケール（両側対称）。
  `decide` は畳み判定マージン `_ACT_MARGIN` を `act_margin_mult` でスケール。プリセット: **aggro**＝攻め係数 1.30／
  守り係数 0.85／マージン 0.6（テンポ攻めを通す）、**control**＝攻め 0.85／守り 1.25／マージン 1.5（曖昧な展開は
  畳んで守りを残す）、**midrange/NEUTRAL**＝全 1.0（plan 無しと完全同値）。`tests/test_cpu_puzzles.py`
  （脅威スケールの交差不変／プリセットの方向性）。
- **A-3 フェア性ガード＋探索健全性テスト【実装済み】**（`tests/test_cpu_puzzles.py`）: normal 探索が相手の
  隠しゾーン（相手手札の中身・裏ライフ）を一切クエリしないことの assert/テスト（evaluate-spy で
  `see_opp_hand=False` 固定＋相手手札の中身に選択不変）に加え、**min ノードのビーム剪定が root 最不利側に
  偏る**（`children.sort(reverse=is_max)` ＝ min では 1-ply 評価キーの小さい＝root 最不利な子を先頭に残す）
  ことの回帰を追加。**所見**: ビーム剪定は 1-ply 評価をプロキシに使うため、剪定で残るのは「1-ply で最不利に
  見える」子であり、深い値での真の最不利とは前後し得る（min 応答数が `HARD_BEAM=3` を超えるとき＝主に
  horizon=2 の相手フルターンで顕在）。実 min ノード（SELECT_BLOCKER/SELECT_COUNTER）は応答数が beam 以下で
  剪定が起きず健全。sort 方向が optimistic 側へ反転していないことを locking した（重大度=低）。

**バッチB（核心・最重要量＝5000/壁の閾値）**
- **B-1(a) 余剰アクティブドンの末端減価【撤去 2026-06-27】**（`_side_score`／`W_DON_ACTIVE`／
  `cpu_self_plan.idle_don_mult`・§2.5.5）: 自デッキ勝ち筋プラン全廃に伴い削除（`idle_don_mult` は plan 供給時のみ作動）。以下は履歴。
  葉（`is_turn=False`）の浮いたアクティブドンを `idle_don_mult`(<1.0)
  で減価＝「両枝でクロック同値→ドンの床でタイブレーク→握る」を断つ。**plan 供給時のみ作動・`plan=None`
  完全同値**。`tests/test_cpu_puzzles.py`（idle のみ減価／配線／プリセット順）。
- **B-1(b) カウンター強要〔推定カウンター応答モデル〕【撤去 2026-06-27】**（`_search` min ノード・
  `_estimate_counter_buffer`／`_counter_needed`／`_apply_modeled_counter`・§2.5.2/§2.5.4）: CPU 評価の L1 単一系統化に
  伴い、相手モデル `profile` ベースのカウンター緩衝推定（`_estimate_counter_buffer`）と modeled-counter min 応答を撤去。
  以下は廃止前の設計記録（履歴）。**静的クレジット
  （有効パワー上限を `対象防御＋緩衝` まで引き上げ）は不健全と判明**＝付与ドンは相手ターン（探索の葉）で
  `get_power(False)` がパワーを乗せないため上限を上げても葉に伝播しない。カウンター強要の価値は**相手が
  実際にカウンターする**ときのみライフ温存差として現れる。そこで normal の保守 min ノード（`opp_public_only`・
  従来は手札カウンターを全除外＝相手は決してカウンターしない）に、**相手手札の中身は読まず**（フェア）
  リーダー推測 profile（§2.5.4）の**カウンター密度 `counter_avg`** から推定した緩衝 power を上限に、
  SELECT_COUNTER で「`counter_buff` を needed 加算＋手札 1 枚消費（枚数のみ＝公開情報）＋PASS」の応答を
  PASS と並べて出し **min に選ばせる**。緩衝内は相手が守り切る（盛っても無駄）／緩衝超で貫通（余剰ドンを
  攻撃に振るのが正の手）。`counter_budget` を探索パスで逓減し過剰カウンターを防ぐ。**hard は
  `opp_public_only=False` で実カウンターを既に読むため非作動**。`profile` 無し／`plan=None` 完全同値。
  `tests/test_cpu_puzzles.py`（緩衝推定の単調性／ライフ温存＋手札消費／配線スモーク）。
- **公開情報ベリーフ更新【撤去 2026-06-27】**（`_estimate_counter_buffer(profile, opp_hand_size, opp_trash)`・
  `_scored_search` で `_other(manager,name)` の生の手札枚数・トラッシュを供給）: B-1(b) の撤去（profile ベース eval
  補正の削除）に伴い消滅。以下は廃止前の設計記録（履歴）。B-1(b) の推定カウンター緩衝を
  **静的テンプレ密度のままにせず、対局中に公開された情報だけで belief を更新**する。(1) **手札枚数**（公開
  count）でコミット想定枚数をキャップ＝相手が手札を吐いて少なくなるほど緩衝が縮む（0 枚＝0＝守れない）。手札の
  *中身*は読まない＝フェア。(2) **トラッシュ**（公開）に見えた消費カウンター値ぶん、テンプレ基準の総カウンター
  power（`counter_avg×n_cards`）に対する残密度を割り引く＝カウンターを使い込むほど緩衝が縮む。これにより
  aggro が手札を吐き切った局面では CPU の攻撃が通りやすく（緩衝小）、control が手札を抱える局面では強要に多めの
  ドンが要る（緩衝大）と、盤外の公開状況に応じて自然に変わる。引数省略時は静的既定（従来値）＝後方互換。
  `tests/test_cpu_puzzles.py`（手札枚数追従／トラッシュ消費による割引）。
  > 実プレイ報告（2026-06・症状の記録）: 自デッキにカウンターイベントが無いのに CPU が**余剰アクティブドンを攻撃に
  > 振らず温存**する（normal/hard 双方）。原因＝(1) 過剰パワーが×0.1で無価値、(2) normal は相手がカウンター
  > を切らないモデル＝盛っても結果が変わらず価値ゼロ、(3) 守りに使えない余剰ドンに価値(200/枚)を置いている。
  > → 本項のカウンター強要クレジット＋「**守りにドンを使えないデッキでは葉評価の余剰アクティブドンを減価**」で是正。
  > 計測: 自己対戦の余剰ドン平均は normal 0.4 / hard 0.2＝**頻度は低く特定局面で発生**（手札無し・展開不可で
  > リーダー/一部攻撃者のみ・相手が守らない局面）。**最初に直した「5000未満へ無意味にドンを振る」バグと方向が
  > 逆**なので、クレジットは「**このターン実際に相手の意味ある対象（リーダー/到達できるキャラ）へ攻撃する体**」
  > に限定・上限付きで付与し、無意味な体への過剰盛り再発を防ぐ。正当な浮かせ（終盤・ブラフ・特定キャラ温存）は
  > 別途の温存ロジック（小課題）。検証はピンポイント回帰（normal が攻撃者/リーダーに余剰ドンを振る）＋
  > 既存ゲート（無意味盛り非増加・守りのブロッカー温存維持・弱<強）。
  > ▼2026-06 外部レビュー収束（機序の精密化）: **両枝とも攻撃が貫通する局面では、クロック項（相手ライフ減）が
  > 両枝で同値**となり差を生まない。残る差は末端のみ＝盛る側は過剰パワー×0.1≒0＋ドンが `don_active` から
  > 外れて `W_DON_ACTIVE`(200/枚)を失う／握る側はアクティブのまま +200/枚。よって**ドンの床だけが
  > タイブレークして「握る」を選ぶ**。クロック未評価が原因ではない（裏取り③）。したがって主役は
  > カウンター強要よりも**アイドルドンの末端減価**で、症状は「ドンをクロックに変換」パズル（下記検証基盤）で固定する。
- **B-2 ドン付与の手生成を意味ある配分のみに限定【実装済み】**（`_prune_don_moves`／`_attach_don_meaningful`・
  §2.5.3）: CPU の探索/方策（`decide`／`decide_with_regret`／`_search`）が `get_legal_actions` から得た手集合の
  うち **ATTACH_DON を「意味ある配分」だけに絞る**（エンジンの合法手列挙＝人間プレイには手を入れない）。
  意味あり＝(A) 付与先（このターン攻撃できる体）が現状では上回れない相手の防御パワー（リーダー/場キャラ）を
  手持ちアクティブドンの範囲で**新たに上回れる**（`p < tp <= p + budget*1000`）／(B) 付与ドン条件【ドン!!×N】を
  開けるカード（戦闘閾値に関わらず保守的に残す＝don 条件効果の起動を潰さない）。過剰（オーバーキャップ）・
  全ドンでも届かない・レスト/召喚酔いの素体への付与は落とす。ビーム3を意味ある配分へ集中＝手生成側の
  組合せ爆発を抑制。`tests/test_cpu_ai.py`（閾値判定／overcap・レスト除外／don 条件残し／非ドン素通し／
  検出器の実カード一致）。重大度=中。
- **無駄攻撃の除外【実装済み】**（`_prune_futile_attacks`・`decide`/`decide_with_regret`/`_search`・§2.5.3）:
  **攻撃側の有効パワー < 対象の有効パワー**（キャラを KO できない／リーダーへライフを取れない）かつ
  **【アタック時】効果を持たない**攻撃を CPU 候補から落とす。無駄攻撃は攻撃者をレストにするだけで何も
  達成せず（相手は防御不要なのでカウンターも強要できない）にもかかわらず、探索は「自ターンが続く＝攻め圧
  `W_ATTACKER` ぶん」TURN_END より高く評価していた（2026-06-19 報告: ナミ OP11-041 の【相手のアタック時】
  +2000 で相手リーダーが 7000 になり、自軍 5000 が顔に届かない局面で、CPU が倒せないニコ・ロビン 8000 へ
  無駄攻撃）。**現在の有効パワーで届かない攻撃のみ**落とす＝届かせるドン付与は別手(ATTACH_DON)として残り
  付与→攻撃の貫通筋は不変。【アタック時】持ち（カタリーナ OP16-104 等）は効果が目的になり得るため残す。
  CPU の探索/方策のみ（人間プレイは無駄攻撃も自由）。`tests/test_cpu_ai.py`
  （`test_prune_futile_attacks_keeps_reachable_drops_unreachable`）。重大度=中。
- **ドン!!返却（ドン-N）のテンポ損を追加減点【実装済み】**（`_don_return_penalty`・`_scored_search`・§2.5.3）:
  アクティブドンをドンデッキへ戻す手は当面の盤面形成力（将来の手出し・ドン付与の上限）を下げるテンポ損
  だが、静的 eval の `W_DON_ACTIVE`(200) だけでは過小評価で、**序盤に 2 ドン戻して軽微な効果を撃つ**不自然手を
  招いた（2026-06-19 報告: 万雷 OP15-078 の【メイン】ドン!!-2＝ドロー+相手キャラのレストを序盤に発動）。
  root 手で actor が**正味で戻したドン枚数**（手の後にドンデッキが増えた分＝紫のドンランプ等の再追加で正味
  増えない手や、ドンデッキから場へ足すランプ手は対象外）× `_W_DON_RETURN`(600) × 序盤係数（残ドンデッキ/10）で
  prelim/deep 双方を減点。終盤ほど軽く、見返りの大きい返却（リーサル設定・強力除去）は eval 利得が上回るので
  従来どおり選ぶ。CPU の手選択のみで eval/合法手列挙は不変。`tests/test_cpu_ai.py`
  （`test_don_return_penalty_scales_with_returned_and_early`）。重大度=中。
  > **戻すドンはレスト優先（既定解決）【実装済み】**（`_suspend_for_don_selection`・2026-06-21 エネル OP15-058 報告）:
  > ドン返却（RETURN_DON）の対象選択は **CPU 探索の対象外**（`_selection_moves` は `SELECT_RESOURCE` を列挙せず常に
  > `default_interaction_payload` で既定解決＝候補先頭から min 枚）。候補順が active 先頭だったため、**レストが有るのに
  > 一番使えるアクティブドンを戻す**無駄をしていた。候補順を **レスト→アクティブ→付与中**（戻す損の少ない順＝レスト=
  > 今ターン消費済み＞アクティブ=今ターン使える＞付与中=+1000 を失う）に変更し、既定解決がレストを戻してアクティブを
  > 温存する。副次効果＝EB02-010（ドン‼-2→2枚アクティブ化）はコストがレスト返却になり最終 active が +2 多くなる（正着）。
  > `tests/test_return_don_priority.py`。重大度=中。
  > **併せて lethal 認識の ply 割引を一貫化**（`_settle_eval(ply)`）: 予算切れ settle で勝者を観測した長い手順が、
  > winner 検出（`W_WIN-ply`）の直接の止めより生 `W_WIN` で高く見える不整合を修正（最短の止めを優先）。B-2 の
  > プルーニングでビームが lethal 手順を拾いやすくなり露見した潜在不整合。`tests/test_cpu_puzzles.py`
  > （`test_puzzle_takes_lethal_on_open_opponent`＝直接アタックを ATTACH_DON より優先）が回帰ガード。
- **B-3 深掘り集合に重要手を強制投入【強制投入は実装済み・拡幅は置換表待ち】**（`_scored_search`／
  `_is_important_root_move`）: ブロッカー設置・除去候補（単一対象選択の RESOLVE）・逆算リーサル/クロック手
  （相手リーダーへのアタック＝戦闘応答待ちでライフ未減なのでターゲットで判定／効果で即時ライフ減）は
  1-ply ランクに関係なく深掘り集合へ強制投入する（上限 `HARD_FORCE_DEEPEN_CAP=3`・child 再利用で追加
  クローン無し）。1-ply の浅さで守備 setup／止め手を落とす取りこぼしを是正。**`HARD_ROOT_BEAM` の 4→6〜8
  拡幅は置換表によるレイテンシ削減が前提のため未実施**（強制投入のみ先取り・中盤 decide 実測 ~0.8s）。
  `tests/test_cpu_ai.py`（分類器の単体＋ビーム0でもクロック手が深掘りに残る統合）。重大度=高。

**バッチC（敗着リスク低減）**
- **C-1 逆算 reach のブロッカー/カウンター控除【撤去 2026-06-27】**（`_plan_progress`・§2.5.5）: 自デッキ勝ち筋プラン全廃に伴い `_plan_progress` ごと削除。以下は履歴。reach 本数から
  相手の可視ブロッカー数（各 1 本を止める）を減算し、隠れ分は `profile`（公開情報ベリーフ更新済み）の
  推定カウンター緩衝 power を `_COUNTER_SAVE_UNIT=2000` でセーブ回数化＋相手手札枚数で上限して控除、
  **割引後 reach** で止め（`_CLOSER_W`）/接近（`_NEAR_W`）を判定（false lethal の soft 精度改善）。`profile`
  無しは控除 0＝従来どおり（plan 単体テスト不変）。`evaluate` が opp 側 profile を `_plan_progress` へ供給。
  `tests/test_cpu_self_plan.py`（可視ブロッカー控除／カウンター緩衝控除）。重大度=中。
- **C-2 テレグラフ致死の減点【撤去 2026-06-27】**（`evaluate`／`_telegraph_lethal`・
  §2.5.3）: 自デッキ勝ち筋プラン全廃に伴い `_telegraph_lethal`/`W_TELEGRAPH_LETHAL` ごと削除（plan 供給時のみ作動だった）。以下は履歴。相手ターン開始の静止点（is_my_turn=False・相手の攻撃が目前）の葉で「相手の次ターン有効打点
  （リーダー＋場の素パワーが自リーダーに届く本数 − 自アクティブブロッカー）≥ 自残ライフ」なら
  `W_TELEGRAPH_LETHAL=6000` を減点。**W_WIN(1e9) に対し十分小さく、本物のリーサル発見（±W_WIN）は決して
  上書きしない**＝引き分け帯で守り（ブロッカー温存・脅威除去・ライフ獲得）へ寄せるだけ。打点見積りは素パワー
  （保守的＝過剰防御回避）。プラン供給時のみ作動（plan=None 完全同値）。**低ライフ時の適応 `horizon=3` は
  予算（レイテンシ）増を伴うため未実施**（減点項のみ先取り）。`tests/test_cpu_self_plan.py`（検出ロジック／
  項の isolate）。重大度=中〜高。
- **C-3 自他ライフの別カーブ＋膝位置の対面依存【撤去 2026-06-27】**（`_side_score`／`_own_life_knee`・§2.5.3）:
  CPU 評価の L1 単一系統化（手書き J値評価 `_side_score` ／相手モデル profile 補正の撤去）に伴い、`_own_life_knee`・
  `W_LIFE_AGGRO_K`・ライフの concave/薄域膝（`W_LIFE`/`W_LIFE_LOW`/`W_LIFE_HIGH`）ごと削除。L1（`cpu_eval_v2`）の
  ライフ評価がこれを置き換える。以下は廃止前の設計記録（履歴）。
  ライフ薄域上乗せ（`W_LIFE_LOW`）を立ち上げる膝位置を `life_knee` で可変化。**自ライフ（守備）は攻め対面
  （相手 `profile.aggro_lean >= 0.6`）で膝を 3 へ**上げてレース下の 3 枚目まで厚く守り、**クロック側＝相手
  ライフは既定 2 のまま**＝自他で別カーブ。`profile` 無し＝両側 2＝従来同値。
  > **【2026-06-27 撤去】** 当初 plan 全廃後も「plan ではなく相手モデル profile 依存ゆえ存続」と注記していたが、
  > 後続の **CPU 評価 L1 単一系統化で手書き J値評価・profile 補正ごと撤去**された。重大度=中。
  > 検証基盤の faithful 化（`tests/cpu_arena.py`）: 旧アリーナ/regret ランナーは normal/hard へ自デッキ構成プランを
  > 供給していたが、自デッキ勝ち筋プラン全廃（2026-06-27）に伴いプラン供給は廃止＝アリーナも plan 非供給のフラット
  > 評価でデプロイ方策を反映する（以下は廃止前の記録）。
- **C-4 手札プレイ価値の精緻化**（`_side_score`／既存「コスト低減の資源価値化」の上位概念）: 手札は
  `max(プレイ価値, カウンター価値)` のオプションだが現状 700 固定＋カウンターのみ。**頭出し＝「次ターン手出し可
  （コスト ≤ 次ターン見込みドン）」の小ボーナスは『コスト低減の資源価値化』として実装済み**（上記既知の限界の
  項を参照）。**残＝打ち切り葉の不確実性ディスカウント【撤去 2026-06-27】**（`_settle_discount`／`_settle_eval`・§2.5.3）:
  > **【2026-06-27 撤去】** 自デッキ勝ち筋プラン全廃に伴い `_settle_discount`/`_SETTLE_CONFIDENCE` を削除（plan 供給時のみ
  > 作動の plan-gated 機能だった）。打ち切り葉の境界評価 `_settle_eval` 自体は存続するが、不確実性ディスカウントは無効。以下は履歴。
  settle は予算切れの局面を**既定解決**で無理に静止点へ整流して採点する＝探索で確かめた値ではないため、勝敗
  未確定（非 lethal）の settle 値を信頼度 `_SETTLE_CONFIDENCE`(=0.9) で中立（盤面差 0＝互角）へ寄せ、既定解決の
  選択が評価を不当に振らせるのを抑える（＝既定解決の中立化）。探索は「予算内で読み切れる線」をやや優先し、
  既定解決頼みの甘い/辛い見積りに賭けすぎない。lethal（±(W_WIN-ply)）は確定事象なので非割引。プラン供給時のみ
  作動（plan=None 完全同値）。`tests/test_cpu_self_plan.py`（plan 限定で正負を中立へ係数倍／lethal 非割引／
  `_settle_eval` 配線）。重大度=中。→ **C-4 完了**。
- **C-5 settle 楽観是正（受け手の地平線外打点の減点）【撤去 2026-06-27】**（`_settle_eval`／`_incoming_reach`／
  `W_SETTLE_PRESSURE`・§2.5.3）: 自デッキ勝ち筋プラン全廃に伴い `_incoming_reach`/`W_SETTLE_PRESSURE` を削除（plan 供給時のみ
  作動の plan-gated 機能だった）。打ち切り葉の境界評価 `_settle_eval` 自体は存続。以下は履歴。
  予算切れの打ち切り葉（settle）は相手のターン開始で止めて**静的**に採点する＝
  相手の反撃を読まない＝**動いた側に楽観バイアス**（殴られる直前でスナップショット）。これが「**手番頭で
  ドン/盤面に過剰コミットした手の深掘り値が楽観的に高く出る → 手番が進み代償が予算内に入って初めて崩落**」
  という非定常（value-realization gap）を生む。実ケース: ナミ(OP11-041)対面で 2000 のバスコ・ショット
  (OP16-110)に**ドン3枚を付与（付与時 deep ≈ +4798）→ ナミの【相手のアタック時】+2000 を静的層
  （`_attach_don_meaningful`/`_prune_futile_attacks` は素パワーのみ）が見落とすため貫通すると誤認 → 攻撃の
  決定まで来て初めて貫通不可が露見（attack deep ≈ −91〜−2050）→ 付与3枚を空振りで全返却**。対策: settle 葉
  （相手ターン静止点・plan 供給時のみ）で、相手の次ターンの**受け切れない打点本数**（`_incoming_reach`＝
  リーダー＋場の素パワーが自リーダーに届く本数 − 自アクティブブロッカー）× `W_SETTLE_PRESSURE`(=2500) を
  減点して楽観を是正する。**致死（reach ≥ 自ライフ）は C-2 telegraph が `evaluate` 内で計上済みなので致死
  未満のみ**扱い二重計上を避ける。`evaluate`/C-2 は不変＝真の地平線葉（読み切れた線）は触らず、**読み切れ
  なかった葉だけ**をペッシミ寄せ（C-4 と同系統＝既定解決頼みを信用しすぎない）。`tests/test_cpu_self_plan.py`
  （`test_b_settle_pressure_isolated`＝致死未満で reach×W 減点／reach0・致死・plan=None・自手番で不作動）。
  重大度=中〜高。**A/B 検証（採用）**: B-on(`W_SETTLE_PRESSURE=2500`) vs B-off(0)・both hard・席交互・
  **n=60＝39/60・wr0.650・+108 Elo（95%CI [+20,+211]＝0 を上回る）**で純利得を確認して採用（horizon3→4 の
  +58 Elo と同等以上）。検証装置は `tests/elo_settle_ab.py`（plan 全廃に伴い削除済み）。なお実ケースの「ドン空振り」症状そのものは B の
  評価地点（相手ターン開始の被打点）では捉えられない（付与ドンは手番終了で返るため空振りと有効活用が同一
  盤面になる）＝B は集計で勝率を上げるが当該症状の体感修正は別項（行為帰属＝自手番クローズでの資源変換評価）
  が必要。→ **C-5 採用（2026-06-20）**。
  > **value-realization gap 計測【実装済み】**（`tests/cpu_arena.py realize`／`decide_with_regret(out=…)`）:
  > regret（deep vs 1-ply 貪欲）は deep を正解とみなすため、**deep 自身が地平線外を楽観視する誤り**を構造的に
  > 検知できない。そこで「1 ターン内で採用手の深掘り値が `max → 最終決定` でどれだけ崩落したか」を gap として
  > 集計する別指標を追加（大きい gap = 予算地平線の外を楽観視して資源を溶かす兆候＝C-5 が縮める対象）。
  > 実行例: `python tests/cpu_arena.py realize --difficulty hard --seed 0`。

#### 2026-06 外部レビュー収束（再優先順位・新規項）
外部AIレビューとの往復で確定した方針。WBS と同期。**進捗（2026-06）**: B-1(a)/(b)・公開情報ベリーフ更新・
バッチA-1/A-2/A-3・**検証基盤（パズル集＋凍結ベースライン Elo＋regret ログ）は実装済み**（上記バックログ参照）。
- **検証基盤を全変更のゲートに（最優先・実装済み）**: 評価関数を触る前に症状を決定論的に固定する。
  **パズル/シナリオ回帰集**（`tests/test_cpu_puzzles.py`・致死を取る／**ドン→クロック変換（decide レベル）**／
  フェア性／脅威評価／ドン特性化ピン）に加え、**絶対強度メトリクス**（`tests/cpu_arena.py`）として
  **凍結ベースライン Elo**（固定参照相手＝既定 easy に対する挑戦者勝率→Elo・席交互で先手有利を相殺）と
  **regret ログ**（`cpu_ai.decide_with_regret`＝deep_value(深掘り最善) − deep_value(1-ply 貪欲)＝崖エラーの
  安価な代理）を実装。実ゲームは低速なので Elo の本走はスクリプト手動/定期実行、pytest スイートには機械
  健全性のみを高速・有界に固定（`tests/test_cpu_arena.py`）。実行例:
  `python tests/cpu_arena.py arena --challenger normal --baseline easy --games 20` ／
  `python tests/cpu_arena.py regret --difficulty normal --seed 0`。
- **B-1 の的を再定義【実装済み】**: 主役を「命中閾値クレジット」から**アイドルドンの末端減価**へ寄せた
  （上記 B-1(a)）。加えてカウンター強要は推定カウンター応答モデルで実現（B-1(b)）。
- **クロックは実装済み最重量項＝独立クロック項は不採用**（裏取り③）。
- **公開情報ベリーフ更新【実装済み】**: §2.5.4 の静的テンプレに、対局中に**見えた公開情報（相手の生の手札
  枚数・トラッシュの消費カウンター）分だけ想定カウンター緩衝を更新**（`_estimate_counter_buffer`）。追加
  クローン不要・相手手札の中身は読まない（フェア）。`normal` 保守モデルの過大な防御想定を是正。
- **時間割引は独立命題【実装済み】**: 「地平線を越える盤面/テンポ価値が残りターンで割り引かれない」（下記
  既知の限界）の是正は、ドン症状とは**機序が独立**（ドン症状は時間非依存＝床のタイブレーク）。**別検出器
  （レース/テンポ・パズル）**で独立に検証済み。是正は全項リスケールでなく**地平線外の盤面価値（場の存在価値
  `W_FIELD_COUNT`）の割引**にスコープを限定（`_board_tempo_factor`・残りゲーム長 `min(自,相手)ライフ` 依存・
  プラン供給時のみ）。詳細は下記「評価で未考慮の効果要素」の **時間割引【実装済み】** を参照。
- **探索強化（move ordering／置換表／ビーム拡幅）**: `HARD_ROOT_BEAM`
  の 4→6〜8 拡幅（B-3）は置換表によるレイテンシ削減が前提（4.4ms×clone 数の予算が逼迫するため）。

#### 評価で未考慮の効果要素（既知の限界・課題）
> **【2026-06-27 撤去】** 本小節は手書き J値評価 `_side_score` 時代の (B) 静的価格化を前提に書かれている。
> CPU 評価はその後 **L1 単一系統（`cpu_eval_v2.evaluate_v2`）へ集約**され `_side_score` ＋重み定数群は撤去された
> （§2.5.2 撤去バナー）。(A) 探索が結果盤面を採点する2層構造の枠組みは現行も同じで、(B) の静的価格化は今や
> L1 が担う。以下の `_side_score`／`W_*` 参照は廃止前の設計記録（履歴）。各項の存続/撤去は項末の個別マーカーに従う
> （例: `W_FIELD_POWER` による有効パワー加点・`_power_cap`/`_effective_power`/`DECK_DANGER` は L1 で存続）。

カード効果の影響は2層で扱われる: **(A) 探索が効果をクローン上で実適用した「結果の盤面」を
`evaluate` が採点する**ため、KO／バウンス／ドロー／ライフ追加／パワー上昇（`get_power` 反映）など
**盤面変化の結果は拾える**。一方 **(B) 静的評価 `_side_score` が明示的に価格化する特徴は限定的**
（ライフ・手札枚数＋カウンター値・場の枚数／有効パワー／ブロッカー／攻め圧・アクティブDON・
デッキ切れ境界）で、以下は**明示評価していない＝今後の課題**。いずれも探索の地平線
（`HARD_DEPTH`）内に結果が出る範囲でのみ間接的に反映される。

- **ステージの評価【撤去 2026-06-27】**（`_side_score`／`W_STAGE_COUNT`=800）: 自デッキ勝ち筋プラン全廃に伴い削除（`W_STAGE_COUNT`/`engine_aware` 加点は plan 供給時のみ作動の plan-gated 機能だった）。以下は履歴。これまで `_side_score` は
  場の評価を `p.field`（キャラ）のみで回し、**ステージ（`p.stage`）を一切採点していなかった**。ステージはキャラと
  違い攻撃/ブロックに出ない永続リソースのため、**存在価値（`W_STAGE_COUNT`・キャラの `W_FIELD_COUNT` より軽い）**を
  加点する（連続バフ＝聖地マリージョア等のコスト/パワー補正はキャラ側に既に反映済み＝二重計上を避ける）。存在は
  時間割引（`field_count_factor`）。ステージは可変・非対称（プレイ/除去/張替で増減）＝手選択に効く（自ステージのプレイ・
  相手ステージ除去の価値を拾う）。**プラン供給時のみ作動**（`engine_aware`＝plan is not None）＝plan=None では一切作動
  せず現行挙動と完全同値。ステージ不在時は `p.stage is None` で no-op＝キャラのみデッキの自己対戦・全ベースラインは不変。
  検証2本（`tests/test_cpu_self_plan.py`）。（注: ステージのエンジンプレミアムは A2 で撤去＝下記参照。）
- **リーダーの可変状態の評価【実装済み・Phase1.5・②】**（`_side_score`／`W_FIELD_POWER`）: 従来 `_side_score` は
  リーダーを一切読まなかった（パワーは `_power_cap` で相手の防御上限としてのみ使用）。リーダーの**能力/存在は
  ゲーム中不変＝探索全ノードで定数＝手選択(argmax)に無影響**のため静的定数項（能力プレミアム・存在 `W_FIELD_COUNT`・
  攻め圧 `W_ATTACKER`）は**不採用**。一方リーダーの**有効パワーはドン!!付与/バフで変動する＝可変**で、従来は
  リーダーへのドン付与（大型アタックを作る筋）が静的に「アクティブドンを失うだけの純損」に見えていた。場のキャラと
  同じく**有効パワー（対面最硬防御 `power_cap` までを線形・超過減衰）を `W_FIELD_POWER` で加点**し、ドン付与/バフの
  戦闘価値（相手の硬い体を KO/トレード）を拾う。
  > **【2026-06-27 注記】** 本項（`W_FIELD_POWER` による有効パワー加点）は存続するが、自デッキ勝ち筋プラン全廃に伴い旧来の
  > プラン依存ゲート（`engine_aware`＝plan is not None）が外れ、現在は **plan に関係なく常時作動**する。旧テキストにあった
  > 「クロックは `_plan_progress` の reach が別軸で評価」「`engine_aware`＝plan=None 完全同値」は `_plan_progress` 削除と
  > 併せて無効。検証テスト `tests/test_cpu_self_plan.py`（削除済み）への参照も無効。以下は履歴。クロック（ライフ削り）は
  `_plan_progress` の reach が別軸で評価済み＝二重計上を避ける（こちらは戦闘 KO/トレード軸）。プラン供給時のみ作動（`engine_aware`）＝plan=None 完全同値。検証2本。
- **【アタック時】を継続価値エンジン化【A2 で撤去・2026-06】**（旧 `_RECUR_TRIGGERS` の `ON_ATTACK`）: 【アタック時】
  持ちをエンジンプレミアム対象に含めていたが、エンジンプレミアム（`W_RECUR_ENGINE`）自体が A2 で正味マイナスと判明し
  撤去したため、本項も消滅した（`_prune_futile_attacks` の `_attacker_has_on_attack`＝無駄攻撃判定は独立で存続）。下記 A2 参照。
  **残**: リーダーのキーワード付与（可変・稀）の脅威評価／手札カードの効果ポテンシャル（除去/大型登場時を白紙と区別
  ＝効果分類が前提）／場のキャラの一度きり遅延誘発の個別期待値化（表現検出前提・出現頻度低）。
- **A2: 事前評価（静的ヒューリスティック項）のバイアス検証【実施済み・2026-06】**: `evaluate`/`_side_score` の
  先読み的・誘導的な静的項が、探索の見出す真の最善から評価を**歪めて（バイアスして）いないか**を検証した。手法＝各項を
  on/off した **ablation A/B 自己対戦**（challenger=フル評価 vs baseline=フル−当該項・per-decision で baseline 側だけ
  該当項を無効化・席交互・hard）。勝率→Elo で「正に寄与（ゲイン）／無寄与／負（バイアス）」を判定。結果（フル側勝率）:

  | 項 | 局数 | フル勝率 | 判定 |
  |---|---|---|---|
  | `_plan_progress`（逆算リーサル/マイルストーン） | 12 | 0.667 | ゲイン（明確）→残す |
  | `_telegraph_lethal`（テレグラフ致死） | 12 | 0.583 | 正→残す |
  | `W_STAGE_COUNT`（ステージ存在） | 12 | 0.583 | 正→残す |
  | `_board_tempo_factor`（時間割引） | 12 | 0.583 | 正→残す |
  | `_threat_value`（脅威キーワード） | 42 | 0.548 | やや正→残す |
  | `W_HAND_PLAYABLE`（手出し可ボーナス） | 42 | 0.500 | **無寄与→撤去** |
  | `W_RECUR_ENGINE`（エンジンプレミアム） | 42 | 0.452 | **正味マイナス→撤去** |

  > **【2026-06-27 追記】** 上表で「残す」と判定した plan-gated 項（`_plan_progress`／`_telegraph_lethal`／
  > `W_STAGE_COUNT`／`_board_tempo_factor`／`_threat_value`）は、その後の control-vs-midrange A/B（105局）で
  > アーキタイプ倍率が vs-midrange −5.7pp と判明したことを受け、自デッキ勝ち筋プラン／アーキタイプ系を全廃した際に
  > **いずれも削除**された（フラット評価ベースライン化）。本表は撤去前の局所判定の履歴として残す。

  **所見＝eval は概ね健全**（致命的バイアスなし・`_plan_progress` は明確なゲイン）。唯一の正味マイナスは `W_RECUR_ENGINE`
  （深い探索＝`HARD_HORIZON=4` が将来発動を結果盤面で既に拾う＝二重計上で評価を歪めていた）。`W_HAND_PLAYABLE` は完全中立
  （探索＋手札枚数項で代替済み）。両者を撤去（`_recurring_engine`/`_RECUR_TRIGGERS`/`_next_turn_don` も併せて削除）＝
  **強さ中立〜微増・eval 簡素化**。統計的には 42 局でも各項は有意ではない（95%CI は 0.5 を含む）が、方向は両独立パスで一貫。
- **耐性・特殊キーワードの評価【脅威項は撤去 2026-06-27・ブロッカーのみ存続】**: §2.5.6 の脅威項
  （`_threat_value`・ダブルアタック/効果耐性/速攻/バニッシュ/アンブロッカブル）は自デッキ勝ち筋プラン全廃に伴い削除
  （plan 供給時のみ作動だった）。**ブロッカー（`W_BLOCKER`）は plan 非依存で `_side_score` に常時加点され存続**する。以下は履歴。§2.5.6 の脅威項で
  **ブロッカー（W_BLOCKER）／ダブルアタック／効果耐性「KOされない」／速攻／バニッシュ／アンブロッカブル
  【ブロック不可】（`W_KW_UNBLOCK`＝900・`_is_unblockable`）**を資産として明示加点済み（プラン供給時・両側対称）。
  アンブロッカブルは `keywords` に "ブロック不可" が載らない（マスタ未格納）ため、**自前キーワードのテキスト**
  `【ブロック不可】(…)` で検出し、他者付与句 `…【ブロック不可】を得る` と区別（付与カードを誤検出しない）。
  付与で timed_keywords に載った場合は `has_keyword` で追従。`tests/test_cpu_puzzles.py`（A-1・スタブ＋実カード
  OP16-032/033/096 検出・付与 OP16-095/ST29-016/OP15-047 非検出）。**「効果では選ばれない」（対象保護）**は
  現カードプールに該当テキストが 0 枚＝表現未確定のため保留（出現時にテキスト/フラグ検出で追加）。
- **コスト低減の資源価値化【A2 で撤去・2026-06】**（旧 `W_HAND_PLAYABLE`=150／`_next_turn_don`）: 「次ターン手出し
  できる（`current_cost` ≤ 次ターン見込みドン）」手札への小ボーナスで、コスト低減を潜在資源として軽く価値化していた。
  **A2 評価バイアス検証（ablation A/B・フル vs 当該項なし・hard 42局）で勝率 0.500＝無寄与**と判明したため撤去。理由＝
  探索が「実際に出した結果の盤面」を直接採点し、基本の手札枚数項（`W_HAND`=700）も併せて持つため、「すぐ出せるか」の
  差は既に十分拾えており +150 の上乗せは何も足していなかった。撤去で eval 簡素化（強さ不変・関連テストも削除）。下記 A2 参照。
- **効果ターゲット選択（単一・多対象とも実装済み）**: **単一対象選択**（KO/除去/バウンス/手札破壊等・
  最大1体）に加え、**多対象「N枚まで」**（is_up_to・max≥2）も探索分岐へ昇格済み（§2.5.2 `_selection_moves`）。
  多対象は**影響度順**（`_rank_select_candidates`＝相手のカードはパワー大きい順に除去／自分のカードは小さい順に
  差し出す）に **min..max 枚の累積**選択を候補化し（候補は max-min+1 手＝有界）、「何枚・どれを選ぶか」を読む。
  **採点は 1-ply（即時盤面）**で行う（`decide` の `is_selection` 分岐）＝対象選択は確定効果の対象/枚数決定で
  即時盤面が信頼信号。多 ply 先読みは『相手のターン中に発火した自分の誘発除去』等で価値が washout/逆転し
  （例『相手のコスト1以下を2枚までKO』が深掘りで 0〜1 枚へ取りこぼす・2026-06-19 報告）採点を歪めるため使わない。
  併せて `_scored_search` は**深掘り同点手を 1-ply で割る**微小タイブレーク（`_TIEBREAK_W`・最大 ~0.005・
  実差>0.005 には不影響）を持つ。回帰=`tests/test_cpu_ai.py`（多対象累積列挙／相手ターン中の全除去・難易度×枚数）。
  **残**: min>1 の**強制**多対象で「どの組合せが最善か」（累積でなく任意部分集合）の網羅は組合せ抑制のため未実施。
- **任意コスト/任意効果の発動可否（accept/decline を採点）【実装済み】**（`_selection_moves`・`decide` の
  `is_selection` 分岐・§2.5.2）: 任意確認（`CONFIRM_OPTIONAL`＝「〜できる：効果」のコスト払いや「〜してもよい」）を
  CPU が **発動する/見送る の2手に分岐して 1-ply で採点**する。従来は `get_legal_actions` が任意確認を**既定
  (accept) の1手しか出さず**、CPU は任意コストを**必ず払って**いた（2026-06-19 報告: ティーチ OP16-080 の
  【相手のアタック時】『トリガー1枚を捨ててアタック対象をリーダー/黒ひげキャラに変更』を、リーダーが既に対象＝
  no-op でも毎回カードを浪費）。CPU 層のみの変更で `get_legal_actions`/`default_interaction_payload`（人間・自己対戦・
  監査の既定解決）は不変＝カード挙動ベースライン不変。検証＝得な任意コスト（ニコ・ロビン EB03-055＝ライフ1枚捨て→
  2枚追加＝純増）は accept・無意味リダイレクトは decline（`tests/test_cpu_ai.py`）。なお `ACTIVATE_MAIN` の任意コストは
  起動自体が意思表示のため確認しない（resolver 既定）＝本分岐の対象外。
- **探索地平線を越える効果価値（エンジンプレミアム）【A2 で撤去・2026-06】**（旧 `W_RECUR_ENGINE`=600／
  `_recurring_engine`／`_RECUR_TRIGGERS`）: 場のキャラ/ステージの「毎ターン価値を生む」能力（起動/常時/毎ターン誘発/
  ON_ATTACK）が探索ホライズンより先でも価値を生む分を、残ゲーム長で割引した小プレミアムで補っていた。**A2 評価バイアス
  検証（ablation A/B・hard 42局）で勝率 0.452＝正味マイナス（バイアス）**と判明したため撤去。理由＝`HARD_HORIZON=4` の
  深い探索が将来発動を結果盤面で既に拾っており、静的プレミアムは**二重計上で評価を歪めていた**（③手札ポテンシャルを
  「探索でカバー済み」として不採用にしたのと同じ機序）。撤去で強さ微増＋eval 簡素化（関連テストも削除）。下記 A2 参照。
- **時間割引【撤去 2026-06-27】**（`_board_tempo_factor`／`_side_score(field_count_factor=…)`・§2.5.3）: 自デッキ勝ち筋プラン全廃に伴い `_board_tempo_factor`/`field_count_factor` を削除（plan 供給時のみ作動の plan-gated 機能だった）＝`W_FIELD_COUNT` は現在 time-discount なしの満額。以下は履歴。静的重み
  `W_FIELD_COUNT`（場の存在価値＝地平線外の盤面ポテンシャル）が時間非依存で、残りターンで使い切れない盤面まで
  満額評価していた点を是正。**残りゲーム長の代理＝先に死ぬ側のライフ `min(自,相手)`** が短い（レース終盤）ほど
  場の存在価値を割り引く（`_TEMPO_FULL_TURNS=4` 以上で満額・未満で線形・`_TEMPO_FLOOR=0.3` で下限・両側対称）。
  スコープは**地平線外の盤面価値の割引に限定**＝ライフ（即時価値）・逆算リーサル/クロック（実レース進捗）・
  ブロッカー（即時防御）・カウンターは割り引かない。プラン供給時のみ作動（plan=None＝1.0＝完全同値・ライフ厚の
  早期も 1.0＝割引なし）。**別検出器＝レース/テンポ・パズル**で独立検証（`tests/test_cpu_self_plan.py`
  `test_race_tempo_puzzle_discounts_board_in_race`＝同じ置物の評価上昇がレース終盤<早期／plan=None は対照で不変）。
  ドン症状とは機序が独立。重大度=中。
- **自ライフの高ライフ逓減（concave）＝序盤の過剰カウンター是正【撤去 2026-06-27】**（`_side_score` ライフ項・
  実プレイ報告 2026-06-19 ナミミラー）:
  > **【2026-06-27 撤去】** 手書き J値評価 `_side_score`（と `W_LIFE`/`W_LIFE_LOW`/`W_LIFE_HIGH`/`_LIFE_KNEE_*`）の
  > 撤去＝CPU 評価の L1 単一系統化に伴い削除。ライフの非線形評価は L1（`cpu_eval_v2`）が担う。以下は履歴。
  旧実装はライフ価値が **線形**（`W_LIFE`×枚数）＋低ライフ膝（`W_LIFE_LOW`・
  `_LIFE_KNEE_DEFAULT`）で、**高ライフでも 1 枚＝6000 のまま**＝膝超のライフを割り引かなかった。このため
  `SELECT_COUNTER` の収支が「自ライフ1点（≈6000・control の `life_mult`=1.15 で ≈6900）＞ カウンター1枚
  （`W_HAND`700＋counter値×`W_COUNTER`0.6×`counter_mult`）」となり、**自ライフ5枚の序盤でも1点を守るために
  カウンターを3枚浪費**する過剰防御を招いた（ターン3で観測）。設計意図（「ライフは非線形＝薄いほど限界価値が
  高い」）に反する不整合。**是正**: 膝（`life_knee`）までは `W_LIFE`(6000) 満額・**膝超は `W_LIFE_HIGH`(2500) に
  減額**する concave カーブにし（near/far 分割）、高ライフ 1 点の限界価値を ~2875（control・膝3）＝カウンター
  1 枚相当まで下げて序盤のカード浪費を抑止。低ライフ域（膝以下）は `W_LIFE`＋`W_LIFE_LOW` 満額で厚く守る（不変）。
  プラン非依存（全評価に作用）＝本項は plan 全廃（2026-06-27）後も存続。検証テスト `tests/test_cpu_self_plan.py`
  （`test_life_value_is_concave_high_life_is_cheap`＝高ライフ限界=`W_LIFE_HIGH`／薄域限界=`W_LIFE`＋`W_LIFE_LOW`）は
  ファイルごと削除済み（参照のみ無効・機能は不変）。重大度=高。
- **control プリセットの受け身緩和（マナ余らせパス抑止）＋ removal 誤検出是正【プリセット側は撤去 2026-06-27・removal 是正は存続】**
  （`cpu_self_plan._PRESETS["control"]`／`cpu_opponent_model._REMOVAL_CUES`・同報告）:
  > **【2026-06-27 注記】** 前半の control プリセット緩和（`_PRESETS["control"]` の `vanilla_body_mult`/`act_margin_mult`）は
  > 自デッキ勝ち筋プラン全廃に伴い消滅。後半の分類器 removal_ratio 誤検出是正（`cpu_opponent_model._REMOVAL_CUES`）は
  > 相手モデル側で**存続**する。以下は履歴。グラインド寄りデッキが極端
  control 分類（`aggro_lean`≈0）になると、旧 control プリセットの `vanilla_body_mult`=0.45（場の小型を45%に割引）
  ＋`counter_mult`=1.4（手札温存）で「1コスト体を場に出す価値（≈700）＜ 手札カウンター価値（≈2380）」となり、
  **ターン2にマナ（2ドン）を余らせて何もせずパス**した（trace `folded:false`＝畳みでなく deep 探索が真に
  「TURN_END＞1ドロー展開 PLAY」と評価）。**是正**: `vanilla_body_mult` 0.45→**0.6**・`act_margin_mult` 1.5→**1.2**
  に緩和（守りの厚さ＝`counter_mult`/`life_mult`/`threat_def_mult` は維持）。併せて分類器の `removal_ratio` 誤検出
  （素の `KO` キューが「【KO時】＝自分が KO された時」の防御/リソース札＝ナミ/ホグバック/ペローナ/マルコ等を除去と
  カウント・素の「デッキの下」が自己ディグも拾う）を、除去動詞 `KOする`/`KOできる` とバウンス `持ち主のデッキ`/
  `手札に戻` に限定して是正（当該デッキは avg_cost 4.32／counter 比 0.61 で分類自体は control のままだが、
  removal_ratio 0.63→0.20 でスケジュール傾き等が正確化）。検証＝`tests/test_cpu_opponent_model.py`
  （`test_removal_cue_excludes_self_ko_triggers`）。全テスト pass・構造監査0。重大度=高。

> 直近の改修（実装済み）: 戦闘の閾値性（有効パワー）・J=デッキ切れ境界・召喚酔いの攻め圧除外・
> 「何もしない」を一級化・**効果の単一対象選択を探索分岐化**（§2.5.2）。
> **【2026-06-27 撤去】** キーワード資産（A-1/A-2）・C-4 settle 不確実性ディスカウント・時間割引（地平線外の盤面価値の割引）・
> 探索地平線を越える効果価値（毎ターン価値エンジンの将来価値プレミアム）等の **plan-gated 項は自デッキ勝ち筋プラン／
> アーキタイプ系の全廃に伴い削除**（control 倍率が vs-midrange −5.7pp の A/B を受けたフラット評価ベースライン化）。
> なお `W_HAND_PLAYABLE`（コスト低減の資源価値化）・`W_RECUR_ENGINE`（毎ターン価値エンジン）は別途 A2 検証で先行撤去済み。
> 残課題（多対象選択・反復深化＝置換表前提・一度きりの遅延誘発）は WBS に課題登録済み。

効果検証ハーネス（CPU 対 CPU 自己対戦・決定論・インバリアント検出）は
[`docs/TEST_SPEC.md`](TEST_SPEC.md) §3.1 を参照。

### 2.5.4 リーダー推測の相手モデル（`normal`・テンプレートデッキ）【廃止 2026-06／eval 補正は 2026-06-27 撤去】
> **廃止**: `normal` 難易度の削除に伴い、本節の相手モデル（リーダー推測）と `/api/cpu_template`
> 登録画面（フロント `cpuTemplate` モード）・`build_opp_profile_for_leader` を撤去した（唯一の消費者が
> `normal` だったため孤児化）。
>
> **【2026-06-27 撤去】** 当初は自デッキ勝ち筋プラン（旧§2.5.5）が自分のデッキ構成集計に `build_profile` を
> 使い、相手モデル `profile` は手書き J値評価の補正（`_own_life_knee` の対面別ライフ膝・`_estimate_counter_buffer`
> の推定カウンター緩衝＝B-1(b)・`defense_factor`/`aggro_lean` 補正）に使っていたが、CPU 評価を L1 単一系統
> （`cpu_eval_v2.evaluate_v2`）へ集約した際に **これら profile ベースの eval 補正をすべて削除**し、plan 系も
> 同日全廃した。その結果 **`cpu_opponent_model.py` は孤児化**（本番コードからの import はゼロ）したため、
> **モジュールごと削除した（`tests/test_cpu_opponent_model.py` も削除）**。以下は廃止前の設計記録（履歴）。

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

### 2.5.5 自デッキ勝ち筋プラン（`normal`/`hard`・`cpu_self_plan.py`）【撤去 2026-06-27】
> **【2026-06-27 撤去】** 自デッキ勝ち筋プラン／アーキタイプ・プリセット系は A/B 実験（control 倍率が
> vs-midrange −5.7pp）を受けて全廃。`plan` パラメータ・`cpu_self_plan.py`（`PlanProfile`/`build_plan`/
> `_PRESETS`/`_classify`/`_plan_progress`/`_derive_delta_schedule`/`_matchup_slope_mult`）・本節が供給して
> いた plan 限定の評価項（`vanilla_body_mult`/`counter_mult`/`life_mult`/`attacker_mult`/`idle_don_mult`・
> 逆算リーサル/マイルストーン/J値スケジュール）を全て削除した。現在の CPU は plan 非供給（旧 plan=None
> 相当）で動作し、相手モデル（`profile`/`cpu_opponent_model`）と手番計画キャッシュ（`plan_turn`/`plan_cache`）
> のみ存続する。以下は廃止前の設計記録（履歴）。

CPU は**自分のデッキ構成は完全情報**で知っているので、構成から「このデッキはどう勝つか」を逆算的に
分類し、勝ち筋に沿うよう自分側の評価重みをデッキ依存で切り替える。これにより「効果なし・低パワーの
置物を出すべきか」のような**同一の手がデッキによって最善／悪手に変わる**判断を表現する。

- **自動分類**（`build_plan`）: 自デッキのカード集計（平均コスト／カウンター密度／除去率＝相手モデルの
  `build_profile` を流用）から攻め寄り度 `aggro_lean` を出し、**aggro／midrange／control** に分類する
  （閾値 `aggro_lean ≥ 0.6 / ≤ 0.4`）。構成のみを使い隠れ情報は読まない＝フェア。空構成は中立
  （`NEUTRAL`・全乗数 1.0＝現行挙動）へフォールバック。
- **動的重み**（`evaluate(plan=...)`・自分側のみ補正）:
  - `vanilla_body_mult`: **効果なし・素パワー<5000・関連キーワード（ブロッカー/速攻/ダブルアタック）
    無し**の「置物」キャラの“場にいるだけ”価値の倍率。**control は強く割り引き**（置物に変えるより
    カウンターを温存）／aggro はやや増し。`_is_low_impact` が対象判定（効果・キーワード持ちは非対象）。
  - `counter_mult`: 自分の手札カウンター価値の倍率。**control は温存重視（>1＝出し渋り）**。
  - `life_mult`: 自分ライフ価値の倍率（control はライフ温存重視）。`attacker_mult`: 攻め圧の倍率
    （aggro 増・control 減）。
  - `idle_don_mult`（B-1(a)・§2.5.3）: **葉（自分の手番でない静止点 `is_turn=False`）での余剰アクティブ
    ドン価値**の倍率（<1.0）。OPCG は防御にドンを付与できないので、ターン終了後に浮いたアクティブドンの
    保持価値は本来低い。1.0 のままだと「両枝でクロック同値→ドンの床(`W_DON_ACTIVE`)でタイブレーク→握る」
    という**余剰ドン温存**を招くため、カウンターの薄い攻め寄りデッキほど強く減価する（プリセット
    aggro 0.4＜midrange 0.7＜control 0.85、NEUTRAL/plan=None=1.0）。自分の手番中は付与でパワーに変換できる
    生きた資源なので減価しない（`is_turn=True` は素通し）。自分側のみ・`plan=None` は完全同値。
- **逆算項**（`_plan_progress`・勝利状態からのサブゴール）:
  - **逆算リーサル**: 「相手リーダーに打点が通る、今/将来攻撃できるアクティブ体」を数え、相手ライフを
    **削り切れる本数**を持つ盤面を加点（`lethal_mult`）。探索の最短リーサル認識を非終端ノードでも
    “止めの形”へ誘導する。
  - **マイルストーン**: アグロ＝想定ダメージクロック（`clock_rate`）より相手ライフが先行して減っている
    分を加点／コントロール＝**J値スケジュール遵守度**（`_J_SCHED_W`）を加点。`aggro_lean` で両者を
    ブレンド（`milestone_mult`）。
  - **J値スケジュール遵守度**（理想ライン・`build_plan._derive_delta_schedule`／`_plan_progress`）:
    構成（攻め寄り度・除去密度）から「ターン t までに開くべき理想の **(相手J値 − 自分J値)**」を線形近似で
    導出（`delta_schedule[t]`・攻め寄り/除去多いほど傾きが急）。`_plan_progress` は**実測 J値差**
    （白＝デッキ残＋トラッシュの**枚数のみ**参照＝公開情報・中身は読まない）が当ターンの理想差を上回る分を
    加点・下回る分を減点する＝「理想の勝ちペースに乗れているか」を中長期視点で評価。プランは手を強制せず
    評価バイアスのみ＝手札的に理想手が打てない局面は**探索が理想差スコアの最も高く残る次善手を自然に選ぶ**。
    `delta_schedule` 空（`NEUTRAL`・`_PRESETS` 直接構築の単体テスト）は**従来の手札＋場リソース差採点へ
    フォールバック**＝回帰不変。詳細設計は `reports/cpu_plan_ideal_line_design_20260616.md`。
  - **マッチアップ補正**（Phase 2・`_matchup_slope_mult`）: 相手リーダー推測 `OpponentProfile`（§2.5.4・
    `normal` で供給・`POST /api/game/create` 配線）から理想ラインの傾きを補正する＝**速い相手（`aggro_lean`
    高）は前倒し（傾き急＝レース前に差を作る）／受け・除去の厚い相手（`blocker_ratio`＋`removal_ratio`）は
    後ろ倒し（傾き緩＝トレードで遅れる前提）**。参照は相手リーダー紐付けテンプレの集計のみ（実手札・実デッキ
    は読まない＝フェア）。`opp_profile=None`（`hard`・テンプレ未登録・自己対戦）は補正なし＝Phase 1 同値。
- **配線**: `POST /api/game/create` で CPU(p2) の自デッキ構成から `build_plan` し `CPU_GAMES[*].self_plan`
  に保持、`/api/game/cpu/step` が `decide_guarded(plan=...)` へ供給（`normal`/`hard` のみ。`easy` 非適用）。
- **フェア性／回帰**: 参照は自分のデッキ構成のみ（相手の実手札・実デッキは読まない）。`plan=None` では
  一切作動せず**現行挙動と完全同値**（既存テスト・挙動ベースライン不変）。

> 自動分類は粗い静的推定。より正確な勝ち筋指定（リーダー/デッキ別のフィニッシャー・温存方針）は
> `cpu_templates`（§2.5.4）への明示メタデータ拡張で上書きする余地があり、継続テーマ（§2.5.3）。

### 2.5.6 脅威評価（対面プランのルールベース実現・`normal`/`hard`）【撤去 2026-06-27】
> **【2026-06-27 撤去】** 自デッキ勝ち筋プラン／アーキタイプ・プリセット系は A/B 実験（control 倍率が
> vs-midrange −5.7pp）を受けて全廃。本節の脅威/キーワード資産項（`_threat_value`/`_is_unblockable`/
> `W_KW_DOUBLE`/`W_KW_RESIST`/`W_KW_RUSH`/`W_KW_BANISH`/`W_KW_UNBLOCK`・`threat_aware`）は plan 供給時のみ
> 作動する plan-gated 機能だったため、plan 全廃に伴い削除した。現在の CPU は plan 非供給（旧 plan=None
> 相当）で動作する。以下は廃止前の設計記録（履歴）。

LLM を使わず、**カードデータ（キーワード・効果耐性）から「除去すべき脅威／温存すべき資産」を動的に評価**し、
①の単一対象選択探索（§2.5.2）が**相手の本当の脅威に除去を向ける**ようにする。盤面を直接読むため、静的な
対面プランの「脅威リスト」より精度が高い（その局面の実際の脅威を毎回評価する）。

- **脅威/キーワード資産項**（`_threat_value`・`_side_score(threat_aware=True)`）: 場のキャラに対し、
  **ダブルアタック**（`W_KW_DOUBLE`＝リーダー打点2倍）・**効果耐性「KOされない」**（`W_KW_RESIST`＝
  除去されにくい永続体）・**速攻**（`W_KW_RUSH`）・**バニッシュ**（`W_KW_BANISH`）・**アンブロッカブル
  【ブロック不可】**（`W_KW_UNBLOCK`＝ブロッカーで止められず確実にリーダーへ通る／自前キーワードのテキスト
  `【ブロック不可】(…)` で検出し付与句 `…を得る` と区別＝`_is_unblockable`）を加点する。ブロッカーは
  既に `W_BLOCKER` で計上済みのため除外。
- **両側対称適用**: `evaluate` は自分側・相手側の双方に脅威項を適用する。相手の脅威キャラは相手側スコアを
  押し上げ→**それを除去すると自分の評価が大きく上がる**→ ① の探索が最善の除去対象（＝最大の脅威）を選ぶ。
  自分側では「キーワード資産」として温存・活用の動機になる。
- **フェア性／回帰**: `plan` 供給時のみ作動（`threat_aware = plan is not None`）。`plan=None` では一切
  作動せず**現行挙動と完全同値**（既存テスト・挙動ベースライン不変）。`easy` は非適用。

> 検証例: P9000ダブルアタックとP8000効果なしが並ぶ相手盤面で、プラン有効時は前者の除去が後者より評価改善が
> 大きく（脅威項分）、① の探索が**ダブルアタックを優先除去**する。plan 無しでは差はパワー差のみ（ほぼ同等）。
>
> 継続テーマ: 多対象選択の探索（§2.5.3）、相手モデル（§2.5.4）との連動（除去多デッキには耐性を厚く見る等）、
> 対面別マリガン方針。脅威シグナルの拡充（アンブロッカブル等）。

### 2.5.7 （削除）MCTS エキスパートエンジン

> **撤去（2026-06）**: ターン粒度マクロ MCTS（`cpu_mcts.py`）を新難易度 `expert` として実験導入していたが、
> **強さ・速度で α-β=`hard` に劣ると判明し撤去**した。出荷 CPU は `hard`（α-β）の単一系統（§2.5.8 が出荷設計の正本）。
> MCTS で得た知見のうち**残す価値があったもの**＝(a) 相手伏せ手札の決定化（公平な隠れ情報処理）と
> (b) 自己対戦で学習した価値関数の葉ブレンドは、いずれも `hard`（α-β）へ移植済み＝**PIMC**＋**学習価値葉ブレンド**として
> §2.5.8 に集約した。学習価値の特徴抽出・推論・収集基盤（`cpu_features`/`cpu_value_model`/`cpu_value_data`・人間ログ
> 収集 [`human_log_collection.md`](human_log_collection.md)）も `hard` の評価関数強化として継続する。

### 2.5.8 強さ=Elo 優先・フェア制約の出荷設計（2026-06・α-β＋PIMC）

> **本節が出荷 `hard`（α-β）CPU の現行設計の正本**（§2.5.2 等の旧「チート既定」記述は本節へ更新済み）。
> 測定駆動で確定。各 Phase の計測は `docs/reports/` 参照
> （`cpu_phase1_horizon_sweep_20260623.md`／`cpu_phase4_pimc_budget_split_20260623.md`）。
>
> **【2026-06-28 撤去】 学習価値（winprob 葉ブレンド）は全廃**＝A/B で Elo 中立と実証され、特徴抽出/推論/収集
> 基盤（`cpu_features`/`cpu_value_model`/`cpu_value_data`/`value_model.json`）・葉ブレンド `_value_blend`・
> 人間ログ/再学習パイプライン・L1 SPSA チューナごと削除。**現行の出荷設計は「α-β＋ビーム＋PIMC（決定化）＋
> L1 葉評価」**。以下の「学習価値（Phase 3a/3b）」「人間ログ活用」を述べた箇所は撤去前の設計記録（履歴）。
> 強化レバーの総括は `cpu_strength_plan_20260628.md` §K（**現アーキは達成可能上限に近い**）。

- **情報方針の引数化＋フェア即切替（Phase -1）**: `decide(..., info_policy="fair")` を既定化。`fair`＝相手手札の
  中身・隠れ手札依存手を読まない（`see_opp_hand=False, opp_public_only=True`）。`cheat`（旧 hard＝
  `see_opp_hand=True`）は凍結ベースライン/診断用に明示指定でのみ到達。本番経路（app.py/decide_client/
  decide_worker）は既定参照のみで切替（無改変）。
- **限界要因の切り分け（Phase 1）**: horizon 掃引（fair vs cheat）で **深さは horizon=4 で頭打ち**
  （−98→−53→−53・ペア差 符号検定 p=0.52 非有意）＝フェア化の損失（~−53 Elo）は**探索深さでなく
  情報の限界**。harness=`tests/phase1_sweep.py`。
- **PIMC＝決定化（Phase 2・本命）**: `_pimc_scored`＝相手の伏せ手札を `_determinize_opponent` で K 通り
  再サンプル→各々を完全情報 α-β（`_scored_search`）で採点→**世界平均**で手を選ぶ。実手札は読まない＝
  フェアのまま隠れ情報を確率的に補い、フェア化の楽観バイアスを埋める。決定論は親 rng から世界 rng を派生。
  単体で **+53〜+61 Elo（vs 素 fair・directional）**。`pimc_worlds=1` で完全休眠＝従来同値。
- **学習価値の葉ブレンド（Phase 3）**: `evaluate` 末尾に `_value_blend = base + α·SCALE·(winprob−0.5)`
  （`OPCG_VALUE_BLEND`/override・**α=0 既定で完全不変**・葉で predict も呼ばない）。`value_model.json` を
  **fair・多シード再学習**（val_acc 0.725>0.670・1/4 データで上回り＝旧不発の主因はデータ/ラベルの汚れ）。
  ブレンドは旧モデルで −70 Elo→新モデルで **+35 Elo（失敗を反転・directional）**。PIMC とは重複し**積み上がらない**
  （+35≦PIMC 単体）。＝blend は「PIMC の安価な代替」。
- **予算按分で1秒運用（Phase 4・出荷配線）**: `_effective_budget`/`set_budget_override` で per-decide 予算を
  按分。**PIMC K=4 を予算 75/世界（合計≈300＝素 fair と等倍計算量）にしても +53 Elo を維持**（利得は計算量
  でなく決定化由来＝Phase 1 と整合）。レイテンシ実測 max~536ms（PyPy 換算 ~250ms）＝**1秒余裕**。全予算版
  （K×フル予算）は max 2.2秒で不可。当初想定の TT/反復深化は**不要だった**。
- **出荷構成（env・Dockerfile）**: `OPCG_PIMC_WORLDS=4`＋`OPCG_HARD_PER_MOVE_BUDGET=75`＝**PIMC K=4・予算
  按分**。探索ノブは env 上書き可（`OPCG_HARD_HORIZON`/`_BEAM`/`_OPP_BEAM`/`_ROOT_BEAM`/`_PER_MOVE_BUDGET`・
  既定値＝従来同値）。**即ロールバック＝`OPCG_PIMC_WORLDS=1`**（PIMC 休眠）。env 未設定（テスト/CI）は全て従来同値。
- **測定基盤（Phase 0）**: `tests/cpu_arena.py` に antithetic 席ペアリング（`arena_paired`）＋Wilson 区間
  （`wilson_interval`/`elo_ci`）＋多ポリシー/PIMC/blend/予算の per-decider A/B。±35 Elo ノイズ帯を縮めるが
  半幅<15 の確定は大局数（CI/定期）が必要＝本検証は限界要因の**定性判定**が目的。
- **留保**: 強さは ±100 級 CI で directional（未有意・確定には桁違いの局数）。本節の**決定化（PIMC）と学習価値の
  葉ブレンド**は、撤去した実験エンジン（旧 §2.5.7 MCTS）が持っていた強み＝決定化・学習価値を α-β(`hard`) へ移植したもの
  （MCTS 本体は α-β に強さ・速度で劣り撤去）。人間ログ活用・非線形価値は将来項（WBS）。


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
- **PASSIVE 再計算中はイベントを発行しない**（`gamestate.resolve_ability` が `_in_passive_recalc` を見て抑制）。`_apply_passive_effects` は `PASSIVE`/`YOUR_TURN`/`OPPONENT_TURN` の継続効果を盤面操作の度に再適用する（cost_buff/passive_power は Step1 でリセット→再適用＝**スタックしない**）。再適用ごとに `action_events` へ `EFFECT`(BUFF) を積むと eventLog/リプレイが同一イベントで膨張する（例: ティーチ OP16-080 の【相手のターン中】コスト+1 が毎リフレッシュ重複）。結果はカードの cost/power に反映済みで表示に不要なため、再計算中は発行を抑制する（**本物の発動＝非再計算経路は従来どおり記録**・挙動不変）。

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
| `opcg_sim/src/core/cpu_ai.py` | CPU(AI) 意思決定（`evaluate`/`evaluate_base`/`decide`/`_search`/`decide_guarded`）。**葉評価は L1 単一系統＝`cpu_eval_v2.evaluate_v2` に集約**（`evaluate_base` が素返し・`evaluate` はその別名。手書き J値評価／評価v2フラグ／effect-value 実験／相手モデル profile 補正は **2026-06-27 撤去**／学習価値葉ブレンド `_value_blend` は **2026-06-28 撤去**）・**情報方針 fair 既定＋PIMC 決定化（§2.5.8）**（`cheat` は診断用に明示指定のみ）・**ターン境界評価探索**（`_settle_eval`＝horizon/手番パリティ是正）・α-β＋ビーム・最短リーサル認識・**効果の単一対象選択の探索分岐**（`_selection_moves`）・L1 で存続する評価補助（`_power_cap`/`_effective_power`/`W_POWER_OVERCAP`/`DECK_DANGER`＝`cpu_eval_v2` が import）・暴走防止（自デッキ勝ち筋プラン補正§2.5.5／脅威評価§2.5.6 は **2026-06-27 に全廃**＝plan 非供給のフラット評価） |
| `opcg_sim/src/core/cpu_eval_v2.py` | **CPU の現用葉評価（L1＝カード通貨）`evaluate_v2`**。トレース時は内訳を `out["v2"]` に出力。`cpu_ai` から `_power_cap`/`_effective_power`/`W_POWER_OVERCAP`/`DECK_DANGER` を import して使う（係数は固定＝SPSA で Elo 余地≈0 と実証され調整器は撤去・2026-06-28） |
| ~~`opcg_sim/src/core/cpu_self_plan.py`~~ | **【削除 2026-06-27】** 自デッキ勝ち筋プラン（`build_plan`/`PlanProfile`・旧§2.5.5）。A/B 実験（control 倍率が vs-midrange −5.7pp）を受けてアーキタイプ・プリセット系を全廃しファイルごと削除 |
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
| `OPCG_LOG_SILENT` | （未設定） | `1` で `resolver.py` のデバッグ print スナップショットを抑止（テスト/診断の必須フラグ） |

### 5.1 ログ／可観測性

汎用のアプリケーションログ（旧 `log_event` ＝ ゲーム内イベント／API／エラーログと、その GCS/Slack
転送・FE 取り込み `/api/log`・セッション ID 伝播ミドルウェア）は **すべて撤去した**。本番は Cloud Run の
素の stdout（Cloud Logging）以外に明示的なアプリログを出さない。例外は各エンドポイントが整形済み
エラー（`success:false`＋`error.code`）として返すのみ。

唯一のログは **CPU 思考トレース**（CPU 挙動改善用）。`log_event` を経由せず GCS にも行かない。

```
ローカル自己対戦: tests/cpu_replay.py → ローカル JSONL（1 行 = 1 意思決定）
実アプリ対局    : create に cpu_trace=true（opt-in）→ CPU_GAMES[gid] にメモリ蓄積
                  → GET /api/game/{game_id}/replay で {リプレイ種, decisions} を取得
```

各意思決定に「選んだ手・上位候補スコア（prelim/deep）・regret・J値成分内訳・読み筋」を記録する。
`decide`/`decide_guarded` の `trace` 引数（既定 None＝無オーバーヘッド・挙動不変）で採取し、RNG 中立
（トレース有無で進行が分岐しない）。詳細・検証観点は [`docs/TEST_SPEC.md`](TEST_SPEC.md) §3.2。

**ログの扱いの正本は [`docs/LOGGING.md`](LOGGING.md)。**

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
