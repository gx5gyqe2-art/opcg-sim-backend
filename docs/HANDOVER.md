# 引き継ぎ資料 — カード効果システム刷新

最終更新: 2026-06-04 / ブランチ: `claude/handoff-docs-review-PHoSv`（`main` から分岐・作業中）

**マージ履歴（新しい順）**
- PR #4 — 合成ルールレジストリ刷新の土台・EffectParserV2 本番化
- PR #6 — ルール拡充・正確性修正・継続効果統合・置換効果
- `claude/handoff-materials-review-u5cqy` — 診断上位 OTHER 表現5種ルール化・サーチ構造修正
- `claude/handoff-materials-review-BWP0A` — 「手札捨て」修正・パーサ Round 2-3・UI拡張フェーズ1-3（main にマージ済み）
- `claude/handoff-docs-review-bIYkG` — GENERIC条件94→1・保護者型置換効果対応（PR #7 で main にマージ済み）
- `claude/handoff-docs-review-PHoSv` — **デッキ公開→条件付き登場の構造修正（本ブランチ・作業中）**

このドキュメントは、本リポジトリ（opcg-sim-backend）の **カード効果処理の刷新作業** を
引き継ぐための資料です。詳細な設計は `docs/parser_v2.md` を、本書はその上位の
オリエンテーション（全体像・運用・残タスク）を担います。

---

## 1. 背景と目的

**課題**: ゲーム中にカード効果が想定通り実行されない場合が多数あった。

**原因（診断で判明）**:
1. 旧 `parser.py` は巨大な if 連鎖で原子句を解釈しており、順序依存・サイレント失敗・
   テスト困難という構造的課題があった。
2. パーサが `ActionType.OTHER`（=解析できても実行系が無く何もしない）に落ちる句が
   **約940件**存在した。
3. エンジン側（`gamestate.apply_action_to_engine` / `resolver`）に未実装のアクションが多く、
   正しい型を出しても盤面が変わらなかった。
4. 「このバトル中」「次の相手のターン終了時まで」等の**期間付き効果**を管理する機構が無かった。

**対応方針**: 中間表現(IR)とインターフェース(`parse_card_text`)を維持したまま、
日本語→IR 変換を**合成ルールレジストリ方式**へ刷新し、エンジン実行系を拡充。
段階的・非破壊で移行し、最終的に新パーサ(V2)を本番有効化した。

---

## 2. 現在の状態（main ブランチ時点）

| 指標 | 刷新開始時 | PR#4 時点 | main（BWP0Aマージ後） | **本ブランチ** |
|---|---|---|---|---|
| 原子句カバレッジ（ルール命中率） | 0% | 約57% | 約95.4% | **約95.4%（変化なし）** |
| `ActionType.OTHER`（実行時に何もしない句） | 942 | 421 | 108 | **108（変化なし）** |
| 未分類条件 `GENERIC`（誤発動の温床） | — | 251 | 94 | **1** |
| パーサルール数 | 0 | 15 | 49 | **49（変化なし）** |
| テスト総数 | 17 | 43 | 137（全緑） | **165（全緑）** |
| 本番パーサ | レガシー | EffectParserV2 | EffectParserV2（既定） | **EffectParserV2（既定）** |

**パーサ改善の軌跡（累積）**

| セッション/ブランチ | 主な変更 | カバレッジ | OTHER | ルール | テスト |
|---|---|---|---|---|---|
| PR#4 | V2 基盤・15ルール | 約57% | 421 | 15 | 43 |
| PR#6 | 継続効果・置換効果 | 約70% | 333 | 25 | 81 |
| u5cqy ラウンド1-4 | 診断上位5種・サーチ構造 | 92.5% | 234 | 42 | 115 |
| BWP0A ラウンド1 | 手札捨て断片修正 | 93.0% | 209 | 42 | 118 |
| BWP0A ラウンド2 | hand_to_deck 等5種 | 93.8% | 175 | 45 | 125 |
| BWP0A ラウンド3 | FREEZE/NEGATE_EFFECT 等 | 95.4% | 108 | 49 | 137 |
| bIYkG ラウンド1-4 | GENERIC条件94→1・保護者型置換効果 | 95.4% | 108 | 49 | 165 |
| PHoSv ラウンド1 | デッキ公開→条件付き登場の構造修正 | 95.9% | 104 | 50 | 170 |
| **PHoSv ラウンド2** | **公開→インライン条件→登場（play_revealed TEMP化）** | **95.9%** | **104** | **50** | **172** |

**UI 拡張フェーズ（opcg-sim-frontend 連携）**

| フェーズ | 主な変更 |
|---|---|
| Phase 1 | FREEZE/NEGATE Pixi オーバーレイ・trigger_text 表示・API シリアライズ拡張 |
| Phase 2 | 効果解決ログパネル（`ActionLog.tsx`・`action_events` API フィールド） |
| Phase 3 | PendingRequest overlay 改善（ソースカード名付きメッセージ・バトル表示カード名化） |

- 全2652カードの能力構築・実デッキ(imu/nami)ロード・ゲーム開始〜数ターン進行を確認済み。
- レガシー vs V2 の全カード比較で **退行(新規OTHER)=0** を一貫して維持。

### 本セッション（本ブランチ）で追加した内容

**パーサルール拡充（15→25種）**
- `grant_keyword`: 「【ブロッカー】等を得る」。構造分解でキーワードが脱落し誤 `BUFF` に
  落ちていたバグを修正（タグ一括除去をキーワード能力タグ保持に限定）。
- ライフ操作 `life_recover`/`life_to_hand`/`hand_to_life`/`life_to_trash`/`life_face`。
  `life_to_hand` は legacy が「ライフの上か下から…手札に加える」を `dest=LIFE` と誤判定して
  いた no-op バグを修正。`FACE_UP_LIFE` 実行系を追加。
- ドン操作 `don_attach`/`don_set_active`/`don_set_rest`/`don_return_deck`（枚数ベース）。

**正確性バグ修正**
- 【ターン1回】(`TURN_LIMIT`) を enforce（従来常に True で無制限発動できた）。
- エンジンに `REST_DON` 実行系が無く【ドン!!×N】コストが no-op だったのを修正。
- 条件 fail-safe: 解釈不能な `OTHER` 条件は False。`GENERIC` は許容＋ログに整理しつつ、
  評価可能なクラスタ（`LEADER_TRAIT 『X』`/`FIELD_COUNT`/`DECK_COUNT`/`LEADER_COLOR`）へ
  分類して誤発動源を削減（GENERIC 251→94）。

**継続効果マネージャの統合（§7-2 完了）**
- `GRANT_KEYWORD` を `timed_keywords`、期間付き `COST` を `timed_cost` に統合。従来は
  `_apply_passive_effects` のリセットで即消えていた（キーワード146句が実質不発・コスト減少も
  消滅）。`has_keyword()`/`current_cost` で参照を一本化し、duration で失効、場を離れる際は
  `drop_for` で破棄（未配線だった）。

**置換効果 MVP（§7-3）**
- 「このキャラが(バトルで)?KOされる/場を離れる場合、代わりに〜」を `REPLACE_EFFECT`
  （置換を `sub_effect` に保持）として実装。`_active_replacement` が除去の瞬間に PASSIVE を
  走査し、条件・実行可能性を満たせば置換を実行して本来の除去をスキップ。

### 本ブランチ（`claude/handoff-docs-review-PHoSv`）で追加した内容

#### ラウンド1: デッキ公開→条件付き登場の構造修正（ルール 49→50, テスト +5）

**問題（トレースで判明）**: 「自分のデッキの上から1枚を公開し、コスト2のキャラカード1枚までを、
登場させる。その後、残りを…」が **1原子句化**していた（`_parse_to_node` の分割パターンに
`公開し、` が無い）。その結果:
- レガシーフォールバックが `PLAY_CARD` の対象を **`zone=FIELD` または `zone=DECK`** と文言次第で
  誤推定（**OTHER指標に出ない隠れミスターゲット**。デッキ全体から自由に登場できてしまうルール違反）。
- 末尾の `残りを…デッキへ` が `DECK_BOTTOM(zone=TEMP)` だが LOOK が無く **TEMP が空＝no-op**。
- 連用形「登場させ、残りを…」は `させ、` split で `登場` が `させ` から切れ **OTHER** 化（OP06-057）。

**修正（サーチ構造修正 u5cqy R4 と同型）**
- `parser.py _parse_to_node`: `(デッキの上から\d+枚(?:まで)?を公開し)、` を `。` へ置換し、LOOK を
  独立クローズに分割（デッキ文脈限定。`公開する`（句点付き別構文）は対象外）。
- `rules/atoms.py look_deck`: 正規表現を `を(?:見て|公開し)` に拡張し、公開句も LOOK(→TEMP) 化。
- `rules/atoms.py play_from_temp`（新規, priority=39）: 分割後の「（フィルタ）1枚までを、
  （レストで）登場させ(る)」→ `PLAY_CARD(zone=TEMP, dest=FIELD)`。明示ゾーン（手札/トラッシュ/
  ライフ）・`このキャラを`（play_self）・条件/トリガー文（「登場させた場合/時」）は除外。
  連用形 `登場`（末尾）も `$` アンカーで拾う。残りは既存 `remaining_*` が TEMP から戻す。
- `matcher.py parse_target`: 特徴抽出の正規表現に `『』` を追加（`『白ひげ海賊団』を含む特徴を持つ`）。
  名前は `「」`、特徴は `《》`/`『』` で衝突しない（condition 側も `『X』` を特徴として扱う）。

**ツール修正**
- `tests/compare_parsers.py _has_other`: `json.dumps(..., default=str)` を追加。summary に
  enum（`CompareOperator` 等）が混じると **退行検知ツールがクラッシュしていた既存バグ**を修正。

**テスト +5**
- golden +3（`deck_reveal_play_cost` / `deck_reveal_play_trait`（『』特徴フィルタ） /
  `deck_reveal_play_rested`（連用形・RESTED））。
- engine +2（`test_deck_reveal_play_from_temp_flow`（LOOK→PLAY_CARD(TEMP)→TEMPリーク無し） /
  `test_deck_reveal_play_rested_status`（レスト登場で is_rest=True））。

**結果**: カバレッジ 95.4%→95.9%（+0.5pt）、OTHER 108→104（−4）、フォールバック 227→201（−26）、
`play_from_temp` 26件命中、退行(新規OTHER)=0。OTHER減に加え、上記**隠れミスターゲット約7枚
（OP06-057/OP06-119/OP08-052/OP08-054/ST12-010/ST12-013/ST12-017）を正しい TEMP 経由に是正**。

**残（本クラスタの未対応・フォロー候補）→ 一部はラウンド2で対応**
- scry-1「公開し、そのカードをデッキの上か下に置く」（OP08-049）＋条件付き速攻 — 登場を伴わない別型。**未対応**。
- ~~条件付き任意登場「そのカードが…の場合、登場させてもよい」~~ **ラウンド2で対応**（deck系）。
- ~~`play_revealed`（…単一句）の `zone=FIELD` ミスターゲット~~ **ラウンド2で TEMP 化**。

#### ラウンド2: 公開→インライン条件→登場（play_revealed の TEMP 化, テスト +2/golden 改修1）

**問題（トレースで判明）**: 「（デッキの一番上を）公開し、そのカードが…の場合、（レストで）登場
させてもよい」が、`_parse_logic_block` で `Branch(REVEALED_CARD_TRAIT, PLAY_CARD)` になった後、
**先頭ゲート条件の lift（§パーサ parse_ability 121行）でアビリティ条件へ引き上げられ**ていた。
結果、(1) 公開(LOOK)が実行されず条件の raw_text に埋もれて消失、(2) `PLAY_CARD` が `zone=FIELD`
へ誤ターゲット、(3) resolver はアビリティ条件を効果実行**前**に評価するため `REVEALED_CARD_TRAIT`
を公開前に判定するという**順序矛盾**を抱えていた（OTHER指標に出ない隠れ不具合）。

**修正（インライン条件を Sequence 内に保持して lift させない）**
- `parser.py _parse_to_node`: 分割対象に `デッキの一番上を公開し、` を追加（Round1 の `\d+枚` 版と統合）。
  分割で `Sequence[LOOK, Branch(...)]` になるため、top-level Branch ではなくなり **lift されない**
  → `REVEALED_CARD_TRAIT` がインライン Branch として残り、LOOK→条件評価の順序が正しくなる。
- `parser.py _parse_condition_obj`: `REVEALED_CARD_TRAIT` の判定を `公開し` 非依存に緩和
  （分割後の条件側に `公開し` が残らないため、`そのカード`＋filter を手掛かりにする）。
- `rules/atoms.py look_deck`: `デッキの一番上を(見て|公開し)` も LOOK(value=1) として拾う。
- `rules/atoms.py play_revealed`: 対象を `zone=FIELD` → **`zone=TEMP`** に修正（公開で TEMP に
  載った1枚＝公開カードを登場。filter は条件側 `REVEALED_CARD_TRAIT` が担う）。
- `resolver.py`: `LOOK` 実行後、`targets` が無くても `player.temp_zone[0]`（公開デッキトップ）を
  `context["last_revealed_card"]` に記録（従来は REVEAL/FACE_UP_LIFE のみ・targets 必須で、
  LOOK では未設定→`REVEALED_CARD_TRAIT` が permissive True に倒れて誤登場の恐れがあった）。

**結果**: deck系3枚（OP01-060/OP07-048/OP12-058）が
`Sequence[LOOK(1) → Branch(REVEALED_CARD_TRAIT → PLAY_CARD(TEMP[,RESTED])) → (残り→デッキ)]`
の正しいインライン構造に。退行(新規OTHER)=0、テスト 170→172。
- golden 改修1（`play_revealed_rested` を新構造 seq+branch に更新）。
- engine +2（`test_reveal_conditional_play_match`（条件一致→対象選択→登場・temp リーク無し） /
  `test_reveal_conditional_play_no_match`（条件不一致→登場せず temp に残る））。

**残（本ラウンドのスコープ外）**
- **ライフ公開系（OP10-022/ST13-007/010/014, 4枚）**: 「自分のライフの上から1枚を公開し、…の場合、
  登場させてもよい」。登場元が**ライフ**（デッキでない）ため、ライフ→TEMP 公開と未登場カードの
  ライフへの戻しが要る。保留中の難所「ライフ look-and-place」と同系統で、別途エンジン拡張が必要。
- OP12-058 の `登場させた場合…【速攻】` は GENERIC（permissive）止まり（PREV_ACTION 化は別途）。
- scry-1（OP08-049, 公開→デッキ上下＋速攻）は登場を伴わない別型で未対応。

### 本ブランチ（`claude/handoff-docs-review-bIYkG`）で追加した内容

#### GENERIC 条件の集中分類（94→1）と保護者型置換効果（Round 1-4）

BWP0A マージ後の状態では GENERIC 条件が **94** 件残存していた。本ブランチでは4ラウンドに分けて集中的に分類し、**94→1** まで削減した（-98.9%）。

##### 新設した ConditionType（6種、`enums.py`）

| ConditionType | 説明 | 代表テキスト |
|---|---|---|
| `PREV_ACTION` | 直前アクションの実行結果（そうした/そうしなかった/登場させた） | 「そうした場合」「そうしなかった場合」 |
| `DON_COUNT_COMPARE` | 自分と相手のドン!!枚数の相対比較 | 「相手よりドン!!が多い場合」 |
| `LEADER_STATE` | リーダーの状態（IS_ACTIVE/IS_RESTED/パワー閾値） | 「リーダーがアクティブの場合」「リーダーのパワーが6000以上」 |
| `FIELD_COUNT_COMPARE` | 自分と相手の場キャラ数の相対比較 | 「キャラが相手より少ない場合」 |
| `REVEALED_CARD_TRAIT` | 公開/LOOK したカードの特徴/コスト/タイプ条件 | 「そのカードが〈海賊〉の特徴を含む場合」 |
| `OPPONENT_REMOVAL` | 相手の効果/バトルで場を離れる/KOされる置換条件（パワー/コスト/特徴フィルタ付き） | 「相手の効果でコスト4以下のキャラがKOされる場合、代わりに」 |

##### `_parse_condition_obj()` の拡張（`parser.py`）

- **HAS_CHARACTER**: 状態付き（`「X」がレスト/アクティブ`）と枚数付き（`「X」が2枚以上いる`）の tuple バリアントを追加。`がある/がない` 表記にも対応。
- **FIELD_COUNT**: `いない`（否定）を `CompareOperator.LT` として正しく分岐するよう修正。
- **LEADER 系**: 特定色（LEADER_COLOR）、属性（LEADER_ATTRIBUTE）、LEADER_STATE（アクティブ/レスト/パワー閾値）を追加。
- **SOURCE_STATE**: パワー正規表現 `パワーが?(\d+)` に修正（`パワー0以下` など `が` 省略形に対応）。
- **DON_COUNT_COMPARE**: `ドン!!` 検出後に相手比較パターンで二段階振り分けし、既存 `DON_COUNT` との衝突を解消。

##### `_check_condition()` へのエバリュエーター追加（`resolver.py`）

6種の新 ConditionType に対応する評価ブランチを実装。また `_execute_game_action()` に以下のコンテキスト更新を追加:
- `context["_last_had_targets"]`: アクションにターゲットがあったか（`PREV_ACTION` 評価用）
- `context["last_revealed_card"]`: 最後に公開されたカード（`REVEALED_CARD_TRAIT` 評価用）

##### 置換効果（REPLACE_EFFECT）の保護者型対応（`parser.py` + `gamestate.py`）

前ブランチ(BWP0A)では「**このキャラ**が〜KOされる場合、代わりに」（自己置換）のみ対応していた。
本ブランチでは **「自分のコストX以上のキャラが相手の効果でKOされる場合、代わりに〜」（保護者型）** を実装:

1. **`parser.py` `_replacement_status()`**: `このキャラ` の存在チェックを撤廃。「代わりに〜」＋「場を離れる/KOされる」があれば REPLACE_EFFECT として識別。
2. **`parser.py` `parse_ability()`**: `このキャラ` を含む場合のみ condition をクリア（自己置換）。含まない場合は `OPPONENT_REMOVAL` 条件を保持（保護者型）。
3. **`gamestate.py` `_active_replacement()`**: スキャン対象を**除去されるカード自身のみ**から**オーナーの全フィールドカード＋リーダー**に拡張。各候補について `source_card=被除去カード` を渡して条件をライブ評価し、実行可能なら置換を実施する。

##### エンジンテストの追加（`test_effects_engine.py`: 56→67、+11件）

`test_prev_action_succeeded` / `test_prev_action_skipped` / `test_don_count_compare` /
`test_leader_state_is_active` / `test_leader_state_power_le` / `test_field_count_compare` /
`test_has_character_rested_state` / `test_revealed_card_trait_match` /
`test_revealed_card_trait_cost_and_type` / `test_opponent_removal_condition_power_filter` /
`test_opponent_removal_condition_trait_filter`

##### 残 GENERIC（1件、許容）

`自分のキャラすべては、このターン中、バトルでKOされない` — COUNTER 能力のテキスト。
`REPLACE_EFFECT(status="BATTLE_KO")` としての解析は正常に行われる。条件テキストが `相手の効果で` を含まないため `OPPONENT_REMOVAL` に分類されず GENERIC のまま。COUNTER はプレイヤーが明示的に発動するため、GENERIC=True（許容）であっても誤発動の実害はない。

---

### 本ブランチ（`claude/handoff-materials-review-BWP0A`）で追加した内容

#### 「手札捨て」split 断片問題の修正（parser.py + atoms.py, テスト +3）

**問題**: `_parse_to_node` の `split_pattern` に `捨て、` がデリミタとして含まれており、
`自分の手札1枚を捨て、このステージをレストにできる` のような起動コストで
`捨て、` ごと消費 → `自分の手札1枚を` が動詞なし断片として `unmatched` に落ちていた（11件）。

**修正（2箇所）**
- `parser.py _parse_to_node`: `捨て、` を `(?<=捨て)、`（lookbehind）に変更。
  `引く、` → `(?<=引く)、` と同じ方式で動詞 `捨て` を前クローズに保持する。
- `atoms.py discard` ルール: `捨てる` チェックを `捨て` に緩和（`捨て` は `捨てる`/`捨て`
  連用形/`捨ててもよい` 等をすべて包含）。手札チェック（`手札` in text）は維持。

**テスト +3**（golden）
- `discard_rest_stage_cost`: 手札捨て＋ステージレストをコストにした起動効果
- `discard_trash_self_cost`: 手札捨て＋このキャラをトラッシュをコストにした起動効果
- `discard_optional`: `捨ててもよい`（連用形）の任意コスト

**結果**: カバレッジ 92.5%→93.0%（+0.5pt）、OTHER 234→209（−25件）、退行=0。

#### ラウンド2: life_to_hand もよい / hand_to_deck / ドン!!スペース / 公開登場 / アクティブアタック（ルール 42→45, テスト +7）

**修正（2箇所）**
- `life_to_hand` ルール: `手札に加える` チェックに `手札に加えてもよい` を追加（`加えてもよい` は
  `加える` を含まないため連用形が落ちていた）。2件解消。
- `don_return` 正規表現: `ドン(?:!!|‼)` → `ドン[ 　]*(?:!!|‼)` に変更。`ドン !!-1` のように
  ドンと !! の間にスペースが入る表記（3件）に対応。

**新ルール（3種）— `rules/atoms.py`**
- `hand_to_deck`（priority=64）: 「自分の手札N枚を（並び替え、）デッキの上か下に置く」→
  DECK_BOTTOM(zone=HAND)。「並び替え」「上か下選択」は UI 未実装のため順序不定・デッキ下扱い。6件解消。
- `play_revealed`（priority=40）: 「（レストで）登場させてもよい」— ゾーン指定なし（デッキ公開
  →条件付き登場の文脈）→ PLAY_CARD(is_up_to=True, status=RESTED if レスト)。
  play_card_from_zone(52) が手札/トラッシュ明示を先に担当し、ここは残余ケース。5件解消。
- `attack_active`（priority=60）: 「アクティブのキャラにもアタックできる」→ GRANT_KEYWORD
  ("ATTACK_ACTIVE")。「このターン中」あれば THIS_TURN、なければ PERMANENT。6件解消。

**エンジン修正 — `gamestate.declare_attack`**
- `ATTACK_ACTIVE` キーワード持ちはアクティブキャラへのアタックを許可。
  従来: `not target.is_rest` で無条件 ValueError。
  修正: `attacker.has_keyword("ATTACK_ACTIVE")` を追加チェックで抑制。

**テスト +7**（golden +7: hand_to_deck_1 / life_to_hand_optional / don_return_space /
play_revealed_rested / attack_active_permanent / attack_active_this_turn / discard_optional 調整;
engine +2: test_hand_to_deck_bottom / test_attack_active_*）

**結果**: カバレッジ 93.0%→93.8%（+0.8pt）、OTHER 209→175（−34件）、退行=0。

#### ラウンド3: FREEZE / NEGATE_EFFECT / ルール処理 / 自己制限 / 各種 fix（ルール 45→49, テスト +12）

**修正（3箇所）— `rules/atoms.py`**
- `life_to_hand`: `ライフの上/下` 必須だった判定を拡張。`ライフN枚` 形式（例:「自分のライフ1枚を手札に加えることができる」）も対応。3件解消。
- `trash_self`: `置く` の必須チェックを削除。「このキャラをトラッシュに」（短縮形）・「代わりにこのキャラをトラッシュに」も拾う。4件解消。
- `life_to_trash`: `置く` チェックを `re.search(r"トラッシュに置")` に変更。「トラッシュに置いてもよい」等の活用形に対応。`is_up_to=True` も付与。2件解消。

**新ルール（4種）— `rules/atoms.py`**
- `freeze_target`（priority=65）: 「（相手の）レストのキャラ1枚までは、次の相手のリフレッシュフェイズでアクティブにならない」→ FREEZE(target=相手のレストキャラ, is_up_to=True)。エンジンの `refresh_all` が `card.flags["FREEZE"]` を確認してからリセットするため、ターン境界を跨ぐ `flags` に直接書き込む。4件解消。
- `negate_effect`（priority=65）: 「（相手の）リーダーかキャラ1枚までを、このターン中、効果を無効にする」→ NEGATE_EFFECT(target=相手, duration=THIS_TURN)。エンジンが `target.ability_disabled=True` をセットし、ability 発動をブロック。`reset_turn_status()` で解除。6件解消。
- `rule_processing`（priority=35）: 「ルール上、このカードはカード名を「X」としても扱う」「ルール上、デッキに何枚でも入れられる」等 → RULE_PROCESSING（エンジン no-op）。ゲームエンジンに影響しないルール注記を吸収。6件解消。
- `self_cannot`（priority=33）: 「自分は（このターン中）…できない/られない」→ RULE_PROCESSING（no-op）。自己制限メカニクス未実装のため解析のみで OTHER 脱出。4件解消。

**エンジン修正（3箇所）— `gamestate.apply_action_to_engine`**
- `FREEZE`: `target.flags.add("FREEZE")`。`refresh_all` が reset 前にフラグを読む設計に乗るため flags に書き込む（timed_flags でなく）。
- `NEGATE_EFFECT`: `target.ability_disabled=True`; `target._refresh_keywords()` でキーワードも無効化。
- `RULE_PROCESSING`: `success=True`（意図的 no-op）。

**テスト +12**（golden +8: freeze_rested_char / negate_effect_char / negate_effect_leader_or_char /
rule_card_alias / self_cannot_life_to_hand / life_to_hand_count_form / trash_self_short /
life_to_trash_optional;
engine +2: test_freeze_keeps_character_rested_after_refresh / test_negate_effect_sets_ability_disabled）

**結果**: カバレッジ 93.8%→95.4%（+1.6pt）、OTHER 175→108（−67件）、退行=0。

#### UI拡張フェーズ1: 凍結・効果無効オーバーレイ＋トリガーテキスト表示（フロントエンド + バックエンド API 拡張）

**背景**: バックエンドで FREEZE/NEGATE_EFFECT を実装したが、フロントエンドの Pixi.js 描画や
カード詳細モーダルには状態表示がなく、ゲーム中にどのカードが凍結・効果無効かが見えなかった。
また `trigger_text`（トリガー効果テキスト）も API から返っておらず、詳細シートに表示できていなかった。

**バックエンド変更（`opcg-sim-backend`）**

- `shared_constants.json`: `CARD_PROPERTIES` に `TRIGGER_TEXT` / `ABILITY_DISABLED` / `IS_FROZEN` を追加。
  両者間で API フィールド名を一元管理している定数ファイルであり、フロント/バック双方がこれを参照する。
- `opcg_sim/src/models/models.py`: `CardInstance.to_dict()` に 3 フィールドを追加。
  - `trigger_text`: `self.master.trigger_text or ''`（トリガー効果テキスト）
  - `ability_disabled`: `self.ability_disabled`（効果無効フラグ）
  - `is_frozen`: `'FREEZE' in self.flags`（凍結フラグ）

**フロントエンド変更（`opcg-sim-frontend`）**

- `src/game/types.ts`: `BaseCard` に `trigger_text?: string` / `ability_disabled?: boolean` /
  `is_frozen?: boolean` を追加（API 新フィールドに型を付与）。
- `src/layout/layout.config.ts`: 状態バッジ用カラー定数を追加。
  - Pixi 用数値 (0xRRGGBB): `BADGE_FROZEN_BG: 0x2980b9` / `BADGE_NEGATE_BG: 0x7f8c8d`
  - CSS 用文字列: `BADGE_FROZEN_CSS: '#2980b9'` / `BADGE_NEGATE_CSS: '#7f8c8d'`
- `src/ui/CardRenderer.tsx`: 重なり枚数バッジの後に状態オーバーレイを追加。
  - `card.is_frozen` が真: 青半透明矩形（alpha 0.3）＋「凍結」ラベル（`'screen'` 回転モード）。
  - `card.ability_disabled` が真: グレー半透明矩形（alpha 0.3）＋「効果無効」ラベル。
  - 両方同時に有効な場合は Y 座標を ±ch×0.12 にずらしてラベルが重ならないよう調整。
- `src/ui/CardDetailSheet.tsx`: 詳細モーダルの表示を拡張。
  - バッジ行（属性・特徴の隣）に `凍結` / `効果無効` 状態バッジを追加。
  - 効果テキスト `<p>` の直後に `trigger_text` ブロックを追加。
    `【トリガー】`（赤太字）＋ テキスト本文を区切り線付きで表示（`trigger_text` が空文字の場合は非表示）。

#### UI拡張フェーズ2: 効果解決ログパネル（フロントエンド + バックエンド API 拡張）

**背景**: カード効果が発動・解決されても UI 側には何も表示されず、何が起きているか追いにくかった。
バックエンドの `EffectResolver.action_history` には既に per-action の解決履歴が蓄積されていたが、
API レスポンスには含まれておらず、フロントに届いていなかった。

**バックエンド変更（`opcg-sim-backend`）**

- `opcg_sim/src/core/gamestate.py`:
  - `GameManager.__init__` に `self.action_events: List[Dict] = []` を追加（per-request バッファ）。
  - `resolve_ability` を修正: `EffectResolver` インスタンスの `action_history` を解決後に
    `action_events` へコピー（フィールド: type="EFFECT", player, card_name, action, targets, value, success）。
- `opcg_sim/api/app.py`:
  - `build_game_result_hybrid` の戻り値に `action_events` キーを追加。
  - `game_action` ハンドラ: try ブロック先頭で `manager.action_events = []` リセット。
    各アクション（PLAY/ATTACK/TURN_END/ATTACH_DON/ACTIVATE_MAIN）に日本語メッセージ付きイベントを追加。
  - `game_battle` ハンドラ: 同様に BLOCK/COUNTER/PASS イベントを追加。

**フロントエンド変更（`opcg-sim-frontend`）**

- `src/api/types.ts`: `ActionEvent` インターフェース追加。`GameActionResult` に `action_events?` フィールド追加。
- `src/api/client.ts`: `sendAction` / `sendBattleAction` の戻り値に `action_events` を含める。
- `src/game/actions.ts`: `useGameAction` に `addEventLog?` コールバックを追加。
  各アクション後、`result.action_events` があれば `addEventLog` を呼び出す。
- `src/ui/ActionLog.tsx`（新規）: 右上固定の折りたたみ式ログパネル。
  - アクションタイプ（PLAY=緑/ATTACK=赤/TURN_END=グレー等）でカラーコーディング。
  - EFFECT サブタイプ（DRAW/KO/BOUNCE/BUFF/FREEZE/NEGATE_EFFECT 等）は日本語ラベルに変換。
  - 失敗イベント（`success=false`）は半透明表示。
  - 最大50件を保持（古いものを押し出し）。
- `src/screens/RealGame.tsx`: `eventLog` ステート + `addEventLog` コールバックを追加。
  `useGameAction` に渡し、`<ActionLog events={eventLog} />` を DOM オーバーレイとしてレンダリング。

#### UI拡張フェーズ3: PendingRequest UI 改善

**背景**: 効果発動中に表示されるインタラクション要求 overlay が `[SELECT_BLOCKER] 対象を選択してください（最大1枚）` のように、
生の enum 名を見せており、どのカードの効果でどう操作すればよいかが分かりにくかった。
バトル表示も UUID プレフィックス（`ATTACK: 12345678 → abcd1234`）を使っていた。

**バックエンド変更（`opcg-sim-backend`）— `resolver.py`**

- `_suspend_for_target_selection`: メッセージを改善。
  - Before: `"対象を選択してください（最大N枚）"`
  - After: `"「カード名」の効果: 対象を選択（N枚まで）"`（N=1のときも対応、up-to=True で「まで」付与）
  - `active_interaction` に `source_card_name` フィールドを追加（将来の参照用）。
- `_suspend_for_choice`: 選択肢メッセージに `"「カード名」の効果: "` を前置。`node.message` が空の場合は `"選択してください"` にフォールバック。

**フロントエンド変更（`opcg-sim-frontend`）— `RealGame.tsx`**

- `PENDING_ACTION_LABELS` 定数: SELECT_BLOCKER/SELECT_COUNTER/SEARCH_AND_SELECT/CHOICE 等の
  生 enum 文字列を日本語サブヘッダーラベルにマッピング。
- `resolveCardName(uuid, gameState)` ヘルパー: gameState の全ゾーン（leader/field/hand/life/trash、両プレイヤー）
  を横断して UUID からカード名を逆引き。見つからない場合は UUID の先頭 8 文字を返す。
- PendingRequest overlay の改善:
  - `[SEARCH_AND_SELECT]` 等の生ラベルを `PENDING_ACTION_LABELS` に基づく日本語サブヘッダー（小さめ、灰色）に変更。
  - 主テキストを `pendingRequest.message` のみに（バックエンド改善でソースカード名が埋め込まれる）。
  - バトル情報の表示: `ATTACK: 12345678 → abcd1234` → `⚔ 「ルフィ」→「カイドウ」`。
  - overlay に `maxWidth: 320px` を追加して長いメッセージでも整形されるよう調整。

**シリアライズ負荷の分析（この変更に付随）**
- ゲーム中の状態更新（約 46 枚）: 3 フィールド追加で raw +10 KB、gzip +0.6 KB → 実質無視できる。
- 全カードリスト（2652 枚）: raw +535 KB、gzip +31 KB → 許容範囲（日本語は gzip で約 70% 圧縮）。
- Cache-Control の追加実装は不要と判断（既存のキャッシュ設定で十分）。

---

### 前ブランチ（`claude/handoff-materials-review-u5cqy`）で追加した内容（最新順）

---

#### ラウンド4: サーチ構造修正（「デッキを見て→公開し手札に加える」, ルール 39→42, parser.py 構造分解に着手）

**最大の構造的修正**。従来 `_parse_to_node` の分割パターンに「見て、」が無く、
「デッキの上からN枚を見て、…M枚までを公開し、手札に加える」が**1原子句化**して
`parse_target` が「N枚」を count に誤取得 → 誤った BOUNCE(対象=FIELD) を生成し LOOK が欠落していた。

**parser.py（構造分解）の修正 — `_parse_to_node`**
- 「デッキの上から\d+枚を見て、」の読点を「。」へ置換し、**LOOK を独立クローズに分割**。
  ライフ等の他の「見て、」（ライフ6枚・その他9枚）には影響させないようデッキ文脈に限定
  （全2652カードへの影響を最小化。`compare_parsers` 退行=0 を確認済み）。

**パーサルール追加（3種）— `rules/atoms.py`**（ルールが `Sequence` でなく単一 GameAction を返す
原則は維持。分割で生じた各クローズを個別ルールが解釈する）
- `look_deck`: 「デッキの上からN枚を見て」→ LOOK（デッキ上 N 枚→TEMP）。
- `search_to_hand`: 「（公開し、）（コスト/特徴/名前で絞った）カードM枚までを手札に加える」→
  MOVE_CARD(zone=TEMP, dest=HAND)。明示ソース（トラッシュ/ライフ/手札から/デッキ）がある句は除外。
  分割により count 誤取得も解消（「N枚」は前クローズ、grab は「M枚」を正しく取得）。
- `temp_to_deck`: 「（好きな順番に並び替え、）デッキの上か下に置く」→ DECK_BOTTOM(TEMP 全件)。
  **scry の戻し**。これが無いと LOOK で TEMP に出したカードが戻されず TEMP リークになる。
  「残り」を含む句は remaining_* が担当。

**重要な設計ポイント — TEMP リーク防止**
- LOOK は候補を TEMP(temp_zone) に移す。後続で TEMP を必ず消費しないとカードが TEMP に
  取り残される（デッキから消失するゲーム上のバグ）。サーチ系は grab(search_to_hand)＋
  remaining_*、scry 系は temp_to_deck が全件を戻すことで TEMP を空にする。

**テスト（111→115）**
- golden +3（deck_search_to_hand / deck_scry_rearrange / deck_search_trait）。
- engine +1（LOOK→grab→DECK_BOTTOM の一連フローで TEMP リーク無しを検証）。
- カバレッジ 87.6→92.5%（+4.9pt）、OTHER 263→234（−29）。退行=0。
- `compare_parsers`: 改善(OTHER解消)が 449→735 に増加（サーチ214枚の構造が正常化）。
  完全一致が減るのは構造刷新によるもので退行ではない。

---

#### ラウンド3: reveal_hand（手札公開, ルール 38→39）

**パーサルール追加（1種）— `rules/atoms.py`**
- `reveal_hand`: 「自分の手札から（コスト/パワー/特徴で絞った）カードN枚を公開する/できる/する
  ことができる」→ REVEAL(zone=HAND)。「公開し、手札に加える」（デッキを見てのサーチ）や
  「手札に戻す」は除外。従来は OTHER（公開できない no-op）、または「パワーN…公開」が誤って
  BUFF に落ちていたのを修正（14件前後）。

**エンジン実行系の追加 — `gamestate.apply_action_to_engine`**
- `REVEAL`（新規）: 盤面を動かさず公開した事実をログに残す（情報開示。手札に残る）。

**テスト（107→111）**
- golden +2（reveal_hand_events / reveal_hand_power_char）、engine +2（手札に残る/対象不在 no-op）。
- カバレッジ 87.1→87.6%、OTHER 281→263（−18）。退行=0。

> 注: 「デッキの上からN枚を見て…公開し、手札に加える」（サーチ）は **未対応のまま**。
> 構造分解で LOOK が欠落し、対象が誤って FIELD/BOUNCE になる構造的問題（§7-参照）。
> reveal_hand は「手札に加える」を含むこのサーチ系を意図的に除外している。

---

#### ラウンド2: bounce・deck_bottom・play_card_from_zone・active_target・blocker_disable・rush_natural（ルール 30→38）

**パーサルール追加（8種）— `rules/atoms.py`**
- `bounce`: 「（コストN以下の）キャラ1枚までを、持ち主の手札に戻す」→ BOUNCE。
  「自分の」明示がなければ OPPONENT をデフォルト（「持ち主の手札」＝相手カード対象が多数派）。
- `deck_bottom_general`: 「（コストN以下の）キャラを持ち主のデッキの下に置く」「自分の手札N枚をデッキの下に置く」「相手は自身の手札1枚をデッキの下に置く」→ DECK_BOTTOM。
  「持ち主」+プレイヤー未指定で OPPONENT に補正。「残り」は除外し remaining_* ルールに委ねる。
- `remaining_deck_top_or_bottom`: 「残りをデッキの上か下に置く」→ DECK_BOTTOM（保守的）。
  「上か下」の選択 UI は未実装のためデッキ下扱いとする。
- `play_card_from_zone`: 「（手札/トラッシュ）からコストN以下のキャラカード1枚までを（レストで）登場させる（ことができる/てもよい）」→ PLAY_CARD(zone=HAND/TRASH, dest=FIELD)。
  レスト登場は status="RESTED" をエンジンに伝え is_rest=True にセット（エンジン側も対応追加）。
- `active_target`: 「自分のキャラ1枚までを、アクティブにする/できる」→ ACTIVE（非自己・非ドン）。
  active_self(priority=75)/don_set_active(priority=74)より低い priority=51 で衝突しない。
- `blocker_disable`: 「相手は（このバトル中）【ブロッカー】を発動できない」→ BUFF(BLOCKER_DISABLE)。
  エンジンの BLOCKER_DISABLE ブランチが対象フィールド全体に "BLOCKER_DISABLED" フラグを立てる。
- `rush_natural`: 「登場したターンにキャラへアタックできる」→ GRANT_KEYWORD("速攻", PERMANENT)。
  【速攻】タグを持たない自然言語表現からキーワード付与を生成。
- `mill_deck`（拡張）: 「置き（連用形）」「置いてもよい」等の活用形に対応。
  `re.search(r"トラッシュに置|トラッシュに$")` で活用形・文末「に」も拾う。

**エンジン実行系の追加 — `gamestate.apply_action_to_engine`**
- PLAY_CARD + status="RESTED": レストで登場させる時 `target.is_rest = True`。

**テスト（103→107）**
- golden +8（active_target / blocker_disable / rush_natural / mill 連用形 / bounce×2 / deck_bottom×2 / remaining_deck_top_or_bottom / play_from_hand / play_from_trash）
- 退行(新規OTHER)=0を維持。カバレッジ 85.1%→87.1%、OTHER 322→281（−41）。

---

#### ラウンド1: 診断上位5表現のルール化

引き継ぎ資料の TDD サイクル（§5）に沿い、診断「未対応(フォールバック)原子句ランキング」
上位5表現をルール化（ルール 25→30）。全表現がランキングから消えたことを確認済み。

**パーサルール追加（5種）— `rules/atoms.py`**
- `trash_self`: 「このキャラ/カード/リーダーをトラッシュに置く（ことができる）」（最頻出49件）。
  KO ではなく単純移動（ON_KO 不誘発）。対象は自身(SOURCE)。「このキャラ**以外**を…」は
  直後が「を(、)?トラッシュ」のものに限定して巻き込まない。
- `active_self`: 「このキャラ/カード/リーダーをアクティブにする/できる」（27件）。対象は自身。
  ドン!!のアクティブ化(`don_set_active`)とは「ドン」の有無で区別。
- `rest`（拡張）: 正規表現を `レストに(する|できる|し[、。])` に拡張。従来「できる」を取りこぼし、
  「このステージをレストにできる」等(20件)が OTHER 化していた。相手キャラの「レストにできる」も
  併せて拾えるようになった（rest 命中 143→211）。
- `mill_deck`: 「（自分/相手の）デッキの上からN枚をトラッシュに置く」（11件）→ `TRASH_FROM_DECK`。
  デッキは並びが意味を持つため対象選択させず枚数(value)ベース。「相手は…」は status="OPPONENT"。
- `remaining_trash`: 「残りを（好きな順番で）トラッシュに置く」（18件）→ TRASH(TEMP/REMAINING)。
  `remaining_deck_bottom`(残り→デッキ下)のトラッシュ版。

**エンジン実行系の新設 — `gamestate.apply_action_to_engine`**
- `TRASH_FROM_DECK`（新規）: デッキ上から value 枚をトラッシュへ送る（mill）。status="OPPONENT"
  で相手デッキ対象。**従来 ActionType は生成されても実行系が無くサイレント no-op だった**のを修正。

**テスト（81→91）**
- golden +5（`trash_self_cost`/`active_self`/`rest_stage_can`/`mill_deck_top`/`remaining_trash`）。
- engine +5（mill の枚数/デッキ枯渇/相手対象、自己トラッシュ、自己アクティブの盤面検証）。
- `compare_parsers.py` 退行(新規OTHER)=0 を維持。

---

## 3. アーキテクチャ全体像

### 効果処理のパイプライン

```
カードDB(日本語テキスト)
   │  loader.py: _create_card_master() / make_parser()
   ▼
[ catalog.py 手動定義があれば優先 ] ─ なければ ─▶ [ EffectParserV2 ]
   │                                                   │ 構造分解(レガシー流用)
   │                                                   │ + 原子句のみ rules で解釈
   │                                                   │ + 未対応はレガシーへフォールバック
   ▼                                                   ▼
   └──────────────▶  Ability(IR) ◀──────────────────────┘
                        │ trigger / condition / cost / effect
                        ▼  ゲーム中、該当タイミングで
                  resolver.py（EffectResolver）
                        │  AST を実行スタックで処理（対象選択は中断/再開）
                        ▼
                  gamestate.py（apply_action_to_engine / continuous / 除去保護）
                        ▼
                     盤面更新
```

### V2 の設計思想（最重要）

`EffectParserV2` は `EffectParser`(レガシー) を**継承**し、`_parse_atomic_action()` だけを
オーバーライドする。トリガー判定・コスト分離・逐次/分岐/選択肢の構造分解はレガシーをそのまま使う。

- 原子句は `default_registry.apply(ctx)` でルール優先解釈
- どのルールも当たらなければ **レガシー実装にフォールバック**し、その句を `unmatched` に記録
- → 本番は決して壊れない。未対応表現は診断で可視化され、ルール追加で burn down できる

### 継続効果（期間付き効果）

`effects/continuous.py` の `ContinuousEffectManager`。

- `CardInstance` の専用フィールド `timed_power` / `timed_cost` / `timed_flags` /
  `timed_keywords` に反映。**これらは `reset_turn_status()` でクリアされない**
  （ターン境界を跨いで存続する鍵）。既存の `power_buff`/`cost_buff`/`flags`/`current_keywords`
  （ターン境界 or passive 再計算でリセット）とは独立で衝突しない。
- kind: `POWER` / `COST` / `FLAG` / `KEYWORD`。Duration: `THIS_BATTLE` / `THIS_TURN` /
  `UNTIL_NEXT_TURN_END` / `PERMANENT`（場を離れるまで持続）。
- 失効は `expire(event)` を **バトル終了**(`resolve_attack`)・**ターン終了**(`end_turn`)で呼ぶ。
  カードが場を離れる際は `move_card` が `drop_for(uuid)` を呼び、その分を破棄する。
- 参照側: `get_power()`=`timed_power` 加算、`current_cost`=`timed_cost` 加算、
  `has_keyword()`=`current_keywords ∪ timed_keywords`、アタック制限=`timed_flags`。

### 除去保護（PREVENT_LEAVE）と置換効果（REPLACE_EFFECT）

`gamestate._active_protection(card, status)` / `_active_replacement(card, status)`。除去が
起こる瞬間に対象の PASSIVE 能力を走査し、条件（例: トラッシュ7枚以上）を
`EffectResolver._check_condition` で**ライブ評価**する（フラグをラッチしないので条件変動に追随）。
- 保護 `PREVENT_LEAVE`: `status="LEAVE"`（相手の効果で場を離れない）/ `"BATTLE_KO"`
  （バトルでKOされない）。
- 置換 `REPLACE_EFFECT`: 「代わりに〜」。実行可能性（`_can_satisfy_node`）も満たせば
  `sub_effect`（置換アクション）を実行し本来の除去をスキップ。同じ `LEAVE`/`BATTLE_KO`
  フックに相乗り（保護を先に判定、無ければ置換を判定）。

---

## 4. ファイルマップ

### 本番コード

| パス | 役割 |
|---|---|
| `opcg_sim/src/core/effects/parser.py` | レガシーパーサ（構造分解を担当・V2が継承） |
| `opcg_sim/src/core/effects/parser_v2.py` | **新パーサ**。原子句をレジストリ化＋フォールバック記録 |
| `opcg_sim/src/core/effects/rules/base.py` | `Rule`/`RuleRegistry`/`ParseContext`/`@rule` |
| `opcg_sim/src/core/effects/rules/atoms.py` | **原子アクションルール群（ここを育てる）** |
| `opcg_sim/src/core/effects/continuous.py` | 継続効果マネージャ |
| `opcg_sim/src/core/effects/matcher.py` | 対象指定の解析(`parse_target`)・実体化(`get_target_cards`) |
| `opcg_sim/src/core/effects/resolver.py` | IR の実行（EXECUTE_MAIN_EFFECT 等もここ） |
| `opcg_sim/src/core/effects/catalog.py` | 手動オーバーライド(MANUAL_EFFECTS, 13枚) |
| `opcg_sim/src/core/gamestate.py` | ゲームエンジン本体（apply_action_to_engine / 除去保護 / 継続効果フック） |
| `opcg_sim/src/models/effect_types.py` | IR 定義（Ability/GameAction/TargetQuery/Condition…）。`GameAction.sub_effect`（置換用） |
| `opcg_sim/src/models/models.py` | CardMaster/CardInstance（`timed_power`/`timed_cost`/`timed_flags`/`timed_keywords`、`has_keyword()`） |
| `opcg_sim/src/models/enums.py` | ActionType/TriggerType/Zone… |
| `opcg_sim/src/utils/loader.py` | カードDB/デッキ読込・`make_parser()` ファクトリ |

### テスト・ツール

| パス | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサの単体テスト（8件） |
| `tests/golden/golden_cases.py` | **ゴールデンコーパス（効果セマンティクスの期待値, 88件）** |
| `tests/golden/summarize.py` | AST→指紋(summary) 変換＋部分一致判定 |
| `tests/test_golden.py` | ゴールデン・ランナー（79件） |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化テスト（67件） |
| `tests/test_gameplay_smoke.py` | 実デッキでのゲーム進行スモーク（2件） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ |
| `tests/effect_diagnostics.py` | **未対応句/OTHER ランキングの可視化** |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |

### フロントエンド（opcg-sim-frontend）— UI拡張フェーズで追加・変更

| パス | 役割 |
|---|---|
| `src/game/types.ts` | `BaseCard` に `trigger_text`/`ability_disabled`/`is_frozen` を追加 |
| `src/api/types.ts` | `ActionEvent` 型・`GameActionResult.action_events` フィールド |
| `src/api/client.ts` | `sendAction`/`sendBattleAction` の戻り値に `action_events` を含める |
| `src/game/actions.ts` | `useGameAction` に `addEventLog` コールバックを追加 |
| `src/layout/layout.config.ts` | `BADGE_FROZEN_BG/CSS`・`BADGE_NEGATE_BG/CSS` 色定数を追加 |
| `src/ui/CardRenderer.tsx` | `is_frozen`/`ability_disabled` の Pixi 半透明オーバーレイを追加 |
| `src/ui/CardDetailSheet.tsx` | 状態バッジ（凍結/効果無効）・`trigger_text` ブロックを追加 |
| `src/ui/ActionLog.tsx` | **効果解決ログパネル**（新規。右上固定・折りたたみ式） |
| `src/screens/RealGame.tsx` | `eventLog` ステート・`<ActionLog>` レンダリング・overlay 改善 |
| `shared_constants.json` | `TRIGGER_TEXT`/`ABILITY_DISABLED`/`IS_FROZEN` を `CARD_PROPERTIES` に追加 |

---

## 5. 開発フロー（ルール追加 TDD サイクル）

```bash
# 1) 標的を選ぶ（OTHERランキング上位＝効果が動かない直接原因）
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40

# 2) ゴールデンケースを追加して赤にする
#    tests/golden/golden_cases.py に text と期待 summary を書く
OPCG_LOG_SILENT=1 python tests/test_golden.py

# 3) ルールを足して緑にする
#    opcg_sim/src/core/effects/rules/atoms.py に @rule を1つ追加
#    （エンジン側の実行が必要なら gamestate/resolver も実装し test_effects_engine に検証追加）

# 4) 回帰・退行・カバレッジ確認
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider
OPCG_LOG_SILENT=1 python tests/compare_parsers.py     # 退行(新規OTHER)=0 を維持
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py  # 命中率↑/OTHER↓
```

ルールは `@rule(name, priority)` で関数登録。`priority` が大きいほど先に試行
（具体的・限定的なルールを高く）。不一致なら `None`、一致なら `EffectNode` を返す。

---

## 6. 運用（環境変数）

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_PARSER` | `v2` | `legacy` でレガシーパーサへ**即ロールバック**（再デプロイ不要） |
| `OPCG_LOG_SILENT` | （未設定） | `1` で stdout ログ抑止（テスト/診断用。バッファ蓄積は維持） |

**ロールバック手順**: 本番で問題が出たら Cloud Run の環境変数に `OPCG_PARSER=legacy` を
設定するだけ。V2 読込失敗時も自動でレガシーへ退避する（フェイルセーフ）。

---

## 7. 既知の課題・残タスク（優先度順）

### UI拡張フェーズ（進行状況）

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 1 | FREEZE/NEGATE オーバーレイ、trigger_text 表示、API シリアライズ拡張 | **完了** |
| Phase 2 | 効果解決ログ（バックエンド: レスポンスに解決ログ追加、フロント: ログパネル表示） | **完了** |
| Phase 3 | PendingRequest UI 改善（どの効果でどのカードが対象か、より具体的なメッセージ） | **完了** |

### ★ 隠れミスターゲット／lift 不具合の横展開調査（OTHER指標に出ない不具合）

**背景**: PHoSv ラウンド1-2 で、`OTHER` には現れないが**意味的に壊れている**不具合を2類型発見した:
- **ミスターゲット型**: ルール未命中でレガシーへフォールバックした句が、`PLAY_CARD` 等の
  ActionType は正しく出すが**対象 zone を文言依存で誤推定**（FIELD/DECK など）。実行はされるが
  盤面操作が誤る（例: デッキ公開→登場の7枚が `zone=FIELD/DECK`、ライフ操作の上下区別欠落 等）。
- **条件 lift 型**: インライン条件（「公開し→その結果で条件分岐→実行」）が、先頭ゲート条件の
  lift でアビリティ条件へ引き上げられ、**前段アクション（公開/LOOK）が消失**し順序矛盾を起こす。

いずれも `effect_diagnostics.py`（OTHER カウント）や `compare_parsers.py`（新規OTHER検知）では
**捕捉できない**ため、専用の検出が要る。

**やること（横展開）**:
1. **検出ツールの追加** — `tests/` に「V2 が生成した `PLAY_CARD`/`MOVE_CARD`/`BOUNCE`/`KO` 等の
   対象 zone とテキストの整合を突き合わせ、疑わしい組み合わせ（例: `PLAY_CARD zone=FIELD`、
   `公開/見て` を含むのに LOOK 無し、`そのカード`条件が ability.condition に lift されている等）を
   洗い出す」診断を新設する（OTHER ランキングと同様にカード番号付きで列挙）。
2. **既知パターンの洗い出し** — まず以下を機械的に列挙し、件数を可視化:
   - `PLAY_CARD` で `target.zone == FIELD`（プレイ元が場＝ほぼ誤り）。
   - `ability.condition.type == REVEALED_CARD_TRAIT`（lift 済み＝公開消失の疑い）。
   - `公開し/見て` を含むカードで `LOOK` アクションが生成されていない。
3. **既存対応の横展開** — 検出された各クラスタに対し、本ブランチと同型（公開句の分割＋TEMP化、
   インライン Branch 化、対象 zone の正規化）で順次是正する。
4. **回帰防止** — 上記診断をスモークに組み込み、`PLAY_CARD zone=FIELD` などの「禁止パターン」
   件数が増えたら気づけるようにする（compare_parsers の OTHER 退行検知の対象外領域を補完）。

> 着手の起点: 本ブランチ PHoSv のラウンド1-2（§本ブランチ）が具体例＋修正パターンの雛形。

1. **裾野の OTHER／フォールバックのルール化** — 頻度は低く多様（上位でも10件前後/表現）。
   `effect_diagnostics.py` の「未対応(フォールバック)原子句ランキング」「OTHER化する原子句
   ランキング」を起点に継続。本セッションで上位5表現（自己トラッシュ/自己アクティブ/
   ステージのレスト/デッキ→トラッシュ mill/残り→トラッシュ）は **対応済み**。
   bounce / deck_bottom / play_card_from_zone / active_target / blocker_disable /
   rush_natural / reveal_hand / **サーチ（デッキを見て公開し手札に加える）/ scry** は
   **対応済み**（本セッション）。残るのは**構造的/UI/専用メカニクス課題が中心**で、
   単純なルール追加では解けないものが上位を占める:
   - ~~**構造断片「自分の手札1枚を」（11件）**~~ **対応済み**（本ブランチ ラウンド1）。
   - ~~**手札→デッキ上か下（6件）**~~ **対応済み**（本ブランチ ラウンド2, `hand_to_deck`）。
   - ~~**公開カードをレストで登場させてもよい（5件）**~~ **対応済み**（本ブランチ ラウンド2, `play_revealed`）。
   - ~~**アクティブのキャラにもアタックできる（6件）**~~ **対応済み**（本ブランチ ラウンド2, `attack_active`）。
   - ~~**「次のリフレッシュフェイズでアクティブにならない」（4件）**~~ **対応済み**（本ブランチ ラウンド3, `freeze_target`）。
   - ~~**「効果を無効にする」（相手キャラ/リーダー, 6件）**~~ **対応済み**（本ブランチ ラウンド3, `negate_effect`）。
   - ~~**「自分は〜できない」形の自己制限（4件）**~~ **対応済み**（本ブランチ ラウンド3, `self_cannot`→RULE_PROCESSING）。
   - **ライフ look-and-place「自分か相手のライフの上から1枚を見て、ライフの上か下に置く」（10件）** —
     ライフは TEMP を介さず上下選択 UI も未実装。デッキサーチとは別系統の設計が要る。
   - **「任意のコストを宣言し、相手のデッキの上から1枚を公開する」（6件）** — コスト宣言という
     ゲーム独自メカニクス＋専用 ActionType の設計が必要。
   - **「パワーが相手と同じになる」（3件, 動的値）** — 対象の base_power_override を
     ゲーム中に動的評価する必要がある。未実装。
   - ~~**「デッキの上から1枚を公開し、コスト2のキャラ1枚までを登場させる」（2件）**~~ **対応済み**
     （本ブランチ PHoSv ラウンド1, `公開し、`分割＋`look_deck`拡張＋`play_from_temp`新設）。
     残: 任意登場「…の場合、登場させてもよい」（OP12-058）・scry-1（OP08-049）・`play_revealed`単一句の TEMP 化。
   - キーワード付与（【ブロッカー】等を得る）は **対応済み**（`grant_keyword`）。
     `GRANT_KEYWORD` は継続効果マネージャ経由で `timed_keywords` に付与され、
     `_apply_passive_effects` のリセットで消えず、duration（THIS_TURN/THIS_BATTLE/
     PERMANENT）で失効する（下記2のうち KEYWORD は対応済み）。
   - ライフ操作（デッキ↔ライフ↔手札／トラッシュ／表・裏向き）は **対応済み**（`life_*`）。
     残: 「ライフを見て上か下に置く」等の look-and-place 系。なお MOVE_CARD は
     `dest_position` フィールドを持たず常に末尾（下）へ入るため「ライフの上に加える」の
     上下区別は未対応（既存制約）。
   - ドン!!操作（付与／アクティブ／レスト／ドンデッキに戻す）は **対応済み**（`don_*`、
     枚数ベース）。残: REST_DON をコストにする句の充足判定（現状 target=None のため
     `_can_satisfy_node` がドン枚数を検証せず常に True）。
2. ~~**COST/KEYWORD の duration 対応**~~ **対応済み**（POWER/FLAG も含め継続効果マネージャに
   統合）。`_apply_passive_effects` が `cost_buff`/`current_keywords` を毎回リセットして
   期間付き効果が消える問題を、専用フィールド（リセット対象外）で解決:
   - KEYWORD → `timed_keywords`（`has_keyword()` で本来＋付与分を参照）
   - COST → `timed_cost`（`current_cost` に加算。期間付きのみ継続効果へ、INSTANT は
     従来どおり `cost_buff`＝PASSIVE 再計算で再適用）
   いずれも `drop_for` で場を離れる際に破棄。残: COST のうち PASSIVE（条件付き常時）の
   duration 統合は対象外（reset+reapply で正しく機能するため不要）。
3. **置換効果（「代わりに〜」）（保護者型も対応済み）** — 「このキャラが(バトルで)?KOされる/
   場を離れる場合、代わりに〜」を `REPLACE_EFFECT`（置換アクションを `sub_effect` に保持）
   として実装。`_active_replacement` が除去の瞬間に PASSIVE 能力を走査し、条件・実行可能性
   （`_can_satisfy_node`）を満たせば置換を実行し本来の除去をスキップする。
   `バトル`→`BATTLE_KO`（戦闘KO）/ それ以外→`LEAVE`（相手効果除去）。
   **本ブランチ（bIYkG）で保護者型（自分の他キャラを守る型）に対応**:
   `_active_replacement` がオーナーの全フィールドカード＋リーダーをスキャンし、
   被除去カードを `source_card` に渡して `OPPONENT_REMOVAL` 条件をライブ評価する。
   - 残: ①置換実行が対象選択で中断する場合（複数候補の捨て札等）の挙動は要検証。
     ②「できる」（任意）の選択UIは未提供（取れるなら実行）。
4. ~~**ターン1回制限の enforce**~~ **対応済み**。`resolver.resolve_ability` が
   `TURN_LIMIT` を検出し `source_card.ability_used_this_turn[ability位置]` で
   使用回数を管理する（条件・コストを満たし発動成立した時点で消費）。カウンタは
   `reset_turn_status`（毎ターン境界で両者に呼ばれる）でクリアされ、ターン単位で機能。
5. **条件の fail-safe 化＋分類拡充（ほぼ完了）** — 真に解釈不能な `OTHER` は False に
   倒す（誤発動防止）。`GENERIC`（実在するが未分類の条件）は一律 False にすると多数の
   効果が永久不発になり有害なため許容(True)＋ログ可視化に留め、**評価可能なクラスタを
   個別に実条件へ分類**して誤発動源を減らす方針。分類実績で **GENERIC 251→1**
   （main で 251→94、本ブランチで 94→1）:
   - リーダー特徴の `『X』` 記法 → `LEADER_TRAIT`（18件）
   - 盤面のキャラ枚数「(レストの/特徴《X》の/コストN以上の)キャラがM枚以上/以下いる/がいる」
     → `FIELD_COUNT`（target フィルタ対応, 85件）
   - デッキ枚数「デッキがN枚以下/以上」→ `DECK_COUNT`（5件）
   - リーダー多色/特定色 → `LEADER_COLOR`（11件+）
   - リーダー属性 → `LEADER_ATTRIBUTE`
   - リーダー状態（アクティブ/レスト/パワー） → `LEADER_STATE` (**本ブランチ**)
   - 自分/相手のドン!!枚数比較 → `DON_COUNT_COMPARE` (**本ブランチ**)
   - 自分/相手のキャラ数比較 → `FIELD_COUNT_COMPARE` (**本ブランチ**)
   - 直前アクション成否 → `PREV_ACTION` (**本ブランチ**)
   - 公開カードの特徴/コスト/タイプ → `REVEALED_CARD_TRAIT` (**本ブランチ**)
   - 相手効果/バトルによる除去置換条件 → `OPPONENT_REMOVAL` (**本ブランチ**)
   - 残1件: COUNTER 能力内の `バトルでKOされない` — 許容（§本ブランチ参照）。
6. **catalog の縮退** — parser が賢くなった分、`MANUAL_EFFECTS`(13枚) を1枚ずつ
   golden で検証しながら削れる。

---

## 8. 注意点・落とし穴

- **本番パスは loader 経由**。catalog(手動定義) > parser(V2) の優先順位（`loader._create_card_master`）。
- **テキスト正規化**: パーサは NFC、loader の DataCleaner は NFKC を使う箇所がある。
  全角/半角・`!!`/`‼`(U+203C)・各種マイナス記号の揺れに注意（ルールの正規表現は両対応にする）。
- **pytest の出力キャプチャ**: logger が `sys.stdout` を直接掴むため、`pytest` は
  `-s`（キャプチャ無効）で実行する。`OPCG_LOG_SILENT=1` 併用推奨。
- **`timed_*`（power/cost/flags/keywords）は reset_turn_status でクリアしない**設計。ここを
  「リセット対象に追加」してしまうと複数ターン跨ぎ効果・付与キーワード・期間付きコストが壊れる。
- **`_apply_passive_effects` は cost_buff/current_keywords を毎回リセット**する（power_buff/flags
  はしない）。期間付きの COST/KEYWORD はこのリセットを避けるため `timed_cost`/`timed_keywords`
  に載せている（直接 cost_buff/current_keywords へ加えると即消える）。INSTANT/PASSIVE の
  コスト・キーワードは従来どおり reset+reapply で機能する。
- **CardMaster は frozen dataclass**。abilities は生成時に確定。テストで能力を差し替える
  場合は `make_master(..., abilities=(...))` で構築する。
- **新パーサの効果は V2 有効化後にのみ反映**。`OPCG_PARSER=legacy` 時はレガシー解釈に戻る
  （= 新 ActionType は生成されない）。

---

## 9. 設計判断と根拠（なぜこの方式か）

- **合成ルールレジストリ（vs 形式文法/構造化データ主導）**: 実カードテキストは
  半構造的で揺れが大きく、形式文法は脆く段階移行しづらい。ルールレジストリは
  「未対応はレガシーへフォールバック」で非破壊・漸進的に移行でき、各ルールが単体テスト可能。
- **IR/インターフェース維持**: resolver/gamestate を無改修にでき、リスクと差分を最小化。
- **継続効果を専用フィールドで実装（vs 既存 power_buff/flags 流用）**: 既存はターン境界で
  リセットされるため複数ターン跨ぎ効果と衝突する。専用フィールド＋イベント失効で
  reapply 不要のクリーンな設計にした。
- **除去保護をライブ評価（vs フラグのラッチ）**: 条件（トラッシュ枚数等）が変動するため、
  除去の瞬間に評価する方が正確。

---

## 10. 参考

- `docs/parser_v2.md` — 設計詳細・ルール一覧・現況・残課題
- PR #4 — 刷新の土台（合成ルールレジストリ＋V2本番化）
- 本ブランチ `claude/card-effect-resolution-6vag2` — ルール拡充・正確性修正・継続効果統合・置換効果
- 計測の起点コマンド: `OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40`
