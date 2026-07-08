# 効果セマンティクス特徴の棚卸し（v3設計の入力）— 2026-07-08

効果セマンティクスv3（スロット別条件付け＋AST効果特徴）の設計に先立ち、**カードDB全2652枚の効果ASTを走査して
パラメータ空間を実測**した結果。設計書はこの表を正本として特徴次元を決める。
背景: `docs/orchestrator_handoff.md` §2（アーキ天井の原因分析）・LC-ValueNet（`lc_value_net_plan_20260708.md`）の一般化。

## 1. 全体像

- 総カード **2652**・効果持ち **2327 (88%)**・能力数分布 {0:325, 1:1500, 2:813, 3:14} ＝ **能力2つまでで99.5%**。
- enum空間: TriggerType **24**種・ActionType **64**種・ConditionType **42**種・Zone 9・Player 4・CompareOperator 7・
  duration 5値（INSTANT 2554 / THIS_TURN 338 / THIS_BATTLE 97 / UNTIL_NEXT_TURN_END 66 / PERMANENT 13）。

## 2. 実測頻度（設計の一次資料）

### TriggerType（能力単位・全カード）
| trigger | n | | trigger | n |
|---|---|---|---|---|
| ON_PLAY | 854 | | TURN_END | 51 |
| ACTIVATE_MAIN | 643 | | ON_BLOCK | 14 |
| TRIGGER | 488 | | ON_REST | 10 |
| PASSIVE | 314 | | ON_DAMAGE_DEALT_TO_LIFE | 6 |
| ON_ATTACK | 246 | | ON_EVENT_PLAY / ON_LEAVE | 2 / 2 |
| COUNTER | 182 | | TURN_START / ON_LIFE_DECREASE / ON_OPP_PLAY / GAME_START | 各1 |
| ON_KO | 174 | | | |
| YOUR_TURN / OPPONENT_TURN | 70 / 55 | | | |
| ON_OPP_ATTACK | 53 | | | |

→ **上位12種で99%**。one-hotは24全部でも安いが、頻度1のものはOTHERに畳んでよい。

### ActionType（効果側・上位）
BUFF 538 / PLAY_CARD 290 / KO 211 / DRAW 166 / GRANT_KEYWORD 134 / REST 119 / RAMP_DON 113 /
EXECUTE_MAIN_EFFECT 78 / ATTACH_DON 67 / REPLACE_EFFECT 65 / ACTIVE 62 / BOUNCE 58 / DECK_BOTTOM 56 /
MOVE_CARD 55 / PREVENT_LEAVE 44 / ACTIVE_DON 37 / DISCARD 36 / ATTACK_DISABLE 31 / HEAL 27 /
TRASH_FROM_DECK 26 / FREEZE 20 / TRASH 19 / RULE_PROCESSING 18 / RETURN_DON 13 / …VICTORY 3。
→ **上位24種＋OTHERで実用十分**。VICTORY（勝利条件変更＝OP03ナミ等）は頻度3でも**必ず独立枠**（符号反転級の意味）。

### ActionType（コスト側）
RETURN_DON 160 / DISCARD 158 / REST 110 / REST_DON 77 / TRASH 63 / MOVE_CARD 46 / DECK_BOTTOM 43 /
FACE_UP_LIFE 20 / BOUNCE 19 / REVEAL 19 / 他少数。→ **6クラス（ドン返却/手札捨て/レスト系/トラッシュ系/ライフ公開/他）**に圧縮可。

### ConditionType（上位）
TURN_LIMIT 300 / LEADER_TRAIT 285 / AND 246 / HAS_DON 219 / FIELD_COUNT 112 / LIFE_COUNT 93 /
LEADER_NAME 92 / DON_COUNT 79 / HAND_COUNT 61 / CONTEXT 43 / DON_COUNT_COMPARE 35 / SOURCE_STATE 29 /
TRASH_COUNT 29 / HAS_CHARACTER 27 / LEADER_COLOR 23 / EVENT_THIS_TURN 23 / OPPONENT_REMOVAL 17 /
LIFE_COUNT_COMPARE 11 / FIELD_ALL_TRAIT 10 / …DECK_COUNT 5 / CHAR_KOED_THIS_TURN 1。

### 数値分布（バケット設計用）
- DRAW: 1〜4（1が中央値）→ {1, 2+} の2バケットで足りる。
- RAMP_DON: 1〜2 → そのまま数値でよい。
- BUFF: 中央値1000・**±1000〜±10000**が本体。⚠️ 異常値あり: **792000**（動的乗算の展開ミス疑い）と **-7〜-4**
  （コスト減がBUFFに混入疑い）。→ 特徴化前に**パーサ側の値正規化を1件ずつ確認**（未決事項①）。
- ATTACH_DON: 1〜3 と **99**（=「全部」のセンチネル疑い）→ 同上（未決事項①）。

### リーダー（137枚）のtrigger分布
ACTIVATE_MAIN 67 / PASSIVE 32 / ON_ATTACK 28 / TURN_END 10 / YOUR_TURN 10 / ON_OPP_ATTACK 10 /
OPPONENT_TURN 10 / ON_KO 6 / ON_DAMAGE_DEALT_TO_LIFE 2。
→ リーダーは**起動メイン・常駐・アタック時**に集中＝リーダー用フル特徴はこの3系を厚く。

## 3. エンコーダ被覆分析（効果が参照する状態変数 vs 現行v2 scalars）

| 効果側が参照する変数 | 参照頻度 | 現行エンコーダ | 判定 |
|---|---|---|---|
| ライフ数・手札数・場数・ドン数 | 93/61/112/79 | scalars にあり | ✅ |
| 付与ドン（HAS_DON） | 219 | v2で追加済（リーダー）＋場キャラ特徴 | ✅ |
| ターン/手番（CONTEXT） | 43 | turn_count / is_my_turn | ✅ |
| キャラのレスト状態 | 29+ | 場キャラ特徴 is_rest | ✅ |
| **トラッシュ枚数（TRASH_COUNT）** | **29** | **無し** | ❌ v3で追加 |
| **山札残数（DECK_COUNT）** | 5 | **無し**（OP03ナミの勝利条件変数！） | ❌ v3で追加 |
| **ターン1回の使用済みフラグ（TURN_LIMIT）** | **300** | **無し**（動的状態） | ❌ 要検討（未決事項②） |
| **今ターンの履歴**（EVENT_THIS_TURN 23 / OPPONENT_REMOVAL 17 / CHAR_KOED 1） | ~41 | **無し** | ❌ 要検討（未決事項②） |
| リーダーの特徴/名前/色ロック | 285/92/23 | リーダーID条件付け（LC）で実質吸収 | △（構築制約＝静的） |

→ **v3 scalars追加の確定枠: 山札残数（自/相手）/50・トラッシュ枚数（自/相手）/20** の4個。
→ TURN_LIMIT（参照300回＝最頻の条件！）と今ターン履歴は**盤面スナップショットに無い動的状態**。
  GameManager から取れるか・エンコーダの「決定的（盤面のみ参照）」原則とどう両立するかが**設計の主要論点**。

## 4. 推奨パラメタ化（設計書へのたたき台）

**決定的効果特徴テーブル `EffFeat[vocab+1, F]`**（DBのASTから起動時に決定的計算・学習しない・新カード自動対応）:
| ブロック | 次元(目安) | 内容 |
|---|---|---|
| trigger | 13 | 頻度上位12 one-hot + OTHER（能力単位→カードはOR集約） |
| action(効果) | 26 | 上位24 + OTHER + **VICTORY独立枠** |
| action(コスト) | 6 | 6クラス圧縮 |
| 数値バケット | ~10 | BUFF{±1k,±2k,±3k+}/DRAW{1,2+}/RAMP/ATTACH |
| condition | ~12 | HAS_DON/TURN_LIMIT/LIFE≤/HAND≤/FIELD数/TRAIT-lock/履歴系/比較系/他 |
| duration | 4 | THIS_TURN/THIS_BATTLE/UNTIL_NEXT/PERMANENT（INSTANTは省略可） |
| target | ~8 | player(自/相手) × zone(場/手札/ライフ/ドン/山/トラッシュ) 主要組合せ |
| **計** | **~80** | 能力2つはOR/加算で1ベクトルに集約（99.5%が2能力以下） |

**ネットへの統合（コスト順の3案）**:
- **案a（最小・LC置換）**: リーダー2枠だけ EffFeat 直結（+160次元）。LCの埋め込みを意味特徴に置き換え＝新リーダーゼロショット化。
- **案b（本命・numpy可）**: 全スロット向けに共有射影 `W_eff: 80→16` を学習し、場キャラ10枠を `[数値8 | eff16]`、
  リーダー2枠はフル80。入力 ~16+4+240+160+pooled24 ≈ **~450次元**・hidden 256 で ~12万param＝numpyで回る規模。
- **案c（将来）**: torch移行＋スロットattention。案bが頭打ちになったら。

## 5. 未決事項（設計前の解決結果・同日調査で確定）

1. **パーサ値の異常系 → 解決（バグではない・機械分離可能）**:
   - 小さいBUFF値（±1〜7）＝**コスト増減**。判別子は `status`＝`COST_REDUCTION`（99件）ほか。status=None の小値9件は
     「コスト+N」（増加方向）＝**(status, |値|<100) の組で機械分離できる**。BUFF status の全分布:
     None/power 397・COST_REDUCTION 99・BLOCKER_DISABLE 17・POWER_OVERRIDE 19・COUNTER 2・COST_OVERRIDE 1。
   - **BUFF=792000 は実在カード**（OP02-082 バーンディ・ワールドのネタ数値）＝パース忠実。バケット「3000+」が吸収。
   - **ATTACH_DON=99 は「すべてに1枚ずつ」の全体対象センチネル**（OP04-004 原文確認）＝ {1,2,3,ALL} の4バケットで扱う。
   → 特徴化ルール: BUFF は (status×値スケール) で **パワーバフ／コスト操作／ブロッカー無効／パワー上書き** に分けて枠を持つ。
2. **動的状態の追加 → 両方とも取得可能・journal安全（確認済み）**:
   - TURN_LIMIT使用済み: `CardInstance.ability_used_this_turn`（**JournaledDict**・`models.py:113`）＝スロット別
     「ターン1能力を使用済み」フラグとして符号化可能。ターン境界と場離れでのみクリア（`resolver.py:55-66` のコメント準拠）。
   - 今ターン履歴: `GameManager._turn_events`（**JournaledDict**・`gamestate.py:157`・ターン開始で再生成 `turn_flow.py:144`）
     ＝ CHAR_KOED_(SELF/OPP) やイベント発動回数を scalars として符号化可能。
   - どちらも盤面/履歴状態でRNG不使用＝エンコーダの決定的原則と両立。JournaledDict なので make/unmake でも整合。
3. **EXECUTE_MAIN_EFFECT（78件）/ REPLACE_EFFECT（65件）の特徴化方針** → 設計書で決める（残る唯一の未決）。
   推奨: 初版は間接フラグ（「参照型効果を持つ」1bit）で妥協し、効果内容の展開は次版。
4. **GRANT_KEYWORD の対象 → ほぼ既存4種で被覆**: 速攻53・ブロッカー39・ダブルアタック13・バニッシュ9（以上は
   エンコーダのキーワード4種と一致）＋ **ATTACK_ACTIVE 13・ブロック不可 5 の2種を追加すれば完全**。

## 6. 再現方法

tests ブート（`PYTHONPATH=tests`＋`_bootstrap`）で `_load_db()` → 全カードの `abilities` を走査し、
`trigger`/`condition`（args再帰）/`cost`/`effect`（sub_effect再帰）の enum を Counter 集計。数値は `value.base`。
