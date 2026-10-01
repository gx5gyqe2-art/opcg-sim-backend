# F-1: 攻撃時・常時・自分のターン中の能力（値付けゼロの群）の棚卸し(診断のみ・2026-09-25)

`docs/game_theory.md`398行目付近(T74・2026-09-17)の「アタック時98・常時53」という古い数字を、
今のカードDB(2,803種・全能力3,386本)で数え直した。ON_ATTACK・PASSIVE・YOUR_TURNの3契機で
計675本——DBが拡張されて元の数字の2.6〜6.5倍になっている。**新定数ゼロ・読み取りのみ・
コード変更なし**。

コーディネータ側で独立に再検証済み: (1) `theory_order.py`の`ATTACK`/`DON_BOX`分岐(1475行目
付近)が攻め手カード自身の`abilities`を一度も読まないことを直接確認、(2) 契機別本数
(256/346/73)を独立に数え直して完全一致、(3) `EV.ON_PLAY_TRIGGERS`に`'PASSIVE'`が含まれる点が
本報告の主張と矛盾しないか調査——`EV.CHAR_ON_PLAY_TRIGGERS = ('ON_PLAY',)`(キャラ自身の
登場時解決で実際に使われる契機集合)は`'PASSIVE'`を含まないため、矛盾しないことを確認。

## 器と測り方

```bash
# 母数と契機別内訳・内側動作型の分類
OPCG_LOG_SILENT=1 python <census script>  # effect_value._all_cards() を契機で全数走査

# 実記録・合成記録での発火頻度（規模感・フルmeasurementはF-4）
OPCG_LOG_SILENT=1 python <freq_scan script> --in <w41> <w39> <w42>
```

使い捨ての計器は本タスクの規約によりスクラッチに置いた(リポジトリには置いていない)。
実記録: w41(300局・`--decks user`)。合成記録: w39+w42(合わせて900局・`--decks synth`)。

## 発見1: `theory_order.score_candidate`のATTACK/DON_BOX分岐はON_ATTACK能力を一度も読まない

`tests/scripts/theory_order.py:1475-1499`は`attack_value(sp, tp, ...) - dcost`だけを返す。`sp`/`tp`は
攻め手・対象の**今のパワー**(`slot_power`が枠の値を渡す)で、攻め手カード`cid`の`abilities`は
参照されない。ON_ATTACKは`ON_PLAY_TRIGGERS`(`ON_PLAY, ACTIVATE_MAIN, MAIN, RULE, PASSIVE, None`)にも
`ACTIVATE_TRIGGERS`(`ACTIVATE_MAIN`)にも入っていない——**呼び出しの配線が構造的に存在しない**。

一方`effect_value.py`は、ON_ATTACK能力の中身の動作型をすでに値付けできる: 256本の内側動作の上位は
BUFF 103・KO 33・DRAW 22・DECK_BOTTOM 20・DISCARD 16・MOVE_CARD 15・LOOK 15・REST 14——これらは
`PRICED`/`TEMPO`/`MOVE_KINDS`/`OBSERVE_KINDS`が既にON_PLAY契機の能力に対して使っている式そのもの。
**F-2は新しい値付け式ではなく、既存の`ability_value`/`card_value`をON_ATTACK契機で呼ぶ配線が主な作業**
になる(例外は条件判定——ACTIVATE_MAINは`offered=True`で無条件に通すが、ON_ATTACKの内部条件
「相手のライフが2枚以下の場合」等はエンジンが事前検査していないので`condition_value`で状態から
判定する必要がある)。

## 発見2: Rustの常在効果再計算がすでに「今のパワー」「ブロッカー」に焼き込んでいる範囲

`rust/opcg_engine/src/effects/passives.rs::apply_passive_effects`はアクション境界のたびにYOUR_TURN→
OPPONENT_TURN→PASSIVEの順で能力を再実行する。ただし`is_reactive_passive`の正規表現
(`(された|した|受けた|なった|離れた)時、`)に一致する能力は**明示的にスキップ**される
(コード中のコメント: 「継続効果ではない」)。再実行された`BUFF`/`GRANT_KEYWORD`はトークン列
0(`power_now`)・6(`is_blocker_active`)に書き込まれ、`theory_order.slot_power`がそのまま
`attack_value_don`/`nu_of`の入力に渡る。

**この機構が「反映済み」と言える範囲は狭い**——v13トークン(22枠×20列)にキーワードのブール列は
`is_blocker_active`1つしかない。速攻・ダブルアタック・バニッシュ・ATTACK_ACTIVE・ブロック不可には
対応する列が無い(v12の`encode/scalars.rs::FIELD_KEYWORDS`には4ブールがあるが別配列で、
`theory_order.py`はそれを読んでいない)。したがって:

- **すでに反映済み(穴ではない)**: 自分・他者の体のパワーを継続的に変えるだけのBUFF、
  ブロッカー/ブロック不可以外を伴わないGRANT_KEYWORD、名前読み替え(RULE_PROCESSING)、
  恒等式で決まるVICTORY。
- **穴(継続実行されているが理論からは見えない)**: KO代替(REPLACE_EFFECT)・退場防止(PREVENT_LEAVE)・
  アタック禁止(ATTACK_DISABLE)・手札を捨てる/トラッシュへ送る(DISCARD/TRASH)・レスト強制(REST)・
  速攻/ダブルアタック/バニッシュ/ATTACK_ACTIVE等の非ブロッカー系キーワード付与。

## 発見3: 契機別・最終バケット(全675本)

| 契機 | 本数 | 穴(本数・割合) | 主因 |
|---|---|---|---|
| ON_ATTACK | 256 | 256(100%) | 配線が存在しない |
| PASSIVE | 346 | 196(56.6%) | サバイブ系118本(KO代替+退場防止)が中心・非ブロッカー系キーワード付与50本を含む |
| YOUR_TURN | 73 | 31(42.5%) | 26本が「〜した時」型のイベント誘発の誤ラベル |
| **計** | **675** | **483(71.6%)** | — |

残る192本(28.4%)は上記の「すでに反映済み／無価値」で、既存の値付けで足りている。

`gap_reactive_mislabeled`(PASSIVE 4本・YOUR_TURN 26本・計30本)はRustエンジン自身が
「継続効果ではない」と判定してスキップしている能力——ON_LEAVE/ON_KOのようなイベント別の
値付けフックが理論側に無いため、**継続効果の穴(197本)とは別の機構**として扱うべき。

## 発見4: 実記録・合成記録での発火頻度(規模感)

| | 実(w41・300局) | 合成(w39+w42・900局) |
|---|---|---|
| 攻撃系候補の総数 | 108,431 | 237,968 |
| うちON_ATTACK能力を持つ攻め手カードの候補 | 11,359(10.5%) | 31,550(13.3%) |
| 攻撃を含む判断点の総数 | 15,271 | 38,605 |
| うち1本以上ON_ATTACK能力の攻め手が絡む判断点 | 3,177(20.8%) | 8,641(22.4%) |
| distinct playable cid | 57 | 967 |
| うちON_ATTACK穴カード | 5 | 90 |
| うちPASSIVE穴カード | 3 | 71 |
| うちYOUR_TURN穴カード | 0 | 7 |

実記録は固定の少数デッキ(57種)なので穴カードの母数自体は小さいが、攻撃の判断点の20.8%には
既に絡んでいる。合成記録(967種＝カード母集団に近い)では規模がはっきり大きい。**F-4は実・合成
どちらも回す**——実だけで判定すると穴の規模を過小評価する。

## 発見5: 攻撃行の残差37.8%/38.9%との関係

`docs/reports/2026-09-25_p8_7c_attack_resid_split.md`は「相手の窓の応答」による説明を否定し、
残差の主因を「価格そのものの誤り」と見ている(確定はしていない)。ON_ATTACK能力の未配線は
「価格そのものの誤り」に分類される機構的に独立な原因の候補だが、**攻撃候補の10.5%/13.3%しか
関与しない**ため、残差全体の説明にはならない部分的な候補である。F-4では「攻め手カードが
ON_ATTACK能力を持つ行」と「持たない行」で残差を層別してから寄与を測るべき。

## 推薦(F-2/F-3のスコープの絞り方)

1. **F-2(ON_ATTACK)**: `score_candidate`のATTACK/DON_BOX分岐に、攻め手カードのON_ATTACK能力の値を
   新規に足す切替を1本作る(既定off・新定数ゼロ)。内部条件は`condition_value`で状態判定(`offered`は
   使わない)。既存の`ability_value`/`_referenced_ability`の再帰をそのまま流用できる——256本の内側動作型は
   ON_PLAY契機で既に値付け済みの型と重複している。
2. **F-3(PASSIVE/YOUR_TURN)は2つに分けて別々に判断する**:
   - (a) 継続実行系の穴(PASSIVE 192本＋YOUR_TURN 5本＝197本): `nu`ではなく`_char_play_value`とは
     別枠の「体の継続価値」の加算項として、既存の`SURVIVE_KINDS`(サバイブ118本)・`TEMPO`
     (ATTACK_DISABLE/REST等)・`keyword_delta`/`granted_attack_value`(速攻等の非ブロッカー系
     キーワード・すでにT54/T55/T56で作られている式を再利用)で埋める。
   - (b) イベント誘発の誤ラベル(PASSIVE 4本＋YOUR_TURN 26本＝30本)は**(a)とは別の切替・別のタスク**
     にする——継続再計算ではなく「ドンが戻った時」「相手がイベントを撃った時」等の**イベント別**の
     値付けが要り、(a)と混ぜるとF-4の測定でどちらの機構が効いたか読めなくなる。母数が小さい
     (30/675=4.4%)ので、F-3の主眼は(a)に置き、(b)は次点(G系またはF-6)に回すことを推薦する。
3. **F-4の層別**: 攻め手カードがON_ATTACK能力を持つ行／持たない行で残差を分けて測ること。

## 再現

```bash
cd /home/user/opcg-sim-backend
OPCG_LOG_SILENT=1 python tests/scripts/effect_value.py --census --out /tmp/ev_census.json
# 契機別・内側動作型の分類はスクラッチ計器（本作業用の一時スクリプト）で行った
# 常設化するかはF-2/F-3の実装タスク側の判断とする
```
