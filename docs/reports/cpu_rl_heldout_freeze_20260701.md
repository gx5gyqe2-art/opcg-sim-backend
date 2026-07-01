# held-out 実デッキ集合 freeze（v4b 実装ステップ1）

日付: 2026-07-01 / 計画: `cpu_rl_frozen_design_v4b_20260701.md` 実行順1
資産: `tests/heldout_decks.json`（凍結データ・sha256 を `tests/test_heldout_decks.py` で CI 監視）
ローダ: `tests/heldout_decks.py`

> スナップショット（改変しない）。リスト変更は新しい日付で freeze し直す。

## 内容
ユーザの実対局リプレイ2件（270d5beb / 5f3528c2・いずれも difficulty=learned）から抽出した
**実構築3種**（全て 50枚・4枚制限・リーダー色一致・DB実在を検証済み）:

| id | リーダー | 色 | 出典 |
|---|---|---|---|
| nami_blue_yellow | OP11-041 ナミ | 青黄 | ユーザ使用デッキ（game 270d5beb p1） |
| blackbeard_black_yellow | OP16-080 マーシャル・D・ティーチ | 黒黄 | 学習CPU崩壊の当該デッキ（両game p2・同一リスト） |
| hancock_blue_yellow | OP14-041 ボア・ハンコック | 青黄 | ユーザ使用デッキ（game 5f3528c2 p1） |

## 使用規則（凍結）
- **訓練（自己対戦・デッキ生成）への使用禁止**（リーク）。
- 許可用途は (a) **vs L1 勝率ゲート**（>0.60・SPRT） (b) **Covering Radius の一方向確認** のみ。
- `test_heldout_decks.py` が sha256・合法性・エンジン投入可を CI で常時検証（うっかり改変で CI が落ちる）。

## 特記
- 3構築とも黄を含む（ユーザの実環境そのもの）。ゲートとしては「実際に遊ばれる環境で強いか」を
  直接測る最良の集合。他色の実リストが入手できたら新しい日付で追加 freeze する
  （表現ロバスト性の広域確認は別途「シナジーパッケージ丸抜きカナリア」が担当）。
- 特に blackbeard_black_yellow は**元の障害（value 盲目化）を起こした当該リスト**＝回帰の意味も持つ。
