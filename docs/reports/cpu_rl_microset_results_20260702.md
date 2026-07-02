# マイクロセット結果: 本走ループの配管検証と「ゲートの検出力不足」の判明

日付: 2026-07-02 / コード: `tests/selfplay_loop.py`（v4c 実装）
設定: gens=3・self-play 15局/世代・PCR fast12/full48・gate 6ペア(n=12/デッキ)・buffer 2世代・seed0

> スナップショット（改変しない）。本走前のハイパラ感度チェック（数字の最終値は見ない設定）。

## 結果
| gate | nami | blackbeard | hancock | avg | minCI下端 |
|---|---|---|---|---|---|
| gen1 | 0.33 | 0.17 | 0.75 | 0.417 | 0.047 |
| gen2 | 0.33 | 0.33 | 0.33 | 0.333 | 0.138 |
| gen3 | 0.33 | 0.67 | 0.08 | 0.361 | 0.015 |

self-play サンプル: gen0=832 → gen1=1935 → gen2=2376（buffer 蓄積・checkpoint 毎世代保存）。

## 判明したこと
1. **配管は堅牢**: gen0 warm-start → PCR自己対戦 → value(z)/policy(訪問)学習 → ペアゲート → checkpoint、
   を**3世代連続で例外なく完走**。resume も別途検証済み。崩壊・発散なし（安定）。
2. **世代トレンドは読めない＝ゲートの検出力不足**: 世代平均 0.417→0.333→0.361 は横ばいだが、
   **6ペア(n=12/デッキ)の Wilson CI は ±0.30**。per-deck の激しい振れ（hancock 0.75→0.08）は
   統計ノイズそのもの。**±0.1 の世代改善はこの n では原理的に見えない**（レビュアーの「N≈100/デッキ」指摘の裏付け）。
3. **絶対値が低いのは sims=48 のため**: bootstrap-only ゲート(sims160)は avg≈0.5-0.6 だった。
   micro は fast12/full48 と浅く、policy 教師（full手の訪問分布）も 48sims/18% と薄い＝低品質。
   マイクロの絶対値は本走(sims160)を代表しない。

## 含意（本走の設定に反映）
- **ゲートは N≈100/デッキ（=約50ペア）必要**。6ペアでは世代比較不能。本走ゲートは pair 数を上げる。
- **self-play も sims/局数を上げないと policy 教師が薄く**、visit-distribution policy が
  （L1模倣policyと同じ失敗機序で）むしろ害になり得る懸念が残る。本走は full=160・局数増で厚くする。
- **未解決の要確認**: 「visit-distribution policy prior は低品質だと害」を、本走スケールで
  uniform vs policy の world 内比較で監視すべき（gen1 以降で policy を入れて悪化しないか）。

## 次段の選択肢
- **フル本走**（sims160・sp-games≥40・gate≥25-50ペア・数世代）: 唯一トレンドが見える。checkpoint 済で
  再起動耐性あり。ただし数時間規模。
- **中規模ミニセット**（sims80・gate15-25ペア）: フルより軽く、トレンドの向きだけ先に確認。
- いずれも「policy 有無の world 内比較」を1点入れて、低品質policyの害を監視する。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/selfplay_loop.py --gens 3 --sp-games 15 --boot-games 120 \
  --gate-pairs 6 --fast-sims 12 --full-sims 48 --seed 0 --ckpt <path>
```
