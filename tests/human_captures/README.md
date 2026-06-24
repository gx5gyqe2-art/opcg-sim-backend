# human_captures — 人間ログ（フロント采取）の蓄積場所

対 CPU 戦（`cpu_trace=true` の対局）で、フロントの「采取」ボタンが出力した JSON をここに置く。
終局した対局には `replay.value_samples`（`{"f":[...],"y":0/1}`・ターン境界の両者視点＋勝敗ラベル）が
含まれ、これが価値関数（評価関数の学習化）の教師データになる。**特徴はフェア**（相手手札の中身は読まない）。

> コンテナは揮発するため、貯めた采取はこのリポジトリにコミットして初めて永続する。

## 置き方
- ファイル名は任意（推奨 `YYYYMMDD_<game_id先頭8桁>.json`）。
- 「采取」ボタンの JSON はそのまま置いてよい（取り込みは `value_samples` 以外を無視する）。
  大きさが気になる場合は `replay`（`value_samples` を含む）と最小メタだけ残してもよい。
- **未決着の采取は `value_samples` が空**＝学習には寄与しない（置いても無害）。

## 使い方
リポジトリルートで:

```bash
OPCG_LOG_SILENT=1 python tests/human_value_pipeline.py
```

ingest → train（候補モデル）→ eval を一括実行する。学習には集約 **50 行以上**が必要
（1 対局 ≒ 20〜30 行）。詳細・昇格手順は `docs/human_log_collection.md`。
