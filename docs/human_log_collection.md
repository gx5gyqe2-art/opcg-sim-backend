# 人間ログによる評価関数強化 — 収集・学習フロー

対 CPU 戦の実プレイ（人間 vs CPU）から価値関数（`hard`／α-β の評価関数の学習化, SPEC §2.5.8）の教師データを集め、
候補モデルを学習し、Elo で検証してから本番へ昇格するための運用手順。

## 全体像

```
[フロント采取ボタン] → 采取JSON → tests/human_captures/ にコミット
        │ (replay.value_samples = ターン境界の {特徴, 勝敗ラベル})
        ▼
  ① ingest   tests/human_log_ingest.py        → 集約 JSONL（特徴長/ラベルを検証）
  ② train    tests/train_value.py             → 候補モデル（非同梱）
  ③ eval     tests/eval_value_on_set.py       → 候補 vs 同梱の汎化を数値比較
        ▼ (ここまで一括 = tests/human_value_pipeline.py)
  ④ Elo検証  自己対戦アリーナで blend ON/OFF を A/B（合格時のみ昇格）
  ⑤ 昇格     value_model.json 差し替え + OPCG_VALUE_BLEND_HARD を有効化
```

①〜③は**安全**（同梱モデル・本番挙動を一切変えない・読み取りと候補生成のみ）。④⑤は**挙動を変える**ため
明示の判断と品質ゲートを伴う。

## データの性質（なぜ使えるか）

- `value_samples` はバックエンドが**ライブ採取**したもの（`api/app.py: _capture_value_samples`）。
  ターン境界ごとに**両者視点**の特徴を貯め、終局時に `/replay` で**勝者からラベル `y`=0/1 を確定**する。
- 特徴抽出は `cpu_features.extract_features(see_opp_hand=False)`＝**フェア**（相手手札の中身を読まない）。
  難易度が `hard`（透視）でも、学習特徴はフェアなので学習データとして偏らない。
- 特徴長は `cpu_features.N_FEATURES`。**采取時とコードの特徴数が一致**している必要がある
  （ingest は不一致行を自動で捨てる＝古い特徴数の采取は混ざらないが、その分は無駄になる）。

## 手順

### ① 采取を貯める
対 CPU 戦（終局まで）をプレイし、フロントの「采取」ボタンで JSON を保存。
`tests/human_captures/` に置いてコミットする（推奨名 `YYYYMMDD_<game_id先頭8桁>.json`）。
未決着の采取は `value_samples` が空＝寄与しないので置いても無害。

### ②〜③ 一括実行
```bash
OPCG_LOG_SILENT=1 python tests/human_value_pipeline.py
```
- 集約行数 < 50 なら学習せず停止（1 対局 ≒ 20〜30 行＝**目安 3 対局以上**で学習可能）。
- 学習すると候補モデル `tests/human_value_model.candidate.json`（**非同梱・gitignore**）を出力し、
  候補 vs 同梱の `val_acc` / logloss を表示する。

個別ツールでも実行できる:
```bash
python tests/human_log_ingest.py --in tests/human_captures/ --out tests/human_value.jsonl
python tests/train_value.py --data tests/human_value.jsonl --out tests/human_value_model.candidate.json
python tests/eval_value_on_set.py --data tests/human_value.jsonl --model tests/human_value_model.candidate.json
```

### ④ Elo 検証（昇格の前提）
`val_acc` が十分（過去の自己対戦学習は 0.645 で強さ未達だった＝**ここを上回るのが目安**）なら、自己対戦
アリーナで `OPCG_VALUE_BLEND_HARD`/`OPCG_VALUE_BLEND` の α を上げた blend ON と OFF を A/B し、
**勝率非劣化＋「変な手」カウンタ非増加**を確認する（WBS「変な手撲滅」Phase0 監査・SPEC §2.5.8）。

### ⑤ 昇格
合格した候補のみ `opcg_sim/src/core/value_model.json` を差し替え、Dockerfile で
`OPCG_VALUE_BLEND_HARD`（`hard`／α-β の葉ブレンド）を有効化する。
**昇格はこのフロー外の明示操作**＝パイプラインは決して自動で同梱モデルを上書きしない。

## 注意
- まだ**量が足りない**段階では blend は OFF のまま（現状の同梱は starter モデル・OFF）。
- 1 対局では評価関数は強くならない。意味のある再学習には数十〜数百対局規模を要する。
- 中間物（`tests/human_value.jsonl`・候補モデル）は gitignore 済み。**采取 JSON 本体だけ**を蓄積する。
