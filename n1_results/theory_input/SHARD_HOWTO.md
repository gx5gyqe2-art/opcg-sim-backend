# theory-input の sidecar のシャードの回し方（24 本・1 本 ≒ 40〜60 分）

WP `theory-input`（`docs/search_efficiency_plan.md` §7.82）の段 1。見える情報だけの理論の列（25 列）を、波 70f〜73f（訓練）と
波 74c（取り置きの誤差の試験台）の**全行**で出し、記録のシャードの隣に `n_record_XXXXX.theory.npz` を書く。
シャード番号 `K`（1〜24）、2 桁の零詰めを `KK`（例 `07`）とする。**コードは変えない**。道具は `tests/scripts/theory_input.py sidecar`。
局は打ち直さない（記録の行＝その席に見える盤面そのもの）。1 行 約 17 ms（1 コア・Rust）・5 波 1,340 万行。

## 1. 用意（約 15 分）

```bash
cd /home/user/opcg-sim-backend            # リポジトリの場所（違えば読み替える）
git fetch origin claude/theory-input
git checkout -B claude/theory-input-sKK FETCH_HEAD
pip install -q maturin && pip install -q -r opcg_sim/requirements.txt
make rust-develop                         # エンジンの wheel（数分・この WP の Rust を含む）
OPCG_LOG_SILENT=1 python -m opcg_sim.tools.export_effects_json > /dev/null   # 理論が読む効果の木（1 回）
python -c "import opcg_engine; print(hasattr(opcg_engine, 'theory_game_call'))"   # True であること
```

5 波の記録（160 ブランチ・合計約 2.5GB）を取る（約 6 分）:

```bash
for wv in 70f 71f 72f 73f 74c; do
  mkdir -p ~/n$wv
  for w in $(seq -w 1 32); do
    [ -d ~/n$wv/n_records/n${wv}_w${w}_p0 ] && continue
    for t in 1 2 3 4; do git fetch -q --depth 1 origin claude/n$wv-w$w && break; sleep $((2**t)); done
    git archive FETCH_HEAD n_records | tar -x -C ~/n$wv
  done
done
ls -d ~/n7*/n_records/*_p[0-3] | wc -l    # 640 であること
```

## 2. 回す（約 40〜60 分・4 並列）

```bash
nohup env OPCG_LOG_SILENT=1 python tests/scripts/theory_input.py sidecar \
  --in '~/n70f/n_records/n70f_w*_p[0-3]' '~/n71f/n_records/n71f_w*_p[0-3]' '~/n72f/n_records/n72f_w*_p[0-3]' \
       '~/n73f/n_records/n73f_w*_p[0-3]' '~/n74c/n_records/n74c_w*_p[0-3]' \
  --shard K/24 --workers 4 > ~/ti_sKK.log 2>&1 &
```

- 背景で回し、**9 分以内の待ちを繰り返して** `~/ti_sKK.log` を見張る（`SIDECAR_DONE` が出たら終わり）。ターンを閉じない。
- 進み具合はシャードのファイルごとに `[n/320] … rows … 残り見込み … 分` と出る（1 本 320 ファイル）。
- 落ちたら同じコマンドを打ち直せば、書けたファイルを飛ばして続きから回る。

## 3. 確かめる

```bash
ls ~/n7*/n_records/*/n_record_*.theory.npz | wc -l     # このシャードの分（320）が揃っていること（全体では 7680）
python - <<'PY'
import glob, numpy as np, os
fs = sorted(glob.glob(os.path.expanduser("~/n7*/n_records/*/n_record_*.theory.npz")))
bad = 0
for f in fs:
    with np.load(f) as d, np.load(f.replace(".theory.npz", ".npz")) as s:
        bad += not (d["th"].shape == (len(s["seed"]), 25) and np.array_equal(d["seed"], s["seed"]))
print("files", len(fs), "bad", bad)
PY
```

## 4. 残す（push）

sidecar は記録の隣（`~/n7*/`・リポジトリの外）に在るので、**結果ブランチには sidecar を tar にして置く**（1 本 約 150MB）:

```bash
mkdir -p n1_results/theory_input/shards
tar -czf n1_results/theory_input/shards/theory_sKK.tar.gz -C ~ $(cd ~ && ls -d n7*/n_records/*/n_record_*.theory.npz)
ls -la n1_results/theory_input/shards/theory_sKK.tar.gz
cp ~/ti_sKK.log n1_results/theory_input/shards/
```

`RESULT.json`（リポジトリ直下）を次の形で書き、結果だけをコミットして push する:

```json
{"wp": "theory-input", "shard": "K/24", "files": <sidecar の数>, "rows": <SIDECAR_DONE の行数>, "sec": <SIDECAR_DONE の秒>,
 "tar": "n1_results/theory_input/shards/theory_sKK.tar.gz"}
```

```bash
git add RESULT.json n1_results/theory_input/shards/theory_sKK.tar.gz n1_results/theory_input/shards/ti_sKK.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "theory-input シャード KK/24 の理論の列"
git push -u origin claude/theory-input-sKK            # 失敗したら 2・4・8・16 秒あけて最大 4 回
```

結果ブランチは **`claude/theory-input-sKK`**（零詰め 2 桁）。PR は作らない・マージしない。コミットにモデル名を書かない。
