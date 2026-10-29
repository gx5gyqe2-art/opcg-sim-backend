# WP `offdist-value` の「窓を閉じた後の盤面」の行を作るシャード（1〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.69 の検証 3（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/offdist_value.py`（`gen --shard i/8`）。
エンジン・ネットは変えない（自己対戦を打って行を足すだけ）＝**ゲートは不要**。コードと `RESULT.json` は変えない。

## 中身
- r27 の自己対戦（波 60 と同じ設定・実デッキ・1,280 回・世界 1 本）を **1 シャード 450 局**打ち、相手の手番の窓（自分が応手を決める）で
  各候補を当てて窓を閉じた盤面を、自分の視点・その局の勝敗で「価値だけの行」として書く（1 局あたり約 75 行）。
- seed は `5100000 + i × 10000 + 0〜449`（既存の波と重ならない帯）。
- 所要の目安: 4 本並べで 1 局 約 10 秒 → **1 シャード 20〜30 分**。出力は約 30MB。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）
cd /home/user/opcg-sim-backend
git fetch origin claude/offdist-value
git checkout -B claude/offdist-value-g$I origin/claude/offdist-value
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/offdist_value.py gen --games 450 --workers 4 --shard $I/8 \
  --gen-out n1_results/offdist_value/fold_rows > n1_results/offdist_value/gen_shard$I.log 2>&1 &
```

- 1 行目に `GEN shard=I/8 todo 450` が出れば正しく始まっている。局ごとに `[k/450] seed=... rows=...` が増える。9 分以内の待ちを繰り返して見張る。
  止まっても同じコマンドで再開できる（`progress.jsonl` に書いた局は飛ばす）。最後に `GEN done {...}` が出れば終わり。

## 終わったら

```bash
git add n1_results/offdist_value/fold_rows/fold_s$I n1_results/offdist_value/gen_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "offdist-value: fold rows shard $I/8"
git push -u origin claude/offdist-value-g$I
```

- push 先は **`claude/offdist-value-g$I`**。

# 検証 2（時点ごとのドンの値）のシャード（1〜7）

道具: `tests/scripts/offdist_value.py`（`timeval --shard i/8`）。§7.66 の紫カイドウ 24 点を 8 本に分ける（各 3 点・1 点目は済み）。
1 点＝世界 16 本 × 4 時点の盤面のネットの値と、窓を閉じた直後の対（ドン 1 枚違い）を両方とも 16 通り最後まで打つ。4 本並べで **1 シャード 40〜60 分**。

```bash
I=1   # ← シャード番号（1〜7）
cd /home/user/opcg-sim-backend
git fetch origin claude/offdist-value
git checkout -B claude/offdist-value-t$I origin/claude/offdist-value
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/offdist_value.py timeval --k 16 --k-roll 16 --workers 4 --shard $I/8 \
  > n1_results/offdist_value/timeval_shard$I.log 2>&1 &
# 1 行目 `TIMEVAL shard=I/8 todo 3`（シャード 0 と同じ点があれば 2）を確かめ、9 分以内の待ちを繰り返して見張る。終わったら:
git add n1_results/offdist_value/timeval_shard$I.jsonl n1_results/offdist_value/timeval_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "offdist-value: timeval shard $I/8"
git push -u origin claude/offdist-value-t$I
```

# 検証 3（1 周の試験訓練）——2 本（対照 `ctrl`・試験 `fold`）

どちらも r27 から warm-start・**波 60 の全部の行を 1 周**（今の作り方と同じ設定）。試験は、さらに上で作った「窓を閉じた直後の盤面」の
価値だけの行（`n1_results/offdist_value/fold_rows/fold_s0〜7`・約 47 万行）を `--z-in` で足す。1 本 4〜6 時間（4 本並べ・torch）。メモリは 14GB 内（1 波）。

```bash
RUN=ctrl   # ← ctrl か fold
cd /home/user/opcg-sim-backend
git fetch origin claude/offdist-value
git checkout -B claude/offdist-value-$RUN origin/claude/offdist-value
pip install -q maturin && make rust-develop
pip install -q --target ~/torch_env torch
mkdir -p ~/w60 && for i in $(seq -w 1 32); do git fetch -q --depth 1 origin claude/n60-w$i && git archive FETCH_HEAD n_records | tar -x -C ~/w60; done
ls ~/w60/n_records | wc -l   # 128 になること
Z=""; [ "$RUN" = fold ] && Z="--z-in n1_results/offdist_value/fold_rows/fold_s0 n1_results/offdist_value/fold_rows/fold_s1 n1_results/offdist_value/fold_rows/fold_s2 n1_results/offdist_value/fold_rows/fold_s3 n1_results/offdist_value/fold_rows/fold_s4 n1_results/offdist_value/fold_rows/fold_s5 n1_results/offdist_value/fold_rows/fold_s6 n1_results/offdist_value/fold_rows/fold_s7"
PYTHONPATH=~/torch_env:$PWD OPCG_LOG_SILENT=1 nohup python -m opcg_sim.learned.train.n_rel_train train \
  --in ~/w60/n_records/n60_w*_p[0-3] $Z --ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 1 --pi-teacher q_improved \
  --pi-steps-mult 1.0 --fast --warm-start opcg_sim/data/learned/nrel_r27.npz \
  --out n1_results/offdist_value/nrel_$RUN.npz > n1_results/offdist_value/train_$RUN.log 2>&1 &
# `ep0 step k/23000` の行が増える（試験はもう少し多い）。9 分以内の待ちを繰り返して見張る。終わったら:
git add n1_results/offdist_value/nrel_$RUN.npz n1_results/offdist_value/train_$RUN.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "offdist-value: trial net $RUN (1 epoch)"
git push -u origin claude/offdist-value-$RUN
```

- push 先は **`claude/offdist-value-ctrl`／`claude/offdist-value-fold`**。ネットは試験用（出荷・同梱しない）。途中で落ちたら最初からやり直す（1 周しかない）。
