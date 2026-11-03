# WP `value-split` 計測 A（教材の打ち方での打ち続け）のシャード（01〜16）の回し方

仕様: `docs/search_efficiency_plan.md` §7.74 の計測 A（本線 `claude/value-split-a`）。道具: `tests/scripts/value_split.py`（`play --shard k/16`）。
エンジン・ネットは変えない（対を作って打ち続けるだけ）＝**シャードにゲートは不要**。コードと `RESULT.json` は変えない。

## 中身
- §7.66 の紫カイドウ 24 点の「窓を閉じた直後」のドン 1 枚違いの対を、配り直した世界 **32 本**（配り方の番号のずらし 17000）で作り、
  **同じ対**を 4 通りで最後まで打つ: `ref`＝従来の物差し（今のターンの残りは両席出荷・その先は審判 r27 160 回）／
  `teach_r29f4`＝打ち手 r28x4・生成役の設定・後半の散らし 4 割（r29f4 の教材の打ち方）／`teach_r28x4`＝r27・2 割／`teach_r27`＝r26・散らしなし。
  (ii) の 3 通りは両席とも生成役の設定（1,280 回・世界 1 本・枝予算 0・q_min_n 1/8・根の事前分布の温度 2.0・根のノイズ 0.25・ε 3% の差し替え）。
- 1 シャード＝(点, 世界) の組 48 個（各点 2 世界）× 4 通り × 2 局＝**384 局**。seed は `move_judge.world_seed(局 seed, step, 17000 + i)`（他の計測と重ならない）。
- 所要の目安: 1 局 10〜60 秒（終盤の点は短く序盤の点は長い。煙試験＝ターン 12 の点で 1 局 約 11 秒・対を作るのに 5〜20 秒）。
  4 本並べで **1 シャード 1〜2.5 時間**。出力は 1 シャード 約 200 行の jsonl（数百 KB）。メモリは 4 本で数 GB。

## 手順（シャード k ＝ 1〜16 のどれか。下の `K` を置き換える）

```bash
K=1   # ← シャード番号（1〜16・零詰めなし）
KK=$(printf %02d $K)
cd /home/user/opcg-sim-backend
git fetch origin claude/value-split-a
git checkout -B claude/value-split-a-s$KK origin/claude/value-split-a
pip install -q maturin
make rust-develop
mkdir -p n1_results/value_split
OPCG_LOG_SILENT=1 nohup python tests/scripts/value_split.py play --k 32 --workers 4 --shard $K/16 \
  > n1_results/value_split/play_s$KK.log 2>&1 &
```

- 1 行目に `PLAY bench=kaido shard=K/16 modes=ref,teach_r29f4,teach_r28x4,teach_r27 k=32 jobs 48 (of 48)` が出れば正しく始まっている。
  (点, 世界) の組ごとに `[n/48] pt=… i=… ref=a/b(秒) teach_r29f4=a/b(秒) …` が増える（a／b＝そのまま／1 枚少ない の勝ち 1・負け 0）。
  9 分以内の待ちを繰り返して見張る。止まっても同じコマンドで再開できる（`rows_sKK.jsonl` に書いた (点, 世界, 通り) は飛ばす）。
  最後に `PLAY_DONE rows 192` が出れば終わり（`x` が混じる行＝決着せず／失敗。数行なら放置してよい）。
- `make rust-develop` が無いと `import opcg_engine` で落ちる。`maturin` が無ければ `pip install maturin`。

## 終わったら

```bash
git add n1_results/value_split/rows_s$KK.jsonl n1_results/value_split/play_s$KK.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "value-split: playout shard $K/16"
git push -u origin claude/value-split-a-s$KK
```

- push 先は **`claude/value-split-a-s01`〜`s16`**（零詰め 2 桁）。PR は作らない。

# 段 2（6キングだけ 53 点・段 1 の集計の後に起こす）

道具は同じ `value_split.py`（`play --bench rocks`）。`n1_results/rocks_bounce/points.jsonl` の 6キングだけ 53 点（a＝CPU の手・b＝戻す・対象は 6キングに固定）を
配り直した世界 16 本（ずらし 17000）で、既定は `teach_r29f4`（r28x4・散らし 4 割）だけ打つ（`--mode ref,teach_r29f4` で物差しも同じ世界で対にできる）。
1 シャード＝53 × 16 ÷ 8＝106 組 × 2 局＝212 局（`--mode` 2 通りなら 424 局）。

```bash
K=1   # ← 1〜8
KK=$(printf %02d $K)
cd /home/user/opcg-sim-backend
git fetch origin claude/value-split-a
git checkout -B claude/value-split-a-r$KK origin/claude/value-split-a
pip install -q maturin && make rust-develop
mkdir -p n1_results/value_split
OPCG_LOG_SILENT=1 nohup python tests/scripts/value_split.py play --bench rocks --k 16 --workers 4 --mode teach_r29f4 --shard $K/8 \
  > n1_results/value_split/rocks_play_s$KK.log 2>&1 &
# 1 行目 `PLAY bench=rocks shard=K/8 … jobs 106`。終わったら:
git add n1_results/value_split/rocks_s$KK.jsonl n1_results/value_split/rocks_play_s$KK.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "value-split: rocks playout shard $K/8"
git push -u origin claude/value-split-a-r$KK
```
