# WP `small-play` の判定シャード（1〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.60。道具: `tests/scripts/small_play.py`（`judge --shard i/8`）。
局は §7.59 の 150 局（`n1_results/hold_bias/games.jsonl`・seed 4171000〜4171149）を使い、uuid → カードの表
（`n1_results/small_play/cards.jsonl`）は作ってある＝**シャードは判定だけ**。

## 分け方
- 判定する点の全体の並び（`small_play.job_order`・局の記録だけで決まる pure な並び・パワー 4000 以下のキャラを出したメインの点 733 点から 400 点）の**番号 mod 8 == i** の点がシャード i（各 50 点）。
- シャード 0 はこのセッション（`claude/small-play`）が回している。
- 所要の目安: 4 本並べで 1 点あたり約 80 秒 ÷ 4 ≈ 20 秒 → **1 シャード 20〜25 分**。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）

cd /home/user/opcg-sim-backend
git fetch origin claude/small-play
git checkout -B claude/small-play-j$I origin/claude/small-play

# Rust エンジンの wheel（maturin が無ければ入れる）
pip install -q maturin
make rust-develop

# 判定（K=8・4 本並べ・出力は n1_results/small_play/points_shard$I.jsonl）
OPCG_LOG_SILENT=1 nohup python tests/scripts/small_play.py judge --k 8 --workers 4 --shard $I/8 \
  > n1_results/small_play/judge_shard$I.log 2>&1 &
```

- 1 行目に `POOL 733 take 400`、2 行目に `JUDGE shard=I/8 todo 50 ...` が出れば正しく始まっている（`todo` が 0 や 50 を超えるならブランチの取り方が違う）。
- 終わると log の最後に `JUDGE_DONE` が出る。再開できる（止まったら同じコマンドを打ち直せば、済んだ点は飛ばす）。

## push（終わったとき）

```bash
git add n1_results/small_play/points_shard$I.jsonl n1_results/small_play/judge_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "small-play: 判定シャード $I/8"
git push -u origin claude/small-play-j$I
```

あわせて `RESULT.json`（リポジトリ直下）を次の形で書いて、同じブランチに push する:

```json
{"wp": "small-play", "shard": "I/8", "status": "done",
 "branch": "claude/small-play-jI",
 "files": ["n1_results/small_play/points_shardI.jsonl", "n1_results/small_play/judge_shardI.log"],
 "points": <points_shardI.jsonl の行数>, "notes": ""}
```

## 決まり
- **コードは変えない**（道具もエンジンも。コードを変えないので `make test` は要らない）。PR は作らない・マージしない。
- `games.jsonl`・`cards.jsonl`・他のシャードのファイルは触らない（集めるのは `claude/small-play` のセッション）。
- 判定できない（`why` が `replay` 等）行が多いときは、wheel が本線と違う疑い。`notes` に書いて完走する。
