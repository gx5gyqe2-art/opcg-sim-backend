# search-deep 段 0 の読みのシャード（§7.91・作業セッション用・Sonnet）

コーディネータの指示で「シャード KK（01〜10）」を回す作業セッションの手順。**コードは変えない**（読み取りだけ）。
1 シャード＝79 点（段 0 の 20 点と点 56 を除いた残り）を 10 に割った 7〜8 点。各点を、下の `SHAPES` の形ごとに 100 倍
（総数は今の 100 倍と同じ）で読み、今の形の 100 倍（§7.90 の記録）と手が替わった組を審判にかける。

- 1 点 × 1 形の CPU は平均 約 5〜10 分（重い点は 40 分）。1 シャードは 4 並列で **約 2〜4 時間**（読み）＋審判 約 1 時間。
- メモリ: 世界 8 本の形は 1 本あたり最大 約 1.3GB（4 並列でよい）。**世界の本数を減らした形（名前が `w1`・`w2` で始まる）は
  1 本あたり 最大 約 5GB＝その形だけ 2 並列**（下のコマンドはそう分けてある）。

## 1. 用意

```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/search-deep && git checkout -B claude/search-deep origin/claude/search-deep
pip install -q -r opcg_sim/requirements.txt maturin && make rust-develop     # wheel（無いと動かない）
python -c "import opcg_engine; print('ok')"
export OPENBLAS_CORETYPE=Haswell
mkdir -p logs
```

## 2. 回す（KK を自分の番号に・零詰め 2 桁）

```bash
KK=01                                   # ← 自分の番号
K=$((10#$KK))
SHAPES=t4,k2,k2c4,ok1                      # 世界 8 本の形（4 並列）
WSHAPES=w2k2f2c4,w1c4                    # 世界を減らした形（2 並列・空なら飛ばす）
F=shards/s$KK.jsonl
nohup sh -c "
python tests/scripts/search_deep.py run --rest --shard $K/10 --shapes $SHAPES --workers 4 --file $F &&
{ [ -z \"$WSHAPES\" ] || python tests/scripts/search_deep.py run --rest --shard $K/10 --shapes $WSHAPES --workers 2 --file $F; } &&
python tests/scripts/search_deep.py judge --rest --shapes $SHAPES${WSHAPES:+,$WSHAPES} --workers 4 --judge-file shards/judge_s$KK.jsonl &&
echo ALL_DONE" > logs/sd_s$KK.log 2>&1 &
```

- 見張りは `tail -3 logs/sd_s$KK.log`（9 分以内の待ちを繰り返す）。最後の行が `ALL_DONE` になれば終わり。
- 途中で止まっても同じコマンドで続きから回る（終えた組は飛ばす）。
- `fail` の行（`timeout`・`exit …`）が出たら、もう 1 回同じコマンドを回す。**`exit -9` はメモリ不足で殺された印**＝その形の
  `--workers` を 1 つ減らして回し直す。2 回落ちた組はそのまま残して報告に書く。

## 3. 確かめて push

```bash
python - "$KK" <<'EOF'
import json, sys
kk = sys.argv[1]
d = "n1_results/search_deep/shards/"
rows = [json.loads(l) for l in open(d + f"s{kk}.jsonl")]
ok = [r for r in rows if not r.get("fail")]
try:
    judge = [json.loads(l) for l in open(d + f"judge_s{kk}.jsonl")]
except FileNotFoundError:
    judge = []
print("rows", len(rows), "ok", len(ok), "fail", len(rows) - len(ok), "judge", len(judge))
json.dump({"job": "search-deep", "shard": kk, "rows": len(rows), "ok": len(ok),
           "fail": [[r["pid"], r["shape"], r.get("fail")] for r in rows if r.get("fail")],
           "judge_pairs": len(judge), "judge_errors": sum(1 for j in judge if j.get("error")),
           "cpu_h": round(sum(r.get("cpu") or 0 for r in ok) / 3600, 2)},
          open(d + f"s{kk}_RESULT.json", "w"), ensure_ascii=False, indent=1)
EOF
git checkout -B claude/search-deep-s$KK
git add -f n1_results/search_deep/shards/s$KK.jsonl n1_results/search_deep/shards/judge_s$KK.jsonl n1_results/search_deep/shards/s${KK}_RESULT.json
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "search-deep s$KK: shapes at 100x and judge"
git push -u origin claude/search-deep-s$KK
```

合格の目安: 行数＝（点の数 7〜8）×（形の数）・fail 0・審判の error 0（手が 1 つも替わらなければ審判のファイルは無い＝`git add` から外す）。
終わったら @parent に 3 行以内で知らせる（行数・fail の数・審判の組の数・ブランチ名）。**PR は作らない・main へ push しない。**
