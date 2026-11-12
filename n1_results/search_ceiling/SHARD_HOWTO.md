# search-ceiling 段 1 の読みのシャード（§7.90・作業セッション用）

コーディネータの指示で「シャード KK（01〜10）」を回す作業セッションの手順。**コードは変えない**（読み取りだけ）。
1 シャード＝80 点（段 0 の 20 点を除いた残り）を 10 に割った 8 点。各点を世界 8 本で 1・10・100・1,000 倍に読む。
1 点あたりの CPU は 1 倍 2 秒・10 倍 26 秒・100 倍 5 分・**1,000 倍 約 60 分（重い点は 4 時間）**。
1 シャードは 3 並列で約 3〜5 時間。メモリは 1,000 倍で 1 本あたり最大 3.5GB（3 並列まで・4 並列にしない）。

## 1. 用意

```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/search-ceiling && git checkout -B claude/search-ceiling origin/claude/search-ceiling
pip install -q -r opcg_sim/requirements.txt maturin && make rust-develop     # wheel（無いと動かない）
python -c "import opcg_engine; print('ok')"
export OPENBLAS_CORETYPE=Haswell
mkdir -p logs
```

## 2. 回す（KK を自分の番号に・零詰め 2 桁）

```bash
KK=01                                   # ← 自分の番号
K=$((10#$KK))
F=shards/s$KK.jsonl
nohup sh -c "
python tests/scripts/search_ceiling.py run --rest --shard $K/10 --mults 1,10,100 --worlds 8 --workers 4 --file $F &&
python tests/scripts/search_ceiling.py run --rest --shard $K/10 --mults 1000 --worlds 8 --workers 3 --file $F &&
echo ALL_DONE" > logs/sc_s$KK.log 2>&1 &
```

- 見張りは `tail -3 logs/sc_s$KK.log`（9 分以内の待ちを繰り返す）。最後の行が `ALL_DONE` になれば終わり。
- 途中で止まっても同じコマンドで続きから回る（終えた組は飛ばす）。
- `fail` の行（`timeout`・`exit …`）が出たら、もう 1 回同じコマンドを回す。2 回落ちた組はそのまま残して報告に書く。

## 3. 確かめて push

```bash
python - "$KK" <<'EOF'
import json, sys
kk = sys.argv[1]
rows = [json.loads(l) for l in open(f"n1_results/search_ceiling/shards/s{kk}.jsonl")]
ok = [r for r in rows if not r.get("fail")]
print("rows", len(rows), "ok", len(ok), "fail", len(rows) - len(ok))
print("1x が記録の手と同じ", sum(r["rec_same"] for r in ok if r["mult"] == 1), "/", sum(r["mult"] == 1 for r in ok))
json.dump({"job": "search-ceiling", "shard": kk, "rows": len(rows), "ok": len(ok),
           "fail": [[r["pid"], r["mult"], r.get("fail")] for r in rows if r.get("fail")],
           "cpu_h": round(sum(r.get("cpu") or 0 for r in ok) / 3600, 2)},
          open(f"n1_results/search_ceiling/shards/s{kk}_RESULT.json", "w"), ensure_ascii=False, indent=1)
EOF
git checkout -B claude/search-ceiling-s$KK
git add n1_results/search_ceiling/shards/s$KK.jsonl n1_results/search_ceiling/shards/s${KK}_RESULT.json
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "search-ceiling s$KK: 8 points x 1/10/100/1000 (worlds 8)"
git push -u origin claude/search-ceiling-s$KK
```

合格の目安: 32 行（8 点 × 4 倍率）・fail 0・「1x が記録の手と同じ」が 8／8。
終わったら @parent に 3 行以内で知らせる（行数・fail の数・ブランチ名）。**PR は作らない・main へ push しない。**
