# reaudit-r29f4 の大きな損の判定シャード（§7.84 の 1・作業セッション用）

コーディネータの指示で「シャード KK（01〜05）」を回す作業セッションの手順。**コードは変えない**（読み取りだけ）。
1 シャード＝20 局（各局 10 点＝200 点）の判定 → 打ち直し → 確かめ → 窓の揺れ。4 並列で約 3〜4 時間（判定 1 点 ≈ 185 秒・1 コア）。

| シャード | 局の seed |
|---|---|
| s01 | 4461000〜4461019 |
| s02 | 4461020〜4461039 |
| s03 | 4461040〜4461059 |
| s04 | 4461060〜4461079 |
| s05 | 4461080〜4461099 |

## 1. 用意

```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/reaudit-r29f4 && git checkout -B claude/reaudit-r29f4 origin/claude/reaudit-r29f4
pip install -q maturin && make rust-develop          # wheel（無いと動かない）
python -c "import opcg_engine; print('ok')"
```

## 2. 回す（KK を自分の番号に・零詰め 2 桁）

```bash
KK=01                                   # ← 自分の番号
LO=$((4461000 + 20 * (10#$KK - 1)))
O=n1_results/reaudit_r29f4/shards/s$KK
mkdir -p $O
python - "$LO" "$O" <<'EOF'
import json, sys
lo, o = int(sys.argv[1]), sys.argv[2]
n = 0
with open(f"{o}/games.jsonl", "w", encoding="utf-8") as w:
    for line in open("n1_results/reaudit_r29f4/blunder/games.jsonl", encoding="utf-8"):
        g = json.loads(line)
        if lo <= g["seed"] < lo + 20:
            w.write(line); n += 1
print("games", n)
EOF
cat > $O/run.sh <<EOF
#!/bin/sh
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
B=n1_results/reaudit_r29f4/ba.py
python \$B judge   --workers 4 --out $O > $O/judge.log 2>&1
python \$B recheck --workers 4 --out $O > $O/recheck.log 2>&1
python \$B verify  --workers 4 --out $O > $O/verify.log 2>&1
python \$B winstab --workers 4 --out $O > $O/winstab.log 2>&1
python \$B summary --out $O > $O/summary.log 2>&1
echo ALL_DONE > $O/ALL_DONE
EOF
setsid nohup sh $O/run.sh > /dev/null 2>&1 < /dev/null &
```

- `games` が 20 と出ること。`ba.py` は器 `tests/scripts/blunder_audit.py` の定数だけを差し替える入口
  （出荷の席＝r29f4・審判 1〔出力の鍵 `r13d`〕＝r29f4・審判 2〔鍵 `r20`〕＝r28x4）。
- 再開できる（終えた点は飛ばす）。落ちたら同じ `run.sh` をもう一度。
- **見張り**: ターンを終えずに 9 分以内の待ちを繰り返し、`tail -2 $O/judge.log` などで進みを見る。`ALL_DONE` が出たら次へ。
  判定は 200 点（`wc -l $O/points.jsonl` が 200 前後・候補が 2 つ以上の点が 10 未満の局はそのぶん少ない）。

## 3. 確かめ

```bash
wc -l $O/points.jsonl $O/recheck.jsonl $O/verify.jsonl $O/window_stability.jsonl
grep -c '"ok": true' $O/points.jsonl
tail -30 $O/summary.log
```
`recheck`／`verify` は損の候補（`cand`）の数だけ・0 行でもよい（候補が無ければ 0）。

## 4. push と RESULT.json

`games.jsonl` はコミットしない（本線のブランチに元が在る）。

```bash
rm $O/games.jsonl
python - "$KK" "$O" <<'EOF'
import json, sys, os
kk, o = sys.argv[1], sys.argv[2]
cnt = lambda f: sum(1 for _ in open(os.path.join(o, f))) if os.path.exists(os.path.join(o, f)) else 0
pts = [json.loads(l) for l in open(os.path.join(o, "points.jsonl"))]
json.dump({"wp": "reaudit-r29f4", "shard": f"s{kk}", "points": len(pts), "ok": sum(1 for p in pts if p.get("ok")),
           "cand": sum(1 for p in pts if p.get("cand")), "recheck": cnt("recheck.jsonl"), "verify": cnt("verify.jsonl"),
           "winstab": cnt("window_stability.jsonl")}, open(os.path.join(o, "RESULT.json"), "w"), indent=1)
print(open(os.path.join(o, "RESULT.json")).read())
EOF
git checkout -b claude/reaudit-r29f4-s$KK
git add $O && git commit -m "reaudit-r29f4 shard s$KK: blunder judge/recheck/verify/winstab for 20 games"
git push -u origin claude/reaudit-r29f4-s$KK      # 失敗したら 2s・4s・8s・16s で再試行
```
最後にコーディネータへ「s$KK 完了（点・候補・確かめの数）」と知らせる。PR は作らない・マージしない・コミットにモデル名を書かない。

---

# シャード s06: 自分の手番の続き 50 局（§7.84 の 3・`commit_audit.py`）

用意は上の §1 と同じ。4 並列で約 1〜3 時間の見込み（深い読みの比べ込み）。

```bash
cd /home/user/opcg-sim-backend
O=n1_results/reaudit_r29f4/commit
mkdir -p $O
cat > $O/run.sh <<'EOS'
#!/bin/sh
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
O=n1_results/reaudit_r29f4/commit
python tests/scripts/commit_audit.py run --games 50 --seed-base 4463000 --workers 4 --judge-window-new --out $O > $O/run.log 2>&1
python tests/scripts/commit_audit.py summary --out $O > $O/summary.log 2>&1
echo ALL_DONE > $O/ALL_DONE
EOS
setsid nohup sh $O/run.sh > /dev/null 2>&1 < /dev/null &
```
- ネットは既定（`E.DEFAULT_NET`＝r29f4）・出荷の設定（`E.PLAY_*`）。深い読み（判定役）の窓も今の形（`--judge-window-new`）。
- 見張り: `tail -2 $O/run.log`（`[k/50]` が進む）。`ALL_DONE` で次へ。
- 確かめ: `ls $O`（`points_4463000_50.jsonl` と `summary.json`）・`tail -40 $O/summary.log`。
- push: `points_*.jsonl` が 20MB を超えたら `gzip` してから。RESULT.json は `{"wp": "reaudit-r29f4", "shard": "s06", "games": 50, "summary": "n1_results/reaudit_r29f4/commit/summary.json"}` を `$O/RESULT.json` に。

```bash
git checkout -b claude/reaudit-r29f4-s06
git add $O && git commit -m "reaudit-r29f4 shard s06: commit audit of shipped r29f4 (50 games, seed 4463000-4463049)"
git push -u origin claude/reaudit-r29f4-s06
```

---

# シャード s07・s08: 守りの見逃し 100 局ずつ（§7.84 の 2・`defense_audit.py`・§7.24 の設定・確かめ込み）

用意は上の §1 と同じ。1 局平均 ≈ 108 秒（4 並列の実時間）＝ run 約 3 時間 ＋ verify。
s07＝seed 4462100〜4462199・s08＝seed 4462200〜4462299（4462000〜099 はコーディネータ側の作業セッションが回す）。

```bash
cd /home/user/opcg-sim-backend
KK=07                                     # ← 07 か 08
BASE=$([ "$KK" = 07 ] && echo 4462100 || echo 4462200)
O=n1_results/reaudit_r29f4/defense_s$KK
mkdir -p $O
cat > $O/run.sh <<EOS
#!/bin/sh
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python tests/scripts/defense_audit.py run --games 100 --seed-base $BASE --workers 4 --out $O > $O/run.log 2>&1
python tests/scripts/defense_audit.py verify --games 100 --seed-base $BASE --workers 4 --out $O > $O/verify.log 2>&1
python tests/scripts/defense_audit.py summarize --games 100 --seed-base $BASE --out $O > $O/summary.log 2>&1
echo ALL_DONE > $O/ALL_DONE
EOS
setsid nohup sh $O/run.sh > /dev/null 2>&1 < /dev/null &
```
- ネットは既定（r29f4）。設定は器の既定（世界 1 本 × 1,280 回・予算なし・証明 off・random × synth_roles）＝何も足さない。
- 見張り: `tail -1 $O/run.log`（`[k/100]`）→ verify.log → `ALL_DONE`。再開できる（落ちたら `run.sh` をもう一度）。
- 確かめ: `ls $O`（`points_${BASE}_100.jsonl`・`verify.jsonl`・`summary.json` 等）・`tail -40 $O/summary.log`。
- RESULT.json: `{"wp": "reaudit-r29f4", "shard": "s$KK", "seed_base": BASE, "games": 100}` を `$O/RESULT.json` に。

```bash
git checkout -b claude/reaudit-r29f4-s$KK
git add $O && git commit -m "reaudit-r29f4 shard s$KK: defense audit of r29f4 (100 games from seed $BASE)"
git push -u origin claude/reaudit-r29f4-s$KK
```
