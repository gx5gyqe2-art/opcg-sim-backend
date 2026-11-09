#!/bin/sh
# 守り 300 局を束ねて集計する（このセッションの 4462000〜099 ＋ シャード s07・s08）。束ねた points は再生成できるのでコミットしない。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_r29f4
cat $D/defense/points_4462000_100.jsonl $D/defense_s07/points_4462100_100.jsonl $D/defense_s08/points_4462200_100.jsonl > $D/defense_all/points_4462000_300.jsonl
cat $D/defense/verify.jsonl $D/defense_s07/verify.jsonl $D/defense_s08/verify.jsonl > $D/defense_all/verify.jsonl
PYTHONHASHSEED=0 OPCG_LOG_SILENT=1 python tests/scripts/defense_audit.py summarize --games 300 --seed-base 4462000 --out $D/defense_all > $D/defense_all/summary.log 2>&1
