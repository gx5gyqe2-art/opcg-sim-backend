#!/bin/sh
# §4: ライフの枠へ守りに働く【トリガー】を入れる配り方（lethal-life-deal の切替 c1fdf65 を今の lethal.rs へ移した
# scratch の wheel・コミットしない）で同じ記録を判定し直す。off は本線と同じはず（確かめ）。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1 PYTHONPATH=/tmp/claude-0/-home-user-opcg-sim-backend/179c806c-8cbe-50e7-880c-be212cf2a3e6/scratchpad/site_life
O=n1_results/weak_points/lethal
R=n1_results/lethal_speed/record_3012000_300.jsonl.gz
for v in off both mode0 add; do
  OPCG_LETHAL_LIFE=$v python tests/scripts/lethal_speed_check.py judge --record $R --out $O --tag life_$v --workers 1 > $O/judge_life_$v.log 2>&1
  echo "LIFE $v DONE" >> $O/life_progress.log
done
