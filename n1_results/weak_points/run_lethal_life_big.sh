#!/bin/sh
# §4 の続き: 節点の上限を 4 倍（80,000）にして off と mode0 を判定し直す（減った証明が予算の打ち切りか本当の no かを分ける）。
cd /home/user/opcg-sim-backend
# PYTHONPATH は作業用の置き場の wheel（life_deal_port.diff を当てた worktree で maturin build → pip install --target）
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1 PYTHONPATH=${LIFE_SITE:?life_deal_port.diff の wheel の site}
O=n1_results/weak_points/lethal
R=n1_results/lethal_speed/record_3012000_300.jsonl.gz
for v in off mode0; do
  OPCG_LETHAL_LIFE=$v python tests/scripts/lethal_speed_check.py judge --record $R --out $O --tag life_${v}_n80k --max-nodes 80000 --workers 1 > $O/judge_life_${v}_n80k.log 2>&1
  echo "LIFE ${v}_n80k DONE" >> $O/life_progress.log
done
