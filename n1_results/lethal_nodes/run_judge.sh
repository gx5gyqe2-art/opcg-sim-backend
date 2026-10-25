#!/bin/sh
# §7.49: 証明の数と偽の証明（§7.25 の記録 300 局・12,7xx 点）を節点の上限 20k／40k／80k／160k で判定し直す。
# 偽の証明は本当の配り方 1 通りだけの判定（--true-deal・上限 160k）で「勝てない」点。
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
O=n1_results/lethal_nodes
R=n1_results/lethal_speed/record_3012000_300.jsonl.gz
CPUS=${CPUS:-0,1}
W=${W:-2}
for n in ${NODES:-20000 40000 80000 160000}; do
  t=n$((n/1000))k
  [ -f $O/judge_$t.jsonl.gz ] && continue
  taskset -c $CPUS python tests/scripts/lethal_speed_check.py judge --record $R --out $O --tag $t --workers $W --max-nodes $n > $O/judge_$t.log 2>&1
  gzip -f $O/judge_$t.jsonl
done
if [ -n "$TRUE" ] && [ ! -f $O/judge_true_n160k.jsonl.gz ]; then
  taskset -c $CPUS python tests/scripts/lethal_speed_check.py judge --record $R --out $O --tag true_n160k --true-deal --workers $W --max-nodes 160000 > $O/judge_true_n160k.log 2>&1
  gzip -f $O/judge_true_n160k.jsonl
fi
echo JUDGE_DONE >> $O/judge.done
