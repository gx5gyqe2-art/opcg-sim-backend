#!/bin/sh
# 分かれた局だけ、入れない形と入れた形の守りの監査を分かれた点から先だけ打ち、確かめ・数える（§7.51）。
cd /home/user/opcg-sim-backend
D=n1_results/defense_proof
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
NET=opcg_sim/data/learned/nrel_r13d.npz
python -c "
import json
s=[json.loads(l) for l in open('$D/diverge_3351000.jsonl')]
print(' '.join(str(r['seed']) for r in sorted(s, key=lambda r: r['seed']) if r['diverged']))
" > $D/div_seeds.txt
for form in off on; do
  F=$D/partial_$form
  X=""; [ $form = on ] && X="--defense-proof"
  python $D/defense_partial.py 3351000 run --seeds $D/div_seeds.txt --workers 4 --net $NET --out $F $X > $D/partial_${form}_run.log 2>&1
  P=$F/points_seeds_div_seeds.jsonl
  python $D/defense_partial.py 3351000 verify --points $P --workers 4 --net $NET --out $F $X > $D/partial_${form}_verify.log 2>&1
  python $D/defense_partial.py 3351000 summarize --points $P --out $F > $D/partial_${form}_summary.log 2>&1
  echo "PARTIAL_${form}_DONE $(date -u +%H:%M)" >> $D/progress.log
done
