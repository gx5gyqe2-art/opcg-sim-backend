#!/bin/sh
# 手が分かれた局を探し終えたら: 窓の揺れ（226 点）→ 分かれた局だけ守りの監査を修正後で打ち直す（2 つの局面）。
cd /home/user/opcg-sim-backend
D=n1_results/window_tree_blocker
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
until grep -q DIVERGE_ALL_DONE $D/progress.log 2>/dev/null; do sleep 30; done
W="python tests/scripts/blunder_audit.py winstab --workers 4 --tag-out $D"
$W --tag blocker > $D/winstab_blocker.log 2>&1
$W --tag counter_legacy --opts '{"window_tree_counter_legacy": true}' > $D/winstab_counter_legacy.log 2>&1
echo WINSTAB_DONE >> $D/progress.log
for B in 3013000 3351000; do
  python -c "import json;print(' '.join(str(d['seed']) for d in map(json.loads, open('$D/diverge_$B.jsonl')) if d['diverged']))" > $D/div_seeds_$B.txt
  python tests/scripts/defense_audit.py run --seeds $D/div_seeds_$B.txt --workers 4 --net opcg_sim/data/learned/nrel_r13d.npz \
    --out $D/defense_$B > $D/defense_run_$B.log 2>&1
  echo DEF_RUN_DONE_$B >> $D/progress.log
done
echo CHAIN_DONE >> $D/progress.log
