#!/bin/sh
# diverge（実行中）の後に、分かれた局の監査 → 費用を順に回す。
cd /home/user/opcg-sim-backend
D=n1_results/defense_proof
until grep -q DIVERGE_DONE $D/progress.log; do sleep 30; done
sh $D/run_partial.sh
sh $D/run_latency.sh
echo "ALL_DONE $(date -u +%H:%M)" >> $D/progress.log
