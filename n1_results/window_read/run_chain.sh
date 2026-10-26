#!/bin/sh
# commit-audit の後に audit-cross・守りの再監査（修正前＝3 つの旧い旗・修正後）を回す（WP rs-window-read）。
cd /home/user/opcg-sim-backend
D=n1_results/window_read
until grep -q RUN_DONE $D/commit_audit_run.log; do sleep 30; done
make audit-cross > $D/audit_cross.log 2>&1; echo "AUDIT_CROSS_EXIT=$?" >> $D/audit_cross.log
side() {  # $1=名前 $2=旗
  export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
  python tests/scripts/defense_audit.py run --games 300 --seed-base 3013000 --workers 2 --out $D/defense_$1 $2 > $D/defense_run_$1.log 2>&1
  python tests/scripts/defense_audit.py verify --games 300 --seed-base 3013000 --workers 2 --out $D/defense_$1 $2 > $D/defense_verify_$1.log 2>&1
  python tests/scripts/defense_audit.py summarize --games 300 --seed-base 3013000 --out $D/defense_$1 > $D/defense_summary_$1.log 2>&1
  echo DONE >> $D/defense_summary_$1.log
}
side legacy --window-legacy &
side new "" &
wait
echo CHAIN_DONE > $D/chain.done
