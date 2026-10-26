#!/bin/sh
# 守りの再監査の修正後の側（§7.24 の設定・4 並走）。修正前は rs-commit-hidden の new300（同じエンジンの挙動・2+2 並走＝同じ負荷）を使う。
cd /home/user/opcg-sim-backend
D=n1_results/window_read
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python tests/scripts/defense_audit.py run --games 300 --seed-base 3013000 --workers 4 --out $D/defense_new >> $D/defense_run_new.log 2>&1
python tests/scripts/defense_audit.py verify --games 300 --seed-base 3013000 --workers 4 --out $D/defense_new > $D/defense_verify_new.log 2>&1
python tests/scripts/defense_audit.py summarize --games 300 --seed-base 3013000 --out $D/defense_new > $D/defense_summary_new.log 2>&1
echo DEFENSE_DONE >> $D/defense_summary_new.log
