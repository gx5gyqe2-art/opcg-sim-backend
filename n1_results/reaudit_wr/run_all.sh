#!/bin/sh
# WP reaudit-wr（§7.43）: 大きな損の洗い出し（100 局・seed 3341000〜）と守りの見逃し（300 局・seed 3351000〜）を 2+2 並走。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_wr
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
blunder() {
  B=$D/blunder
  python tests/scripts/blunder_audit.py play --games 100 --seed-base 3341000 --workers 2 --out $B > $D/blunder_play.log 2>&1
  echo STAGE_PLAY_DONE >> $D/progress.log
  python tests/scripts/blunder_audit.py judge --workers ${BW:-2} --out $B > $D/blunder_judge.log 2>&1
  echo STAGE_JUDGE_DONE >> $D/progress.log
  python tests/scripts/blunder_audit.py recheck --workers ${BW:-2} --out $B > $D/blunder_recheck.log 2>&1
  python tests/scripts/blunder_audit.py verify --workers ${BW:-2} --out $B > $D/blunder_verify.log 2>&1
  echo STAGE_VERIFY_DONE >> $D/progress.log
  python tests/scripts/blunder_audit.py winstab --workers ${BW:-2} --out $B > $D/blunder_winstab.log 2>&1
  python tests/scripts/blunder_audit.py summary --out $B > $D/blunder_summary.log 2>&1
  echo BLUNDER_DONE >> $D/progress.log
}
defense() {
  F=$D/defense
  python tests/scripts/defense_audit.py run --games 300 --seed-base 3351000 --workers 2 --out $F > $D/defense_run.log 2>&1
  echo STAGE_DEF_RUN_DONE >> $D/progress.log
  python tests/scripts/defense_audit.py verify --games 300 --seed-base 3351000 --workers 2 --out $F > $D/defense_verify.log 2>&1
  python tests/scripts/defense_audit.py summarize --games 300 --seed-base 3351000 --out $F > $D/defense_summary.log 2>&1
  echo DEFENSE_DONE >> $D/progress.log
}
case "$1" in
  blunder) blunder ;;
  defense) defense ;;
  *) blunder & defense & wait ;;
esac
