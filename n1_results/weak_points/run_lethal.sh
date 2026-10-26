#!/bin/sh
# §4: 詰みの証明の不変量の器（§7.25 の記録 300 局）を今の本線で判定し直す（既定＝cost・本当の配り方だけも）。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
O=n1_results/weak_points/lethal
R=n1_results/lethal_speed/record_3012000_300.jsonl.gz
python tests/scripts/lethal_speed_check.py judge --record $R --out $O --tag now --workers 1 > $O/judge_now.log 2>&1
python tests/scripts/lethal_speed_check.py judge --record $R --out $O --tag now_true --true-deal --workers 1 > $O/judge_now_true.log 2>&1
echo LETHAL_DONE >> $O/judge_now.log
