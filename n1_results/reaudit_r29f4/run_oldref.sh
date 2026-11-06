#!/bin/sh
# 切り分け（コーディネータの照会）:
#  1) 損の候補 17 点を、審判だけ前回の r13d／r20 にして打ち直し・確かめ（出荷の側は r29f4 のまま）。
#  2) 守りの確かめ後 12 点を、旧い確かめの形（守る側だけ出荷の設定）で打ち比べ。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_r29f4
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python $D/ba_oldref.py recheck --workers 4 --out $D/blunder_oldref > $D/blunder_oldref/recheck.log 2>&1
python $D/ba_oldref.py verify --workers 4 --out $D/blunder_oldref > $D/blunder_oldref/verify.log 2>&1
python $D/ba_oldref.py summary --out $D/blunder_oldref > $D/blunder_oldref/summary.log 2>&1
python tests/scripts/defense_audit.py verify --workers 4 --points $D/defense_points12.jsonl --out $D/defense_oldform --verify-mover-only --verify-file verify_mover.jsonl > $D/defense_oldform/verify_mover.log 2>&1
echo OLDREF_DONE >> $D/progress.log
