#!/bin/sh
# 判定の続き（4 並走・最初の 40 局＝400 点で止める〔1 点の所要が 10-23 の監査の約 3 倍のため・notes〕）→ 打ち直し → 確かめ → 窓の揺れ → 集計。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_wr; B=$D/blunder
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
DONE=$(wc -l < $B/points.jsonl)
python tests/scripts/blunder_audit.py judge --workers 4 --limit $((400 - DONE)) --out $B >> $D/blunder_judge.log 2>&1
echo STAGE_JUDGE_DONE >> $D/progress.log
python tests/scripts/blunder_audit.py recheck --workers 4 --out $B > $D/blunder_recheck.log 2>&1
python tests/scripts/blunder_audit.py verify --workers 4 --out $B > $D/blunder_verify.log 2>&1
echo STAGE_VERIFY_DONE >> $D/progress.log
python tests/scripts/blunder_audit.py winstab --workers 4 --out $B > $D/blunder_winstab.log 2>&1
python tests/scripts/blunder_audit.py summary --out $B > $D/blunder_summary.log 2>&1
echo BLUNDER_DONE >> $D/progress.log
