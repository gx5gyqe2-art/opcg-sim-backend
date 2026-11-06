#!/bin/sh
# シャード s01〜s05 の判定・確かめ・窓の揺れを blunder/ に束ね（あるものだけ）、recheck をここで回してから summary。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_r29f4; B=$D/blunder
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
for f in points verify window_stability; do cat $D/shards/s*/$f.jsonl > $B/$f.jsonl 2>/dev/null; done
python $D/ba.py recheck --workers ${W:-2} --out $B >> $D/blunder_recheck.log 2>&1
python $D/ba.py summary --out $B > $D/blunder_summary.log 2>&1
echo MERGE_DONE >> $D/progress.log
