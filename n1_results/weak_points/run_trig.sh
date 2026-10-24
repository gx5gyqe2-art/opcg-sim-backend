#!/bin/sh
# §5: トリガーの発動確認の点（生成と同じ設定・r22・seed 3671000〜3671199）を集めて審判で比べる。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python n1_results/weak_points/trig_collect.py --seed-base 3671000 --games 200 --workers 3 --k 16 > n1_results/weak_points/trig.log 2>&1
