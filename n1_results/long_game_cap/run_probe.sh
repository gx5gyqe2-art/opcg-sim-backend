#!/bin/sh
# 長い局の収集（§7.47）: 生成と同じ設定（gen_n52.json の引数・生成役 r22・デッキ real）・800 局・seed 3661000〜3661799・上限 1600 手。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python n1_results/long_game_cap/long_probe.py n1_results/long_game_cap/probe 3661000 3661799 4 >> n1_results/long_game_cap/probe.log 2>&1
