#!/bin/sh
# 長い局の収集（§7.47）: 生成と同じ設定（gen_n52.json の引数・生成役 r22・デッキ real）・800 局・seed 3661000〜3661799・上限 1600 手。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python n1_results/long_game_cap/long_probe.py n1_results/long_game_cap/probe 3661000 3661799 4 >> n1_results/long_game_cap/probe.log 2>&1
# 追加: ベガパンク OP07-097 が席に着く局だけ（seed 3662000〜3671999 のうち 190 局・vega_seeds.json）
SEEDS_FILE=n1_results/long_game_cap/vega_seeds.json python n1_results/long_game_cap/long_probe.py n1_results/long_game_cap/probe_vega 3662000 3671999 4 >> n1_results/long_game_cap/probe_vega.log 2>&1
