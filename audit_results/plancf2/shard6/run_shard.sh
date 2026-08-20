#!/bin/bash
cd /home/user/opcg-sim-backend
OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness:tests/scripts nohup python \
  tests/scripts/plan_cf2_gen.py --games 40 --seed-base 716000 --workers 4 \
  --points 3 --worlds 3 --rollout-turns 4 --drift 0 --shard-size 16 \
  --cand /home/user/cand_planA2/value.npz,/home/user/cand_planA2/policy.npz \
  --out /home/user/plancf2_s6 > /home/user/plancf2_s6.log 2>&1 &
