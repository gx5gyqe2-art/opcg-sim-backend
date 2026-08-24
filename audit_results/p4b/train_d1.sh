#!/bin/bash
cd /home/user/opcg-sim-backend
export PYTHONPATH=tests:tests/harness:tests/scripts OPCG_LOG_SILENT=1
python tests/scripts/exit_head_finetune.py --head battle --replace-head --base gen15 \
  --enc-version 12 --margin 0.03 --head-hidden 4 --epochs 64 \
  --dirs /home/user/defcf_v12,/home/user/plandef_a,/home/user/plandef_smoke --globs "*.npz" \
  --center-dirs /home/user/defcf_v12 --center-glob "*.npz" \
  --out /home/user/cand_D1 2>&1 | tail -3
echo TRAIN_D1_DONE
