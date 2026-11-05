#!/bin/bash
cd /home/user/opcg-sim-backend
until grep -q EP0_BEDS_DONE ~/vn/logs/ep0beds.log; do sleep 20; done
pgrep -f n_rel_band >/dev/null && sleep 30
rm -rf $HOME/packcache/n74c_*; echo "disk after rm 74c pack: $(df -h / | tail -1)"
nohup bash ~/vn/train.sh > /dev/null 2>&1 &
echo "TRAIN_RERUN_STARTED main arm $(date -u +%H:%M)"
