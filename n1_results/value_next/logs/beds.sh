#!/bin/bash
# $1 = tag ; $2 = "fast" (tvval+jitter) | "band74c" | "band70f" | "rocks" ; rest = nets
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache PYTHONPATH=$HOME/torch_env
tag=$1; what=$2; shift 2
D=n1_results/value_next/beds; mkdir -p $D
case $what in
fast)
  python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag vn_$tag --nets "$@" > $D/tvval_$tag.log 2>&1 && mv n1_results/value_teach/tvval_vn_$tag.json $D/tvval_$tag.json
  python tests/scripts/don_jitter_rows.py rows --nets "$@" > $D/jitter_rows_$tag.log 2>&1 && mv n1_results/don_jitter/jitter_rows.json $D/jitter_rows_$tag.json && git checkout -q n1_results/don_jitter/jitter_rows.json
  echo FAST_DONE $tag >> $D/jitter_rows_$tag.log ;;
band74c)
  python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel "$@" --holdout-mod 7 --out $D/band_n74c_$tag.json > $D/band_n74c_$tag.log 2>&1
  echo BAND74C_DONE $tag >> $D/band_n74c_$tag.log ;;
band70f)
  python -m opcg_sim.learned.train.n_rel_band --in $HOME/n70f/n_records/n70f_w*_p[0-3] --nrel "$@" --holdout-mod 7 --out $D/band_n70f_$tag.json > $D/band_n70f_$tag.log 2>&1
  echo BAND70F_DONE $tag >> $D/band_n70f_$tag.log ;;
rocks)
  python tests/scripts/value_teach.py rocks --out $D --tag $tag --nets "$@" > $D/rocks_$tag.log 2>&1
  echo ROCKS_DONE $tag >> $D/rocks_$tag.log ;;
esac
