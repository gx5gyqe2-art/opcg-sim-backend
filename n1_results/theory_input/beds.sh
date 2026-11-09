# $1 = tag ; 残り = ネット（理論の列を読むネットも可・sidecar は 70f/74c に要る）
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend
tag=$1; shift
D=n1_results/theory_input/beds; mkdir -p $D
python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag ti_$tag --nets "$@" > $D/tvval_$tag.log 2>&1 && mv n1_results/value_teach/tvval_ti_$tag.json $D/tvval_$tag.json
python tests/scripts/don_jitter_rows.py rows --nets "$@" > $D/jitter_rows_$tag.log 2>&1 && mv n1_results/don_jitter/jitter_rows.json $D/jitter_rows_$tag.json && git checkout -q n1_results/don_jitter/jitter_rows.json
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel "$@" --holdout-mod 7 --theory-in --out $D/band_n74c_$tag.json > $D/band_n74c_$tag.log 2>&1
python tests/scripts/theory_input.py calib --in '~/n74c/n_records/n74c_w*_p[0-3]' --theory-in --nets "$@" --tag n74c_$tag --out $D > $D/calib_n74c_$tag.log 2>&1
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n70f/n_records/n70f_w*_p[0-3] --nrel "$@" --holdout-mod 7 --theory-in --out $D/band_n70f_$tag.json > $D/band_n70f_$tag.log 2>&1
python tests/scripts/theory_input.py weights --nets "$@" --tag $tag --out $D > $D/weights_$tag.log 2>&1
echo BEDS_DONE $tag >> $D/band_n70f_$tag.log
