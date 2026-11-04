# $1 = tag ; rest = nets
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packlinks
tag=$1; shift
D=n1_results/data_scale
python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag ds_$tag --nets "$@" > $D/tvval_$tag.log 2>&1 && mv n1_results/value_teach/tvval_ds_$tag.json $D/tvval_$tag.json
python tests/scripts/don_jitter_rows.py rows --nets "$@" > $D/jitter_rows_$tag.log 2>&1 && mv n1_results/don_jitter/jitter_rows.json $D/jitter_rows_$tag.json && git checkout -q n1_results/don_jitter/jitter_rows.json
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel "$@" --holdout-mod 7 --out $D/band_n74c_$tag.json > $D/band_n74c_$tag.log 2>&1
echo BEDS_DONE $tag >> $D/band_n74c_$tag.log
