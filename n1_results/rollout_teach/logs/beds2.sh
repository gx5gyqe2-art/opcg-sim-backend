cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache
D=n1_results/rollout_teach/beds
N="$HOME/nets/nrel_rt_a_e0.npz $HOME/nets/nrel_rt_b_e0.npz $HOME/nets/nrel_rt_b_e1.npz"
python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag rt_b_e1 --nets $HOME/nets/nrel_rt_b_e1.npz > $D/tvval_b_e1.log 2>&1 && mv n1_results/value_teach/tvval_rt_b_e1.json $D/tvval_b_e1.json
python tests/scripts/rollout_teach.py bench --nets $HOME/nets/nrel_rt_b_e1.npz --tag b_e1 --out $D > $D/bench_b_e1.log 2>&1
python tests/scripts/don_jitter_rows.py rows --nets $HOME/nets/nrel_rt_b_e0.npz $HOME/nets/nrel_rt_b_e1.npz > $D/jitter_rows_b.log 2>&1 && mv n1_results/don_jitter/jitter_rows.json $D/jitter_rows_b.json && git checkout -q n1_results/don_jitter/jitter_rows.json
echo FAST_DONE >> $D/bench_b_e1.log
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n70f/n_records/n70f_w*_p[0-3] --nrel $N --holdout-mod 7 --out $D/band_n70f_rt.json > $D/band_n70f_rt.log 2>&1
rm -rf $HOME/packcache/n71f_* $HOME/packcache/n72f_* $HOME/packcache/n73f_*
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel $N --holdout-mod 7 --out $D/band_n74c_rt.json > $D/band_n74c_rt.log 2>&1
echo BEDS2_DONE >> $D/band_n74c_rt.log
