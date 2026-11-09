cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1
PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend python -m opcg_sim.learned.train.n_rel_band --in $HOME/n73f/n_records/n73f_w*_p[0-3] \
  --nrel opcg_sim/data/learned/nrel_r28x4.npz $HOME/nrel_djc.ep0.npz $HOME/nrel_djc.ep1.npz $HOME/nrel_djt.ep0.npz $HOME/nrel_djt.ep1.npz \
  --policy --holdout-mod 7 --out n1_results/don_jitter/band_n73f.json > n1_results/don_jitter/band_n73f.log 2>&1
echo BAND_EXIT $? >> n1_results/don_jitter/band_n73f.log
python tests/scripts/value_teach.py rocks --out n1_results/don_jitter --tag dj \
  --nets $HOME/nrel_djc.ep0.npz $HOME/nrel_djc.ep1.npz $HOME/nrel_djt.ep0.npz $HOME/nrel_djt.ep1.npz > n1_results/don_jitter/rocks_dj.log 2>&1
echo ROCKS_EXIT $? >> n1_results/don_jitter/rocks_dj.log
