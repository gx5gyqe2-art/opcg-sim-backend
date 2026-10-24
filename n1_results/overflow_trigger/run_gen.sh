#!/bin/sh
# 決着せずの数（§7.44）: 教材の生成と同じ設定（gen_n51.json の引数・生成役 r22・デッキ real）で 400 局・seed 3461000〜3461399。
# 100 局 × 4 部（部ごとに meta が出る＝途中で止まっても部単位で再開できる）。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
D=n1_results/overflow_trigger/gen
mkdir -p $D
for j in 0 1 2 3; do
  seed=$((3461000 + 100 * j))
  out=$D/part$j
  [ -f $out/meta_n_record.json ] && continue
  rm -rf $out
  python -m opcg_sim.loop.record_gen --games 100 --seed-base $seed --workers 4 \
    --sims 1280 --worlds 1 --budget 0 --select-rule q_min_n --q-min-frac 0.125 --root-prior-temp 2.0 \
    --eps-play 0.03 --eps-hold 0.03 --decks real --shard-games 10 \
    --net opcg_sim/data/learned/nrel_r22.npz --out $out >> n1_results/overflow_trigger/gen.log 2>&1
  echo "PART $j DONE $(date -u +%H:%M)" >> n1_results/overflow_trigger/progress.log
done
echo GEN_DONE >> n1_results/overflow_trigger/progress.log
