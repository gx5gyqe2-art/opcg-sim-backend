#!/bin/bash
cd /home/user/opcg-sim-backend
for k in $(seq -w 1 16); do
  f=~/pdl/shards/pairs_s$k.jsonl.gz; [ -s $f ] && continue
  git ls-remote --exit-code -q origin refs/heads/claude/pair-diff-loss-s$k >/dev/null 2>&1 || continue
  git fetch -q origin claude/pair-diff-loss-s$k || continue
  git show FETCH_HEAD:n1_results/pair_diff_loss/pairs_s$k.jsonl.gz > $f.tmp && mv $f.tmp $f
  git show FETCH_HEAD:RESULT.json > ~/pdl/shards/RESULT_$k.json
  git show FETCH_HEAD:n1_results/pair_diff_loss/logs/pd_s$k.log > ~/pdl/shards/pd_s$k.log
done
ls ~/pdl/shards/*.gz | wc -l
