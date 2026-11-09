cd /home/user/opcg-sim-backend
D=n1_results/rollout_teach; mkdir -p $D/logs $D/shard_results
for b in $(git ls-remote origin "refs/heads/claude/rollout-teach-s*" | awk '{print $2}' | sed 's#refs/heads/##'); do
  k=${b##*-s}
  [ -s $D/rows_s$k.jsonl ] && continue
  for t in 1 2 3 4; do git fetch -q origin $b && break; sleep $((2**t)); done
  git show FETCH_HEAD:$D/rows_s$k.jsonl > $D/rows_s$k.jsonl.tmp && mv $D/rows_s$k.jsonl.tmp $D/rows_s$k.jsonl
  git show FETCH_HEAD:$D/logs/rt_s$k.log > $D/logs/rt_s$k.log 2>/dev/null
  git show FETCH_HEAD:RESULT.json > $D/shard_results/RESULT_s$k.json 2>/dev/null
  git branch -r -d -q origin/$b 2>/dev/null
done
ls $D/rows_s*.jsonl | wc -l
