#!/bin/bash
# n20-22 w03: 3チャンク直列実行＋30分毎 commit/push。sleep/tail/git/ps/kill/lsのみ使用。
cd /home/user/opcg-sim-backend || exit 1

BRANCH=claude/n20-w03
NET=/tmp/gen_net.npz

push_progress() {
  local msg="$1"
  for d in n20_records n21_records n22_records; do
    [ -d "$d" ] && git add -f "$d"
  done
  for f in n20_gen.log n21_gen.log n22_gen.log n20_w03_orchestrate.log; do
    [ -f "$f" ] && git add -f "$f"
  done
  if ! git diff --cached --quiet; then
    git commit -m "$msg"
    for i in 1 2 3 4; do
      git push -u origin "$BRANCH" && break
      sleep $((2 ** i))
    done
  fi
}

run_chunk() {
  local seed=$1 out=$2 log=$3
  if [ -f "$log" ] && grep -q N_RECORD_DONE "$log"; then
    echo "[$(date -u +%FT%TZ)] chunk $out already done, skip"
    return 0
  fi
  while true; do
    if ! pgrep -f "n_record_gen.py --games 960 --seed-base $seed " >/dev/null; then
      echo "[$(date -u +%FT%TZ)] launching seed=$seed out=$out"
      OPCG_LOG_SILENT=1 nohup python3 tests/scripts/n_record_gen.py \
        --games 960 --seed-base "$seed" --workers 4 --sims 128 --shard-games 10 \
        --neff-net "$NET" --out "$out" > "$log" 2>&1 &
    fi
    sleep 600
    echo "[$(date -u +%FT%TZ)] --- $log tail ---"
    tail -3 "$log"
    push_progress "n20-22 w03 progress"
    if grep -q N_RECORD_DONE "$log"; then
      echo "[$(date -u +%FT%TZ)] chunk $out DONE"
      push_progress "n20-22 w03: chunk $out complete"
      break
    fi
    if ! pgrep -f "n_record_gen.py --games 960 --seed-base $seed " >/dev/null; then
      echo "[$(date -u +%FT%TZ)] process died without N_RECORD_DONE, restarting same chunk from scratch"
    fi
  done
}

run_chunk 2203000 n20_records n20_gen.log
run_chunk 2213000 n21_records n21_gen.log
run_chunk 2223000 n22_records n22_gen.log

push_progress "n20-22 w03: final"
echo "ALL_CHUNKS_DONE"
grep N_RECORD_DONE n20_gen.log n21_gen.log n22_gen.log
