#!/bin/bash
# One-shot resume for n28 w08: relaunch the generator from the next unused seed
# band (if it is not already running), (re)start the watchdog, and push WIP.
SRC=/root/opcg-sim-backend
SEED_HI=2288999
TARGET=960

cd "$SRC" || exit 1

read -r DONE MAXSEED < <(python - <<'EOF'
import glob, numpy as np
s = set()
for p in glob.glob('n28_records*/n_record_*.npz'):
    try: s |= set(np.load(p)['seed'].tolist())
    except Exception: pass
print(f"{len(s)} {max(s) if s else 2287999}")
EOF
)
echo "DONE=$DONE MAXSEED=$MAXSEED"

if [ "$DONE" -ge "$TARGET" ]; then echo "TARGET REACHED"; exit 0; fi

if pgrep -f "^python tests/scripts/n_record_gen\.py" >/dev/null; then
  echo "generator already running"
else
  REMAIN=$((TARGET - DONE))
  NEXT=0
  for K in 8 2 1; do
    C=$((MAXSEED + K))
    [ $((C + REMAIN - 1)) -le "$SEED_HI" ] && { NEXT=$C; break; }
  done
  if [ "$NEXT" -eq 0 ]; then
    NEXT=$((MAXSEED + 1)); REMAIN=$((SEED_HI - NEXT + 1))
    echo "band tight: remain clamped to $REMAIN"
  fi
  N=2; while [ -e "$SRC/n28_records_r$N" ]; do N=$((N + 1)); done
  OPCG_LOG_SILENT=1 PYTHONPATH=tests nohup python tests/scripts/n_record_gen.py \
    --neff-net /root/gen_net.npz --games "$REMAIN" --seed-base "$NEXT" --sims 128 \
    --workers 4 --shard-games 10 --dump-v2 --out "n28_records_r$N" \
    > "/root/n28_gen_r$N.log" 2>&1 &
  echo "launched r$N: seed_base=$NEXT games=$REMAIN"
fi

pgrep -f "bash /root/watchdog.sh" >/dev/null || {
  nohup setsid /root/watchdog.sh >> /root/watchdog.out 2>&1 < /dev/null &
  echo "watchdog started"
}
