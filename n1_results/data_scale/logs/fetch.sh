#!/bin/bash
# usage: fetch.sh wave nshards
w=$1; n=$2
for i in $(seq -w 1 $n); do
  i2=$(printf %02d $((10#$i)))
  [ -d ~/n$w/n_records/n${w}_w${i2}_p0 ] && continue
  rm -rf ~/fr; mkdir -p ~/fr; cd ~/fr; git init -q .; git remote add origin https://github.com/gx5gyqe2-art/opcg-sim-backend
  for t in 1 2 3 4 5; do git fetch -q --depth 1 origin claude/n$w-w$i2 && break; sleep $((t*3)); done
  mkdir -p ~/n$w && git archive FETCH_HEAD n_records | tar -x -C ~/n$w || echo "FAIL $w $i2"
done
rm -rf ~/fr
echo "WAVE $w $(ls -d ~/n$w/n_records/n${w}_w*_p* | wc -l)"
