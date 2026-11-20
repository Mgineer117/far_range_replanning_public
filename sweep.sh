#!/bin/bash
mkdir -p logs
for s in 1 0.5 0.25; do
  for r in 0 1 2 3 4; do
    echo "scale $s repeat $r started $(date)"
    python minduli_compare.py --noise-scale $s --repeat $r > logs/s${s}_r${r}.log 2>&1
  done
done
echo "sweep finished $(date)"