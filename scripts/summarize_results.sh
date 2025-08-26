#!/usr/bin/env bash
echo '| run | mse |'
echo '|---|---|'
for d in results/*; do
  if [ -f "$d/summary.txt" ]; then
    mse=$(sed -E 's/.*: ([0-9.]+)/\1/' "$d/summary.txt")
    echo "| $(basename "$d") | $mse |"
  fi
done
