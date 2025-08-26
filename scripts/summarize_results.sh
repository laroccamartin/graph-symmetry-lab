#!/usr/bin/env bash
echo '| run | train_mse | test_mse |'
echo '|---|---:|---:|'
for d in results/*; do
  s="$d/summary.txt"
  if [ -f "$s" ]; then
    trn=$(grep -Eo 'training set: [0-9.]+$' "$s" | awk '{print $3}' | tail -n1)
    tst=$(grep -Eo 'test set: [0-9.]+$' "$s" | awk '{print $3}' | tail -n1)
    echo "| $(basename "$d") | ${trn:-} | ${tst:-} |"
  fi
done
