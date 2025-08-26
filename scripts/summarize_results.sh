#!/usr/bin/env bash
echo '| run | train_mse | test_mse |'
echo '|---|---:|---:|'
for d in results/*; do
  s="$d/summary.txt"
  if [ -f "$s" ]; then
    trn=$(awk '/Final MSE on training set:/ {print $NF}' "$s" | tail -n1)
    tst=$(awk '/Final MSE on test set:/     {print $NF}' "$s" | tail -n1)
    echo "| $(basename "$d") | ${trn:-} | ${tst:-} |"
  fi
done
