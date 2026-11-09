#!/usr/bin/env bash
# Correctness check: every parallel configuration must produce a tree that is
# byte-identical to the one from the serial reference implementation.
#
# usage: scripts/validate.sh BUILD_DIR TRAIN.bin [TEST.bin] [MAX_RANKS] [--cuda]
#   MPIRUN can be overridden, e.g. MPIRUN="mpirun --oversubscribe".
set -euo pipefail

BUILD=${1:?build directory}
TRAIN=${2:?training file}
TEST=${3:-}
MAX_RANKS=${4:-4}
CUDA=${5:-}
MPIRUN=${MPIRUN:-mpirun}
DEPTH=${DEPTH:-8}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

common=(--train "$TRAIN" --max-depth "$DEPTH")
[[ -n "$TEST" ]] && common+=(--test "$TEST")

echo "== serial reference"
"$BUILD/dt_serial" "${common[@]}" --dump-tree "$OUT/ref.txt" | tail -n 1

pass=0
fail=0
run_case() {
  local name=$1 ranks=$2 threads=$3
  shift 3
  OMP_NUM_THREADS=$threads $MPIRUN -np "$ranks" "$BUILD/dt_mpi" "${common[@]}" "$@" \
    --dump-tree "$OUT/tree.txt" > "$OUT/log.txt" 2>&1 || { echo "  run failed:"; cat "$OUT/log.txt"; }
  if cmp -s "$OUT/ref.txt" "$OUT/tree.txt"; then
    printf '  [PASS] %-55s %s\n' "$name" "$(tail -n 1 "$OUT/log.txt")"
    pass=$((pass + 1))
  else
    printf '  [FAIL] %-55s\n' "$name"
    fail=$((fail + 1))
  fi
}

echo "== parallel configurations vs. serial"
for ranks in 1 2 3 "$MAX_RANKS"; do
  for threads in 1 2; do
    for omp in feature sample; do
      run_case "data/allreduce     np=$ranks t=$threads omp=$omp" "$ranks" "$threads" --mode data --comm allreduce --omp "$omp"
    done
    run_case "data/reducescatter np=$ranks t=$threads" "$ranks" "$threads" --mode data --comm reducescatter
    run_case "feature            np=$ranks t=$threads" "$ranks" "$threads" --mode feature
  done
  run_case "data/allreduce     np=$ranks no-subtract" "$ranks" 1 --mode data --no-subtract
  if [[ "$CUDA" == "--cuda" ]]; then
    run_case "data/allreduce     np=$ranks cuda" "$ranks" 1 --mode data --backend cuda
    run_case "data/reducescatter np=$ranks cuda" "$ranks" 1 --mode data --comm reducescatter --backend cuda
  fi
done

echo "== $pass passed, $fail failed"
[[ $fail -eq 0 ]]
