#!/bin/bash
# Run all the agentic-expansion tests that work locally (CPU/MPS, no GPU, no pytest).
#   bash runs/test_agentic.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

tests=(
  tests/test_rl_core.py
  tests/test_execution_session.py
  tests/test_tools.py
  tests/test_coding_env.py
  tests/test_engine_tools.py
  tests/test_posttrain_reward.py
)

fail=0
for t in "${tests[@]}"; do
  echo "=== $t ==="
  PYTHONPATH=. "$PY" "$t" || fail=1
  echo
done
[ $fail -eq 0 ] && echo "ALL AGENTIC TESTS PASSED" || { echo "SOME TESTS FAILED"; exit 1; }
