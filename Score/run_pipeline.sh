#!/usr/bin/env bash
# Run the generation pipeline from the Score directory so imports resolve.
# Usage (from repo root or Score):
#   ./Score/run_pipeline.sh [args...]
#   cd Score && ./run_pipeline.sh --model gpt2 --dataset coqa --num_generations_per_prompt 2 --fraction_of_data_to_use 0.01

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
exec python pipeline/generate_simple.py "$@"
