#!/usr/bin/env bash

set -u

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <log_filename>"
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$script_dir"
python_bin="${ONLYSTAIRS_PYTHON_BIN:-python}"
gpu_id="${ONLYSTAIRS_GPU_ID:-1}"
log_filename="$1"
console_log="${project_dir}/output_onlystairs_${log_filename}.log"
training_log_root="${project_dir}/logs/instinct_rl/g1_parkour_onlystairs"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin"
    echo "Set ONLYSTAIRS_PYTHON_BIN to the Python executable for the project_instinct environment."
    exit 1
fi

mkdir -p "$training_log_root"

{
    echo "Task: Instinct-Parkour-Only-Stairs-G1-v0"
    echo "CUDA_VISIBLE_DEVICES: $gpu_id"
    echo "Training log root: $training_log_root"
    echo "Start time: $(date)"
} > "$console_log"

CUDA_VISIBLE_DEVICES="$gpu_id" nohup "$python_bin" \
    "${project_dir}/scripts/instinct_rl/train.py" \
    --task=Instinct-Parkour-Only-Stairs-G1-v0 \
    --headless \
    --logroot="$training_log_root" \
    >> "$console_log" 2>&1 &

pid=$!
echo "Process ID: $pid" >> "$console_log"
echo "OnlyStairs training started with PID: $pid"
echo "Console log: $console_log"
echo "Training logs: $training_log_root"
