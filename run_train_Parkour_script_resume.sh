#!/bin/bash

# 检查是否提供了参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <log_filename>"
    exit 1
fi

# 获取参数作为日志文件名
log_filename=$1

# 使用 nohup 运行 Python 脚本，并将输出重定向到日志文件
CUDA_VISIBLE_DEVICES=1 nohup bash -c '
    echo "Process ID: $$"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "Start time: $(date)" 
    exec python scripts/instinct_rl/train.py \
        --task=Instinct-Parkour-Target-Amp-G1-v0 \
        --headless \
        --logroot=/data3/dhy/Train_and_Play/InstinctLab/logs/instinct_rl/g1_parkour \
        --resume \
        --load_run=/data3/dhy/Train_and_Play/InstinctLab/logs/instinct_rl/g1_parkour/20260907_210520 \
        --checkpoint=model_25000.pt 
    '> "./output_${log_filename}.log" 2>&1 &

# 获取后台运行的进程号
pid=$!

# 将进程号追加到日志文件
echo "Process ID: $pid" >> "./output_${log_filename}.log"

echo "Training script started with PID: $pid. Check the log file for progress."