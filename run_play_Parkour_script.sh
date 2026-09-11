#!/bin/bash

# 检查是否提供了参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <log_filename>"
    exit 1
fi

# 获取参数作为日志文件名
log_filename=$1

# 使用 nohup 运行 Python 脚本，并将输出重定向到日志文件
CUDA_VISIBLE_DEVICES=2 
nohup bash -c '
    echo "Process ID: $$"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "Start time: $(date)"
    exec python /data4/dhy/InstinctLab/scripts/instinct_rl/play_show_depth_4.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --num_envs=1 --checkpoint=model_30000.pt --load_run /data4/dhy/InstinctLab/logs/instinct_rl/g1_parkour/20260802_201156 --headless --video --video_length=3000 --show_depth --exportonnx 
    '> "./output_${log_filename}.log" 2>&1 &


# 获取后台运行的进程号
pid=$!

# 将进程号追加到日志文件
echo "Process ID: $pid" >> "./output_${log_filename}.log"

echo "Playing script started with PID: $pid. Check the log file for progress."
