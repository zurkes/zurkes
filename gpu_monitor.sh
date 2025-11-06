#!/bin/bash
cd ~/stpls3d-lite
LOG_FILE="gpu_usage.log"

echo "=== GPU Monitor Started: $(date) ===" >> $LOG_FILE

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    
    echo "[$TIMESTAMP] MEM:${GPU_MEM}MB | UTIL:${GPU_UTIL}% | TEMP:${GPU_TEMP}°C" >> $LOG_FILE
    
    sleep 5
done