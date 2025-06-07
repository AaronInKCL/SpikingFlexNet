#!/bin/bash

# 最多同时运行的任务数
MAX_PARALLEL=3

# 当前运行中的任务数
running=0

for i in {4..6}
do
    echo "Starting model $i..."
    nohup python src/train_initialiser.py --config configs/$i.json > outputs/output$i.log 2>&1 &
    
    # 运行计数加1
    ((running++))

    # 如果达到了最大并行数，就等待所有后台任务完成
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait
        running=0
    fi
done

# 等待最后一批任务结束（如果总数不是MAX_PARALLEL的倍数）
wait

echo "All training jobs completed."
