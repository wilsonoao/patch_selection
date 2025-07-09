#!/bin/bash

BASE_DIR="/work/data/4_fold/LUAD"
SAVE_BASE_DIR="/work/PAMIL_two_round/result_MoE_ensemble"

for mutation_path in "$BASE_DIR"/*; do
    if [ -d "$mutation_path" ]; then
        mutation_name=$(basename "$mutation_path")  # e.g., CSMD3, MUC16

        for csv in "$mutation_path"/dataset_fold_*.csv; do
            fold_name=$(basename "$csv" .csv)  # e.g., dataset_fold_0
            save_dir="$SAVE_BASE_DIR/$mutation_name"

            # echo "▶️ 執行: $csv"
            # echo "📁 儲存到: $save_dir"

            python train.py \
                --csv "$csv" \
                --save_dir "$save_dir" \
                --test_total_T 3 \
                --train_total_T 3 \
                --action_size 60
        done
    fi
done
