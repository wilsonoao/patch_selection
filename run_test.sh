#!/bin/bash

BASE_DIR="/work/data/4_fold/LUAD"
TEST_BASE_DIR="/work/PAMIL_two_round/result_ensemble"

for mutation_path in "$BASE_DIR"/*; do
    if [ -d "$mutation_path" ]; then
        mutation_name=$(basename "$mutation_path")  # e.g., CSMD3, MUC16

        for csv in "$mutation_path"/dataset_fold_*.csv; do
            fold_name=$(basename "$csv" .csv)  # e.g., dataset_fold_0
            test_dir="$TEST_BASE_DIR/$mutation_name/$fold_name"

            # echo "📁 儲存到: $test_dir"

            python test.py \
                --csv "$csv" \
                --test_dir "$test_dir" \
                --action_size 60
        done
    fi
done
