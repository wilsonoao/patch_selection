#!/bin/bash

BASE_DIR="/work/data/4_fold/LUAD"
TEST_BASE_DIR="/work/SingleAgent_noiseLearning_twoModel/uni_baseline_LUAD"

for mutation_path in "$BASE_DIR"/*; do
    if [ -d "$mutation_path" ]; then
        mutation_name=$(basename "$mutation_path")  # e.g., CSMD3, MUC16

        # if [ "$mutation_name" = "CSMD3" ]; then
        #         echo "Skipping $mutation_name"
        #         continue
        #     fi

        for csv in "$mutation_path"/dataset_fold_*.csv; do
            fold_name=$(basename "$csv" .csv)  # e.g., dataset_fold_0
            test_dir="$TEST_BASE_DIR/$mutation_name/$fold_name"

            # if [ "$fold_name" = "dataset_fold_0" ]; then
            #     echo "Skipping $mutation_name"
            #     continue
            # fi

            # if [ "$fold_name" = "dataset_fold_1" ]; then
            #     echo "Skipping $mutation_name"
            #     continue
            # fi

            # echo "▶️ 執行: $csv"
            # echo "📁 儲存到: $save_dir"

            python test_baseline.py \
                --csv "$csv" \
                --test_dir "$test_dir" \
                --action_size 60 \
                --seed 42 \
                --num_epoch 100 \
                --chief_feature_dir "/work/data/TCGA-LUAD-FS/UNI/20X/pt_files(stain_norm)" 
                
        done
    fi
done
