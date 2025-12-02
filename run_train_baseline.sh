#!/bin/bash

BASE_DIR="/work/data/4_fold/LUAD"
SAVE_BASE_DIR="/work/SingleAgent_noiseLearning_twoModel/chief_baseline_LUAD_KMEANs40"

for mutation_path in "$BASE_DIR"/TP53; do
    if [ -d "$mutation_path" ]; then
        mutation_name=$(basename "$mutation_path")  # e.g., CSMD3, MUC16

        # if [ "$mutation_name" = "CSMD3" ]; then
        #         echo "Skipping $mutation_name"
        #         continue
        #     fi

        for csv in "$mutation_path"/dataset_fold_*.csv; do
            fold_name=$(basename "$csv" .csv)  # e.g., dataset_fold_0
            save_dir="$SAVE_BASE_DIR/$mutation_name"

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

            python train_baseline.py \
                --csv "$csv" \
                --save_dir "$save_dir" \
                --action_size 60 \
                --seed 42 \
                --num_epoch 100 
                # --chief_feature_dir "/work/data/TCGA-LUAD-FS/UNI/20X/pt_files(stain_norm)"
                
        done
    fi
done
