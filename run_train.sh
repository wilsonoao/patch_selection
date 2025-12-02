#!/bin/bash

BASE_DIR="/work/data/4_fold/LUAD"
SAVE_BASE_DIR="/work/NoisetwoModel_group_rewardCalibration_MCDCP/result_SLC_LUAD"

for mutation_path in "$BASE_DIR"/*; do
    if [ -d "$mutation_path" ]; then
        mutation_name=$(basename "$mutation_path")  # e.g., CSMD3, MUC16

        if [ "$mutation_name" = "CSMD3" ]; then
            echo "Skipping $mutation_name"
            continue
        fi

        if [ "$mutation_name" = "MUC16" ]; then
            echo "Skipping $mutation_name"
            continue
        fi

        if [ "$mutation_name" = "RYR2" ]; then
            echo "Skipping $mutation_name"
            continue
        fi

        for csv in "$mutation_path"/dataset_fold_*.csv; do
            fold_name=$(basename "$csv" .csv)  # e.g., dataset_fold_0
            save_dir="$SAVE_BASE_DIR/$mutation_name"

            # if [ "$fold_name" = "dataset_fold_0" ]; then
            #     echo "Skipping $fold_name"
            #     continue
            # fi

            # echo "▶️ 執行: $csv"
            # echo "📁 儲存到: $save_dir"

            python train.py \
                --csv "$csv" \
                --save_dir "$save_dir" \
                --action_size 1 \
                --seed 42 \
                --num_epoch 200 \
                --patience 10 \
                --state_dim 770
                # --chief_feature_dir "/work/data/TCGA-LUAD-FS/UNI/20X/pt_files(stain_norm)" \
                # --state_dim 1024
                # --gigapath_feature_dir "/work/data/TCGA-BRCA-FS/GIGAPATH/20X/pt_files(stain_norm)" \
                # --train_h5 "/work/data/TCGA-BRCA-FS/CHIEF/20X/h5_files(stain_norm)"
        done
    fi
done
