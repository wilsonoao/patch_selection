#!/bin/bash

BASE_DIR="/work/data/4_fold/LUAD"
TEST_BASE_DIR="/work/NoisetwoModel_group_rewardCalibration_MCDCP/result_SLC_LUAD"

for mutation_path in "$BASE_DIR"/TP53; do
    if [ -d "$mutation_path" ]; then
        mutation_name=$(basename "$mutation_path")  # e.g., CSMD3, MUC16

        for csv in "$mutation_path"/dataset_fold_*.csv; do
            fold_name=$(basename "$csv" .csv)  # e.g., dataset_fold_0
            test_dir="$TEST_BASE_DIR/$mutation_name/$fold_name"

            # echo "📁 儲存到: $test_dir"

            python test.py \
                --csv "$csv" \
                --test_dir "$test_dir" \
                --action_size 1 \
                --seed 42 \
                --num_epoch 100 \
                --state_dim 770
                # --chief_feature_dir "/work/data/TCGA-LUAD-FS/UNI/20X/pt_files(stain_norm)" 
                # --state_dim 1024
                # --chief_feature_dir "/work/data/TCGA-BRCA-FS/CHIEF/20X/pt_files(stain_norm)" \
                # --gigapath_feature_dir "/work/data/TCGA-BRCA-FS/GIGAPATH/20X/pt_files(stain_norm)" \
                # --train_h5 "/work/data/TCGA-BRCA-FS/CHIEF/20X/h5_files(stain_norm)"
        done
    fi
done
