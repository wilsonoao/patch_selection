#!/bin/bash

FEATURE_BASE_DIR="/work/data"
SAVE_BASE_DIR="/work/result"
MUTATION_PICKLE_DIR="/work/data_pickle/mutataion_pickle/LUAD"

FOUNDATION_MODEL="CHIEF" # CHIEF, GIGAPATH, UNI, VIRCHOW2, CHIEF_WSI, GIGAPATH_WSI
DATA_SOURCE="TCGA"
SLIDE_TYPE="FS"
MAGNIFICATION="20X"
PT_FOLD_NAME="pt_files(stain_norm)"
H5_FOLD_NAME="h5_files(stain_norm)"

CANCER="LUAD"

PT_FILES_PATH="${FEATURE_BASE_DIR}/${DATA_SOURCE}-${CANCER}-${SLIDE_TYPE}/${FOUNDATION_MODEL}/${MAGNIFICATION}/${PT_FOLD_NAME}"
H5_FILES_PATH="${FEATURE_BASE_DIR}/${DATA_SOURCE}-${CANCER}-${SLIDE_TYPE}/${FOUNDATION_MODEL}/${MAGNIFICATION}/${H5_FOLD_NAME}"

for mutation_pickle_file in "${MUTATION_PICKLE_DIR}"/*; do

    if [ -f "$mutation_pickle_file" ]; then
        mutation_name=$(basename "$mutation_pickle_file" .pkl)  # e.g., CSMD3, MUC16

        save_dir="$SAVE_BASE_DIR/${DATA_SOURCE}_${CANCER}_${SLIDE_TYPE}/${FOUNDATION_MODEL}/${MAGNIFICATION}/${mutation_name}"
        csv_dir="${save_dir}"
        cluster="${save_dir}"

        python main.py \
            --config "/work/GroupConstraintMIL/config.yaml" \
            --csv_dir "${csv_dir}" \
            --feature_dir "${PT_FILES_PATH}" \
            --h5_dir "${H5_FILES_PATH}" \
            --clinical_pkl_path "${mutation_pickle_file}" \
            --cluster_pkl_dir "$cluster" \
            --save_dir "$save_dir" \
            --test_dir "$save_dir" \
            --baseline_dir "$save_dir" \

    fi

done