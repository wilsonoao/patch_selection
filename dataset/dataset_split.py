import os
import pickle
import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def build_fold_csv(
    feature_dir,
    clinical_pkl_path,
    out_dir,
    n_splits=4,
    exclude_substring="DX",
    random_state=42,
):
    """
    Build case-level stratified 4-fold split and export path-level CSVs.

    Output:
        out_dir/gene_name/dataset_fold_{k}.csv
    """

    # ======================
    # 0. 防呆檢查
    # ======================
    # print(feature_dir)
    if not os.path.isdir(feature_dir):
        raise NotADirectoryError(f"feature_dir 不存在或不是資料夾: {feature_dir}")

    if not os.path.isfile(clinical_pkl_path):
        raise FileNotFoundError(f"clinical_pkl 不存在: {clinical_pkl_path}")

    os.makedirs(out_dir, exist_ok=True)

    # ======================
    # 1. 讀取資料
    # ======================
    feature_path_list = os.listdir(feature_dir)

    with open(clinical_pkl_path, "rb") as f:
        clinical = pickle.load(f)

    # ======================
    # 2. case → paths mapping
    # ======================
    case_to_paths = defaultdict(list)

    for _, row in clinical.iterrows():
        case_id = row["case_submitter_id"]
        pattern = rf"(?=.*{case_id})(?!.*{exclude_substring})"
        matched = [p for p in feature_path_list if re.search(pattern, p)]
        case_to_paths[case_id].extend(matched)

    # ======================
    # 3. case-level label table
    # ======================
    case_df = clinical[["case_submitter_id", "mutation"]].drop_duplicates()
    case_ids = case_df["case_submitter_id"].values
    case_labels = case_df["mutation"].values

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    case_df["base_fold"] = -1
    for fold_id, (_, idx) in enumerate(skf.split(case_ids, case_labels)):
        case_df.loc[idx, "base_fold"] = fold_id

    # ======================
    # 4. Fold 規則
    # ======================
    FOLD_RULES = {
        0: {"train": [0, 1], "val": 2, "test": 3},
        1: {"train": [1, 2], "val": 3, "test": 0},
        2: {"train": [2, 3], "val": 0, "test": 1},
        3: {"train": [3, 0], "val": 1, "test": 2},
    }

    def get_cases(fold_ids):
        return case_df[case_df.base_fold.isin(fold_ids)].case_submitter_id

    def collect_paths(cases):
        data = []
        for cid in cases:
            label = int(
                case_df.loc[
                    case_df.case_submitter_id == cid, "mutation"
                ].values[0]
            )
            paths = case_to_paths.get(cid, [])

            # ❗ 可選：跳過沒有 patch 的 case
            if len(paths) == 0:
                continue

            for p in paths:
                data.append((p, label))
        return data

    def format_split_data(data, split_name):
        rows = []
        for path, label in data:
            row = {
                "train": "",
                "train_label": "",
                "val": "",
                "val_label": "",
                "test": "",
                "test_label": "",
            }
            row[split_name] = path
            row[f"{split_name}_label"] = label
            rows.append(row)
        return rows

    # ======================
    # 5. Export each fold
    # ======================
    for fold_idx, rule in FOLD_RULES.items():
        train_cases = get_cases(rule["train"])
        val_cases = get_cases([rule["val"]])
        test_cases = get_cases([rule["test"]])

        train_data = collect_paths(train_cases)
        val_data = collect_paths(val_cases)
        test_data = collect_paths(test_cases)

        all_rows = (
            format_split_data(train_data, "train")
            + format_split_data(val_data, "val")
            + format_split_data(test_data, "test")
        )

        df = pd.DataFrame(all_rows)

        out_path = os.path.join(
            out_dir, f"dataset_fold_{fold_idx}.csv"
        )
        df.to_csv(out_path, index=False)

        print(f"[Fold {fold_idx}] saved -> {out_path}")

    print("✅ case-level 4-fold CSV 建立完成")