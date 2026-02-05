import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import combine_pvalues
from utilmodule.compare_auc_delong_xu import delong_roc_test

def significance_test(
    new_method_root: str,
    baseline_root: str,
    out_dir: str,
    n_folds: int = 4
):
    """
    Compare AUC between new method and baseline using 4-fold results.

    Assumes:
      {root}/dataset_fold_{i}/probability.csv

    CSV must contain:
      - label
      - prob
    """

    fold_auc_new = []
    fold_auc_base = []
    fold_pvalues = []

    y_true_all = []
    y_pred_new_all = []
    y_pred_base_all = []

    for fold_id in range(n_folds):
        path_new = os.path.join(
            new_method_root, f"dataset_fold_{fold_id}", "probability.csv"
        )
        path_base = os.path.join(
            baseline_root, f"dataset_fold_{fold_id}", "probability.csv"
        )

        df_new = pd.read_csv(path_new)
        df_base = pd.read_csv(path_base)

        assert np.array_equal(
            df_new["label"].values, df_base["label"].values
        ), f"Fold {fold_id} label mismatch!"

        y_true = df_new["label"].values
        y_new = df_new["prob"].values
        y_base = df_base["prob"].values

        auc_new = roc_auc_score(y_true, y_new)
        auc_base = roc_auc_score(y_true, y_base)

        fold_auc_new.append(auc_new)
        fold_auc_base.append(auc_base)

        log10_p = delong_roc_test(y_true, y_new, y_base)
        p_value = float(10 ** log10_p)

        fold_pvalues.append(p_value)

        # print(
        #     f"  Fold {fold_id} | "
        #     f"New AUC: {auc_new:.4f} | "
        #     f"Baseline AUC: {auc_base:.4f} | "
        #     f"p-value: {p_value:.4f}"
        # )

        y_true_all.append(y_true)
        y_pred_new_all.append(y_new)
        y_pred_base_all.append(y_base)
    
    mean_auc_new = float(np.mean(fold_auc_new))
    mean_auc_base = float(np.mean(fold_auc_base))

    fold_auc_new.append(mean_auc_new)
    fold_auc_base.append(mean_auc_base)
    

    # ===== Concatenated evaluation =====
    y_true_cat = np.concatenate(y_true_all)
    y_new_cat = np.concatenate(y_pred_new_all)
    y_base_cat = np.concatenate(y_pred_base_all)

    auc_new_cat = roc_auc_score(y_true_cat, y_new_cat)
    auc_base_cat = roc_auc_score(y_true_cat, y_base_cat)

    log10_p = delong_roc_test(y_true_cat, y_new_cat, y_base_cat)
    p_cat = float(10 ** log10_p)

    # ===== Combine fold-wise p-values =====
    methods = ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']
    combined_pvals = {
        method: combine_pvalues(fold_pvalues, method=method)[1]
        for method in methods
    }

    fold_pvalues.append(-1)

    # ===== Save CSVs =====
    os.makedirs(out_dir, exist_ok=True)

    # (1) fold-level CSV
    fold_df = pd.DataFrame({
        "fold": list(range(n_folds)) + ["average"],
        "auc_new": fold_auc_new,
        "auc_baseline": fold_auc_base,
        "p_delong": fold_pvalues
    })
    fold_df.to_csv(
        os.path.join(out_dir, "fold_results.csv"),
        index=False
    )

    # (2) summary CSV
    summary_data = {
        "auc_new_concat": auc_new_cat,
        "auc_baseline_concat": auc_base_cat,
        "p_delong_concat": p_cat,
        **combined_pvals
    }
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(
        os.path.join(out_dir, f"summary_pValue_results.csv"),
        index=False
    )

    # print(f"\n💾 Results saved to: {out_dir}")
