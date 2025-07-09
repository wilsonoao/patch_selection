import os
import pandas as pd
import numpy as np
from compare_auc_delong_xu import delong_roc_test  # 你已經有這個函數
from sklearn.metrics import roc_auc_score

tumor_name = "TTN"

# 設定四個fold的路徑
fold_root_1 = '/work/PAMIL_two_round/result_ensemble/' + tumor_name
fold_root_2 = '/work/weight/CHIEF_WSI_baseline/' + tumor_name

y_true_all = []
y_pred_1_all = []
y_pred_2_all = []

for fold_id in range(4):
    path_1 = os.path.join(fold_root_1, f'dataset_fold_{fold_id}', 'probability.csv')
    path_2 = os.path.join(fold_root_2, f'dataset_fold_{fold_id}', 'probability.csv')

    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)

    assert np.array_equal(df1['label'].values, df2['label'].values), f"fold {fold_id} label mismatch!"

    y_true_all.append(df1['label'].values)
    y_pred_1_all.append(df1['prob'].values)
    y_pred_2_all.append(df2['prob'].values)

# 合併四個 fold 的資料
y_true = np.concatenate(y_true_all)
y_pred_1 = np.concatenate(y_pred_1_all)
y_pred_2 = np.concatenate(y_pred_2_all)

# 計算拼接後的 AUC
auc_1 = roc_auc_score(y_true, y_pred_1)
auc_2 = roc_auc_score(y_true, y_pred_2)

print(f"📈 拼接後 New method AUC: {auc_1:.6f}")
print(f"📈 拼接後 baseline AUC: {auc_2:.6f}")

# 執行 DeLong test
log10_p = delong_roc_test(y_true, y_pred_1, y_pred_2)
p_value = 10 ** log10_p

print(f"📊 DeLong Test log10(p-value): {log10_p}")
print(f"📊 DeLong Test p-value: {p_value.item():.6f}")

# 顯著性判斷
if p_value < 0.05:
    print("✅ 結論：兩個模型的 AUC 差異具有統計顯著性")
else:
    print("❌ 結論：無法確認兩個模型的 AUC 有顯著差異")
