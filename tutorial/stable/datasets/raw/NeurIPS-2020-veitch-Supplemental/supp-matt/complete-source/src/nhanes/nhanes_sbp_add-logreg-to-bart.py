import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob


def predict_g(X, y, do_SMOTE):
    # Preprocessed data to be fed!
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        clf = LogisticRegression(random_state=42, solver='liblinear')
        if do_SMOTE:
            X_balanced, y_balanced = SMOTE().fit_resample(
                X[train_index], y[train_index])
            clf.fit(X_balanced, y_balanced)
        else:
            clf.fit(X[train_index], y[train_index])
        predictions[test_index] = clf.predict_proba(X[test_index])[:, 1]
    assert np.isnan(predictions).sum() == 0
    print(accuracy_score(np.rint(predictions), y))
    return predictions


def prepare_data(df, do_SMOTE, bart_preds, output_path, covs=[]):
    X = df.drop(columns=['trt_sbp', 'ave_sbp']+covs)
    y = df['trt_sbp']
    p = predict_g(X.values, y.values, do_SMOTE)
    if covs != []:
        bart_preds['ghat'] = p
    else:
        bart_preds['g'] = p
    bart_preds.to_csv(output_path, index=False)


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
sbp = pd.read_csv("../data/nhanes-cleaned/hbp_sbp.csv")

cont_vars = ['age_mo', 'hhsize', 'edu', 'income', 'packyr', 'bmi',
             'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol', 'together']
sbp[cont_vars] = preprocessing.scale(sbp[cont_vars])
input_df = pd.read_csv('../out/nhanes_sbp_bart/input_df.csv')
accuracy_score(np.rint(input_df['g']), input_df['t'])
# 0.78

# ----testing accuracy_scores with and without SMOTE--
prepare_data(sbp, False, input_df,
             '../out/nhanes_sbp_log_reg_bart/input_df.csv')
# 0.76
prepare_data(sbp, True, input_df,
             '../out/nhanes_sbp_log_reg_bart/input_df.csv')
# 0.72. therefore no SMOTE

# # ------------testing over---------------------------------
prepare_data(sbp, False, input_df,
             '../out/nhanes_sbp_log_reg_bart/input_df.csv')

cov_paths = glob.glob('../out/nhanes_sbp_bart/covariates/*.csv')


for df_path in cov_paths:
    sub_dir = df_path.split('nhanes_sbp_bart')[-1]
    # print(sub_dir)
    df = pd.read_csv(df_path)
    cov_name = os.path.basename(df_path).split('.csv')[0]
    prepare_data(sbp, False, df,
                 f"../out/nhanes_sbp_log_reg_bart{sub_dir}", covs=[cov_name])
