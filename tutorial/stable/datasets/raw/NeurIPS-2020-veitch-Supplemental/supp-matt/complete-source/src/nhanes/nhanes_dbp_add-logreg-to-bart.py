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
    X = df.drop(columns=['trt_dbp', 'ave_dbp']+covs)
    y = df['trt_dbp']
    p = predict_g(X.values, y.values, do_SMOTE)
    if covs != []:
        bart_preds['ghat'] = p
    else:
        bart_preds['g'] = p
    bart_preds.to_csv(output_path, index=False)


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
dbp = pd.read_csv("../data/nhanes-cleaned/hbp_dbp.csv")

cont_vars = ['age_mo', 'hhsize', 'edu', 'income', 'packyr', 'bmi',
             'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol', 'together']
dbp[cont_vars] = preprocessing.scale(dbp[cont_vars])
input_df = pd.read_csv('../out/nhanes_dbp_bart/input_df.csv')
accuracy_score(np.rint(input_df['g']), input_df['t'])
# 0.75

# ----testing accuracy_scores with and without SMOTE--
prepare_data(dbp, False, input_df,
             '../out/nhanes_dbp_log_reg_bart/input_df.csv')
# 0.74
prepare_data(dbp, True, input_df,
             '../out/nhanes_dbp_log_reg_bart/input_df.csv')
# 0.73. therefore no SMOTE

# # ------------testing over---------------------------------
prepare_data(dbp, False, input_df,
             '../out/nhanes_dbp_log_reg_bart/input_df.csv')

cov_paths = glob.glob('../out/nhanes_dbp_bart/covariates/*.csv')


drugs = ['alcohol', 'packyr']
race = ['black', 'hisp', 'white']
wealth = ['income', 'edu', 'insurance']
bloodwork = ['bmi', 'potassium', 'pulse', 'r_sodipota', 'sodium']
social = ['divorced', 'hhsize', 'married', 'widowed', 'together']
socioeconomic = drugs+race+wealth+social
combined_covs = [drugs, race, wealth, bloodwork, social, socioeconomic]
combined_covs_names = ["drugs", "race", "wealth",
                       "bloodwork", "social", "socioeconomic"]

for name, cov_group in zip(combined_covs_names, combined_covs):
    df = pd.read_csv(f'../out/nhanes_dbp_bart/covariates/{name}.csv')
    prepare_data(
        dbp, False, df, f"../out/nhanes_dbp_log_reg_bart/covariates/{name}.csv", covs=cov_group)


for df_path in cov_paths:
    sub_dir = df_path.split('nhanes_dbp_bart')[-1]
    # print(sub_dir)
    df = pd.read_csv(df_path)
    cov_name = os.path.basename(df_path).split('.csv')[0]
    if cov_name not in combined_covs_names:
        prepare_data(dbp, False, df,
                     f"../out/nhanes_dbp_log_reg_bart{sub_dir}", covs=[cov_name])


# ----------without age
os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
dbp = pd.read_csv("../data/nhanes-cleaned/hbp_dbp.csv")
cont_vars = ['age_mo', 'hhsize', 'edu', 'income', 'packyr', 'bmi',
             'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol', 'together']
dbp[cont_vars] = preprocessing.scale(dbp[cont_vars])
dbp_noage = dbp.drop(columns=['age_mo'])
input_df = pd.read_csv('../out/nhanes_dbp_bart_noage/input_df.csv')
prepare_data(dbp_noage, False, input_df,
             '../out/nhanes_dbp_log_reg_bart_noage/input_df.csv')

cov_paths = glob.glob('../out/nhanes_dbp_bart_noage/covariates/*.csv')


for df_path in cov_paths:
    sub_dir = df_path.split('nhanes_dbp_bart_noage')[-1]
    # print(sub_dir)
    df = pd.read_csv(df_path)
    cov_name = os.path.basename(df_path).split('.csv')[0]
    prepare_data(dbp_noage, False, df,
                 f"../out/nhanes_dbp_log_reg_bart_noage{sub_dir}", covs=[cov_name])
