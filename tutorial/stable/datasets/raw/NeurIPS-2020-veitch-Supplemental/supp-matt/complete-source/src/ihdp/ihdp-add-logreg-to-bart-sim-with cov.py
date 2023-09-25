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
    # print(accuracy_score(np.rint(predictions[y == 1]), y[y == 1]))
    return predictions


def prepare_data(df, do_SMOTE, bart_preds, output_path, covs=[]):
    X = df.drop(columns=['treatment', 'y_factual']+covs)
    y = df['treatment']
    p = predict_g(X.values, y.values, do_SMOTE)
    if covs != []:
        bart_preds['ghat'] = p
    else:
        bart_preds['g'] = p
    bart_preds.to_csv(output_path, index=False)


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
sim = pd.read_csv("../data/ihdp-cleaned/ihdp_npci_1_cleaned.csv")
sim.head()
input_df = pd.read_csv('../out/ihdp_bart_sim/input_df.csv')
# accuracy_score(np.rint(
#     input_df.loc[input_df['t'] == 1, 'g']), input_df.loc[input_df['t'] == 1, 't'])

# # 0.0144

# # ----testing accuracy_scores with and without SMOTE--
# prepare_data(sim, False, input_df,
#              '../out/ihdp_log_reg_bart_sim/input_df.csv')
# # 0.028
# prepare_data(sim, True, input_df,
#              '../out/ihdp_log_reg_bart_sim/input_df.csv')
# # 0.59. therefore use SMOTE

# # ------------testing over---------------------------------
prepare_data(sim, True, input_df,
             '../out/ihdp_log_reg_bart_sim/input_df.csv')

cov_paths = glob.glob('../out/ihdp_bart_sim/covariates/*.csv')

loc = ["ark", "ein", "har", "mia", "pen", "tex", "was"]
mom_ed = ["mom.lths", "mom.hs", "mom.scoll"]
habits = ["booze", "drugs", "cig"]
baby_health = ["bw", "b.head", "preterm", "birth.o", "nnhealth"]
combined_covs = [loc, mom_ed, habits, baby_health]
combined_covs_names = ["loc", "mom_ed", "habits", "baby_health"]

for df_path in cov_paths:
    sub_dir = df_path.split('ihdp_bart_sim')[-1]
    df = pd.read_csv(df_path)
    cov_name = os.path.basename(df_path).split('.csv')[0]
    if cov_name not in combined_covs_names:
        prepare_data(sim, True, df,
                     f"../out/ihdp_log_reg_bart_sim{sub_dir}", covs=[cov_name])


for name, cov_group in zip(combined_covs_names, combined_covs):
    df = pd.read_csv(f'../out/ihdp_bart_sim/covariates/{name}.csv')
    print(name, cov_group, df.head())
    prepare_data(
        sim, True, df, f"../out/ihdp_log_reg_bart_sim/covariates/{name}.csv", covs=cov_group)
