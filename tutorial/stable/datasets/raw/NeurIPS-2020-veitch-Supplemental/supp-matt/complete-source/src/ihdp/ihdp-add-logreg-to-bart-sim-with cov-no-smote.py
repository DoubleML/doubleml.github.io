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

prepare_data(sim, False, input_df,
             '../out/ihdp_log_reg_bart_sim_no_smote/input_df.csv')

cov_paths = glob.glob('../out/ihdp_bart_sim/covariates/*.csv')

for df_path in cov_paths:
    sub_dir = df_path.split('ihdp_bart_sim')[-1]
    df = pd.read_csv(df_path)
    cov_name = os.path.basename(df_path).split('.csv')[0]
    print(cov_name)
    prepare_data(sim, False, df,
                 f"../out/ihdp_log_reg_bart_sim_no_smote{sub_dir}", covs=[cov_name])
