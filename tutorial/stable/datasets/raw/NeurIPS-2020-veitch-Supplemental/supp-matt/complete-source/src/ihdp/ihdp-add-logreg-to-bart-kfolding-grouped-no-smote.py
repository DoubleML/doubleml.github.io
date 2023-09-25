import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def predict_g(X, y):
    # Preprocessed data to be fed!
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        clf = LogisticRegression(random_state=42, solver='liblinear')
        clf.fit(X[train_index], y[train_index])
        predictions[test_index] = clf.predict_proba(X[test_index])[:, 1]
    assert np.isnan(predictions).sum() == 0
    # print(accuracy_score(np.rint(predictions[y == 1]), y[y == 1]))
    return predictions


def prepare_data(df, bart_preds, output_path, covs=[]):
    X = df.drop(columns=['dose400', 'iqsb.36']+covs)
    y = df['dose400']
    p = predict_g(X.values, y.values)
    if covs != []:
        bart_preds['ghat'] = p
        bart_preds = bart_preds[['ghat', 'Qhat', 't', 'y']]
    else:
        bart_preds['g'] = p
        bart_preds = bart_preds[['g', 'Q', 't', 'y']]
    bart_preds.to_csv(output_path, index=False)


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
usek = pd.read_csv("../data/ihdp-cleaned/usek.csv")
input_df = pd.read_csv('../out/ihdp_bart_kfold_grouped/input_df.csv')

continuous_cols = ['bw', 'momage', 'nnhealth', 'cigs', 'alcohol', 'ppvt.imp']
usek[continuous_cols] = preprocessing.scale(usek[continuous_cols])

prepare_data(usek, input_df,
             '../out/ihdp_bart_logreg_kfold_grouped/input_df.csv')

combined_covs = defaultdict(list)
combined_covs['location'] = ["site1", "site2", "site3",
                             "site4", "site5", "site6", "site7", "site8"]
combined_covs['socioeconomic'] = ['mom.lths', "mom.hs", "mom.scoll", "mom.coll",
                                  "b.marry", "livwho", "language", "momblack", "momhisp", "momwhite"]
combined_covs["mom_preg"] = ["birth.o", "parity", "moreprem",
                             "cigs", "alcohol", "mlt.birt", "whenpren", "drugs", "workdur.imp"]
combined_covs["baby"] = ["bw", "nnhealth", "female"]
combined_covs["mom_age"] = ["momage"]
combined_covs["others"] = ["ppvt.imp", "bwg", "othstudy"]
combined_covs["treatment"] = ["dose400"]

for cov_name, cov_group in combined_covs.items():
    df = pd.read_csv(
        f'../out/ihdp_bart_kfold_grouped/covariates/{cov_name}.csv')
    prepare_data(usek, df,
                 f"../out/ihdp_bart_logreg_kfold_grouped/covariates/{cov_name}.csv", covs=cov_group)
