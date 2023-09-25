import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
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


def predict_Q(X, y):
    # Preprocessed data to be fed!
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        reg = LinearRegression()
        reg.fit(X[train_index], y[train_index])
        predictions[test_index] = reg.predict(X[test_index])
    assert np.isnan(predictions).sum() == 0
    return predictions


def prepare_data(df, treat_col, response_col, output_path, do_SMOTE, covs=[]):
    g = predict_g(df.drop(
        columns=[treat_col, response_col]+covs).values, df[treat_col].values, do_SMOTE)
    Q = predict_Q(
        df.drop(columns=[response_col]+covs).values, df[response_col].values)
    out_df = pd.DataFrame(
        {'g': g, "Q": Q, 't': df[treat_col], 'y': df[response_col]})
    if covs != []:
        out_df = out_df.rename(columns={'g': 'ghat', 'Q': 'Qhat'})
    out_df.to_csv(output_path, index=False)


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
dbp = pd.read_csv("../data/nhanes-cleaned/hbp_dbp.csv")

cont_vars = ['age_mo', 'hhsize', 'edu', 'income', 'packyr', 'bmi',
             'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol', 'together']
dbp[cont_vars] = preprocessing.scale(dbp[cont_vars])
prepare_data(dbp, 'trt_dbp', 'ave_dbp',
             '../out/nhanes_dbp_linlog/input_df.csv', False)

covs = ['white', 'black', 'hisp', 'female', 'age_mo', 'hhsize', 'edu',
        'married', 'widowed', 'divorced', 'separated', 'income', 'packyr',
        'bmi', 'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol',
        'insurance', 'together']
covs_dict = {n: [n] for n in covs}
covs_dict['treatment'] = ['trt_dbp']
covs_dict['drugs'] = ['alcohol', 'packyr']
covs_dict['race'] = ['black', 'hisp', 'white']
covs_dict['wealth'] = ['income', 'edu', 'insurance']
covs_dict['bloodwork'] = ['bmi', 'potassium', 'pulse', 'r_sodipota', 'sodium']
covs_dict['social'] = ['divorced', 'hhsize', 'married', 'widowed', 'together']
covs_dict['socioeconomic'] = covs_dict['drugs'] + \
    covs_dict['race']+covs_dict['wealth']+covs_dict['social']

for name, cov_group in covs_dict.items():
    prepare_data(
        dbp, 'trt_dbp', 'ave_dbp', f"../out/nhanes_dbp_linlog/covariates/{name}.csv", False, covs=cov_group)

# ------check rmse
linlog = pd.read_csv('../out/nhanes_dbp_linlog/input_df.csv')
bart = pd.read_csv('../out/nhanes_dbp_bart/input_df.csv')

np.sqrt(mean_squared_error(linlog['Q'], linlog['y']))
# 13.0
np.sqrt(mean_squared_error(bart['Q'], bart['y']))
# 11.8
