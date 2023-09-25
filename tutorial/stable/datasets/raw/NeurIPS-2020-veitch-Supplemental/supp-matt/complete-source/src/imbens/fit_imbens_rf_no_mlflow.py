import pathlib
import numpy as np
import pandas as pd
import scipy as sp
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold


def predict_g(X, y, do_SMOTE):
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        clf = RandomForestClassifier(
            random_state=42, n_estimators=N_EST, max_depth=MAX_DEPTH)
        if do_SMOTE:
            X_balanced, y_balanced = SMOTE().fit_resample(
                X[train_index], y[train_index])
            clf.fit(X_balanced, y_balanced)
        else:
            clf.fit(X[train_index], y[train_index])
        predictions[test_index] = clf.predict_proba(X[test_index])[:, 1]
    assert np.isnan(predictions).sum() == 0
    return predictions


def predict_Q(X, y):
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        reg = RandomForestRegressor(
            random_state=42, n_estimators=N_EST, max_depth=MAX_DEPTH)
        reg.fit(X[train_index], y[train_index])
        predictions[test_index] = reg.predict(X[test_index])
    assert np.isnan(predictions).sum() == 0
    return predictions


def fit_imbens_df(df, response_col, treatment_col, do_SMOTE, output_path, covs=[]):
    X_g = df.drop([response_col, treatment_col]+covs, axis=1).values
    y_g = df[treatment_col].values
    X_Q = df.drop([response_col]+covs, axis=1).values
    y_Q = df[response_col].values
    g = predict_g(X_g, y_g, do_SMOTE)
    Q = predict_Q(X_Q, y_Q)
    if covs == []:
        out_df = pd.DataFrame({'g': g, 'Q': Q, 't': y_g, 'y': y_Q})
    else:
        out_df = pd.DataFrame({'ghat': g, 'Qhat': Q, 't': y_g, 'y': y_Q})
    out_df.to_csv(output_path, index=False)
    return None


def get_ate(df, response_col, treatment_col):
    X_Q = df.drop([response_col], axis=1).values
    y_Q = df[response_col].values
    df_t0 = df.copy(deep=True)
    df_t1 = df.copy(deep=True)
    df_t0[treatment_col] = 0
    df_t1[treatment_col] = 1
    X_Q_t0 = df_t0.drop([response_col], axis=1).values
    X_Q_t1 = df_t1.drop([response_col], axis=1).values
    preds_t0 = np.full_like(y_Q, np.nan, dtype=float)
    preds_t1 = np.full_like(y_Q, np.nan, dtype=float)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X_Q):
        reg = RandomForestRegressor(
            random_state=42, n_estimators=N_EST, max_depth=MAX_DEPTH)
        reg.fit(X_Q[train_index], y_Q[train_index])
        preds_t0[test_index] = reg.predict(X_Q_t0[test_index])
        preds_t1[test_index] = reg.predict(X_Q_t1[test_index])
    assert(np.isnan(preds_t0).sum() == 0)
    assert(np.isnan(preds_t1).sum() == 0)
    ate = np.mean(preds_t1-preds_t0)
    sd = np.std(y_Q)
    return ate, sd


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")

imbens1 = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens1.csv'))
imbens2 = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens2.csv'))
imbens3 = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens3.csv'))
imbens4 = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens4.csv'))


covariates_to_drop = {x: [x] for x in ['age', 'education', 'black', 'hispanic',
                                       'married', 'RE74', 'RE75', 'pos74', 'pos75', 'treatment']}
covariates_to_drop['recent_earnings'] = ['RE75', 'pos75']
covariates_to_drop['pre_program_earnings'] = ['RE74', 'RE75', 'pos74', 'pos75']


N_EST = 100
MAX_DEPTH = 5

for df, name in zip([imbens1, imbens2, imbens4], ['imbens1', 'imbens2', 'imbens4']):
    fit_imbens_df(df, 'RE78', 'treatment', True,
                  f'../out/{name}_rf/input_df.csv')
    for cov_name, cov_val in covariates_to_drop.items():
        fit_imbens_df(df, 'RE78', 'treatment', True,
                      f'../out/{name}_rf/covariates/{cov_name}.csv', covs=cov_val)


# imbens 3!
fit_imbens_df(imbens3, 'change_earnings', 'treatment',
              True, f'../out/imbens3_rf/input_df.csv')
for cov_name, cov_val in covariates_to_drop.items():
    fit_imbens_df(imbens3, 'change_earnings', 'treatment', True,
                  f'../out/imbens3_rf/covariates/{cov_name}.csv', covs=cov_val)

# ------no_smote
for df, name in zip([imbens1, imbens2, imbens4], ['imbens1', 'imbens2', 'imbens4']):
    fit_imbens_df(df, 'RE78', 'treatment', False,
                  f'../out/{name}_rf_no_smote/input_df.csv')
    for cov_name, cov_val in covariates_to_drop.items():
        fit_imbens_df(df, 'RE78', 'treatment', False,
                      f'../out/{name}_rf_no_smote/covariates/{cov_name}.csv', covs=cov_val)


# imbens 3!
fit_imbens_df(imbens3, 'change_earnings', 'treatment',
              False, f'../out/imbens3_rf_no_smote/input_df.csv')
for cov_name, cov_val in covariates_to_drop.items():
    fit_imbens_df(imbens3, 'change_earnings', 'treatment', False,
                  f'../out/imbens3_rf_no_smote/covariates/{cov_name}.csv', covs=cov_val)

# ----imbens4 minus--education--no--smote
imbens4_minus_edu = imbens4.drop(columns=['education'])

covariates_to_drop = {x: [x] for x in ['age', 'black', 'hispanic',
                                       'married', 'RE74', 'RE75', 'pos74', 'pos75', 'treatment']}
covariates_to_drop['recent_earnings'] = ['RE75', 'pos75']
covariates_to_drop['pre_program_earnings'] = ['RE74', 'RE75', 'pos74', 'pos75']


N_EST = 100
MAX_DEPTH = 5

fit_imbens_df(imbens4_minus_edu, 'RE78', 'treatment',
              False, f'../out/imbens4_rf_no_smote_minus_edu/input_df.csv')
for cov_name, cov_val in covariates_to_drop.items():
    fit_imbens_df(imbens4_minus_edu, 'RE78', 'treatment', False,
                  f'../out/imbens4_rf_no_smote_minus_edu/covariates/{cov_name}.csv', covs=cov_val)

# ----get ATE imbens4
imbens4_ate, imbens4_sd = get_ate(imbens4, 'RE78', 'treatment')
# 2508.63, 7795.98
imbens4_minus_edu_ate, imbens4_minus_edu_sd = get_ate(
    imbens4_minus_edu, 'RE78', 'treatment')
# 1982.54, 7795.98
