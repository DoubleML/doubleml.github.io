import pathlib
from do_sensitivity import do_sensitivity as sen
from collections import defaultdict
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy as sp
from imblearn.over_sampling import SMOTE
from plotnine import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import os
GIT_PYTHON_REFRESH = 0


def predict_g(X, y, do_SMOTE, logs=False):
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        clf = RandomForestClassifier(
            random_state=42, oob_score=False, n_estimators=N_EST, max_depth=MAX_DEPTH)
        if do_SMOTE:
            X_balanced, y_balanced = SMOTE().fit_resample(
                X[train_index], y[train_index])
            clf.fit(X_balanced, y_balanced)
        else:
            clf.fit(X[train_index], y[train_index])
        predictions[test_index] = clf.predict_proba(X[test_index])[:, 1]
    assert np.isnan(predictions).sum() == 0
    if logs:
        mlflow.log_param('n_est', N_EST)
        mlflow.log_param('max_depth', MAX_DEPTH)
        mlflow.log_param('g_SMOTE', do_SMOTE)
        mlflow.log_metric('g_accuracy', accuracy_score(
            np.rint(predictions), y))
        column_names = ['age', 'education', 'black', 'hispanic',
                        'married', 'RE74', 'RE75', 'pos74', 'pos75']
        feature_importances = pd.DataFrame(
            {'variables': column_names, 'score': clf.feature_importances_}).sort_values(by='score', ascending=False)
        feature_importances.to_csv(TEMP_FOLDER+'/g_feat_imp.csv')
    return predictions


def predict_Q(X, y, logs=False):
    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        reg = RandomForestRegressor(
            random_state=42, n_estimators=N_EST, max_depth=MAX_DEPTH)
        reg.fit(X[train_index], y[train_index])
        predictions[test_index] = reg.predict(X[test_index])
    assert np.isnan(predictions).sum() == 0
    if logs:
        mlflow.log_metric('q_rmse', np.sqrt(
            mean_squared_error(predictions, y)))
        column_names = ['treatment', 'age', 'education', 'black',
                        'hispanic', 'married', 'RE74', 'RE75', 'pos74', 'pos75']
        feature_importances = pd.DataFrame(
            {'variables': column_names, 'score': reg.feature_importances_}).sort_values(by='score', ascending=False)
        feature_importances.to_csv(TEMP_FOLDER+'/q_feat_imp.csv')
    return predictions


def fit_imbens_all(df, response_col, treatment_col, do_SMOTE):
    g = predict_g(df.drop([response_col, treatment_col], axis=1).values,
                  df[treatment_col].values, do_SMOTE, logs=True)

    Q = predict_Q(df.drop([response_col], axis=1).values,
                  df[response_col].values, logs=True)

    return g, Q, df[treatment_col], df[response_col]


def fit_imbens_excl_cov(df, covariate_col, response_col, treatment_col, do_SMOTE):
    ghat = predict_g(df.drop(
        [response_col, treatment_col]+covariate_col, axis=1).values, df[treatment_col], do_SMOTE)
    Qhat = predict_Q(df.drop(
        [response_col]+covariate_col, axis=1).values,
        df[response_col].values, logs=False)
    return ghat, Qhat


def fit_imbens(df, response_col, treatment_col, covariates_to_drop, do_SMOTE):
    g, Q, t, y = fit_imbens_all(df, response_col, treatment_col, do_SMOTE)
    covariate_params = defaultdict(list)
    for name, col in covariates_to_drop.items():
        ghat, Qhat = fit_imbens_excl_cov(
            df, col, response_col, treatment_col, do_SMOTE)
        covariate_params[name].append(ghat)
        covariate_params[name].append(Qhat)
    return g, Q, t, y, covariate_params


imbens1 = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens1.csv'))
imbens2 = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens2.csv'))


covariates_to_drop = {x: [x] for x in ['age', 'education', 'black', 'hispanic',
                                       'married', 'RE74', 'RE75', 'pos74', 'pos75']}
covariates_to_drop['recent_earnings'] = ['RE75', 'pos75']
covariates_to_drop['pre_program_earnings'] = ['RE74', 'RE75', 'pos74', 'pos75']

mlflow.set_tracking_uri('file:./out/mlruns')
mlflow.set_experiment('fit_imbens_rf')
for data, name in [(imbens1, 'imbens1'), (imbens2, 'imbens2')]:
    for smote in [True, False]:
        for N_EST, MAX_DEPTH in [(100, 5), (10, 3)]:
            TEMP_FOLDER = './out/temp_mlruns'
            os.makedirs(TEMP_FOLDER, exist_ok=True)
            N_EST = N_EST
            MAX_DEPTH = MAX_DEPTH
            BIAS = 1000
            GIT_PYTHON_REFRESH = 0
            mlflow.start_run()
            mlflow.log_param('file', 'fit_imbens_rf.py')
            mlflow.log_param('data', name)
            mlflow.log_param('bias', BIAS)
            mlflow.log_param('model', 'rf')
            g, Q, t, y, covariate_params = fit_imbens(
                data, 'RE78', 'treatment', covariates_to_drop, smote)
            # print(g, Q, t, y, covariate_params)
            sensitivity_df, variable_importances_df, p = sen.plot_sensitivity_graph(
                g, Q, t, y, BIAS, covariate_params)
            sensitivity_df.to_csv(
                TEMP_FOLDER+'/sensitivity_df.csv', index=False)
            variable_importances_df.to_csv(
                TEMP_FOLDER+'/variable_importances_df.csv', index=False)
            p.save(TEMP_FOLDER+'/plot.png')
            mlflow.log_artifacts(TEMP_FOLDER)
            mlflow.end_run()
