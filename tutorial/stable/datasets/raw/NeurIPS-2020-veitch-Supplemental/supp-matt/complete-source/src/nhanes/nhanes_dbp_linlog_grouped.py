import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
import numpy as np
import glob


def optimize_for_g(X, y):
    np.random.seed(42)
    clf = GridSearchCV(LogisticRegression(random_state=42, solver='liblinear'), param_grid={
                       'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='accuracy')
    clf.fit(X, y)
    print(clf.cv_results_)


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
    # print(accuracy_score(np.rint(predictions), y))
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


def prepare_data(df, treat_col, response_col, output_path, covs=[]):
    g = predict_g(df.drop(
        columns=[treat_col, response_col]+covs).values, df[treat_col].values)
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

optimize_for_g(dbp.drop(
    columns=['trt_dbp', 'ave_dbp']).values, dbp['trt_dbp'].values)
# mean_test_score': array([0.7361668 , 0.74218123, 0.73977546, 0.74097835, 0.74017642])
# doesn't matter, hence use default

prepare_data(dbp, 'trt_dbp', 'ave_dbp',
             '../out/nhanes_dbp_linlog_grouped/input_df.csv')

combined_covs = defaultdict(list)

combined_covs["habits"] = ['alcohol', 'packyr']
combined_covs["race"] = ['black', 'hisp', 'white']
combined_covs["wealth"] = ['income',  'insurance']
combined_covs["bloodwork"] = [
    'bmi', 'potassium', 'pulse', 'r_sodipota', 'sodium']
combined_covs["social"] = ['divorced', 'hhsize',
                           'married', 'widowed', 'together', 'separated']
combined_covs["gender"] = ["female"]
combined_covs["age"] = ["age_mo"]
combined_covs["education"] = ["edu"]
combined_covs["treatment"] = ["trt_dbp"]
combined_covs["socioeconomic"] = combined_covs['social'] + combined_covs['wealth'] + \
    combined_covs['habits']+combined_covs['race'] + combined_covs['education']

for name, cov_group in combined_covs.items():
    # print(name, cov_group)
    prepare_data(
        dbp, 'trt_dbp', 'ave_dbp', f"../out/nhanes_dbp_linlog_grouped/covariates/{name}.csv", covs=cov_group)


------check rmse
os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
linlog = pd.read_csv('../out/nhanes_dbp_linlog_grouped/input_df.csv')
bart_log = pd.read_csv(
    '../out/nhanes_dbp_bart_logreg_kfold_grouped/input_df.csv')
bart_rf = pd.read_csv('../out/nhanes_dbp_bart_rf_kfold_grouped/input_df.csv')

np.sqrt(mean_squared_error(linlog['Q'], linlog['y']))
# 13.06644474228842
np.sqrt(mean_squared_error(bart_log['Q'], bart_log['y']))
# 12.659021449479413
np.sqrt(mean_squared_error(bart_rf['Q'], bart_rf['y']))
# 12.659021449479413

log_loss(linlog['t'], linlog['g'])
# 0.5093452057379549
log_loss(bart_log['t'], bart_log['g'])...
# 0.5093452057379549
log_loss(bart_rf['t'], bart_rf['g'])
# 0.5139777376873012
