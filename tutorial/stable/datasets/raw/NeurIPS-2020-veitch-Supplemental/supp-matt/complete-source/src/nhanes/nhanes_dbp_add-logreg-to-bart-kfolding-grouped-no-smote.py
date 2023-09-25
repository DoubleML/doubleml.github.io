import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob
from collections import defaultdict


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
    print(accuracy_score(np.rint(predictions), y))
    return predictions


def prepare_data(df, bart_preds, output_path, covs=[]):
    X = df.drop(columns=['trt_dbp', 'ave_dbp']+covs)
    y = df['trt_dbp']
    p = predict_g(X.values, y.values)
    if covs != []:
        bart_preds['ghat'] = p
        bart_preds = bart_preds[['ghat', 'Qhat', 't', 'y']]
    else:
        bart_preds['g'] = p
        bart_preds = bart_preds[['g', 'Q', 't', 'y']]
    bart_preds.to_csv(output_path, index=False)


os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
dbp = pd.read_csv("../data/nhanes-cleaned/hbp_dbp.csv")

cont_vars = ['age_mo', 'hhsize', 'edu', 'income', 'packyr', 'bmi',
             'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol', 'together']
dbp[cont_vars] = preprocessing.scale(dbp[cont_vars])
input_df = pd.read_csv('../out/nhanes_dbp_bart_kfold_grouped/input_df.csv')


prepare_data(dbp, input_df,
             '../out/nhanes_dbp_bart_logreg_kfold_grouped/input_df.csv')

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
    df = pd.read_csv(
        f'../out/nhanes_dbp_bart_kfold_grouped/covariates/{name}.csv')
    prepare_data(
        dbp, df, f"../out/nhanes_dbp_bart_logreg_kfold_grouped/covariates/{name}.csv", covs=cov_group)


# noage
os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
dbp = pd.read_csv("../data/nhanes-cleaned/hbp_dbp.csv")
dbp_noage = dbp.drop(columns=['age_mo'])
cont_vars = ['hhsize', 'edu', 'income', 'packyr', 'bmi',
             'pulse', 'sodium', 'potassium', 'r_sodipota', 'alcohol', 'together']
dbp_noage[cont_vars] = preprocessing.scale(dbp_noage[cont_vars])
input_df = pd.read_csv(
    '../out/nhanes_dbp_bart_kfold_grouped_noage/input_df.csv')


prepare_data(dbp_noage, input_df,
             '../out/nhanes_dbp_bart_logreg_kfold_grouped_noage/input_df.csv')

combined_covs = defaultdict(list)

combined_covs["bloodwork"] = [
    'bmi', 'potassium', 'pulse', 'r_sodipota', 'sodium']
combined_covs["gender"] = ["female"]
combined_covs["treatment"] = ["trt_dbp"]
combined_covs["socioeconomic"] = ['alcohol', 'packyr',
                                  'black', 'hisp', 'white', 'income',  'insurance', 'divorced', 'hhsize', 'married', 'widowed', 'together', 'separated', "edu"]

for name, cov_group in combined_covs.items():
    df = pd.read_csv(
        f'../out/nhanes_dbp_bart_kfold_grouped_noage/covariates/{name}.csv')
    prepare_data(
        dbp_noage, df, f"../out/nhanes_dbp_bart_logreg_kfold_grouped_noage/covariates/{name}.csv", covs=cov_group)
