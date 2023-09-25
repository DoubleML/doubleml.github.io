import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")

usek = pd.read_csv('../data/ihdp-cleaned/usek.csv')
usek.head()


def predict_g(X_train, y_train, X_test, y_test, do_SMOTE):
    # Preprocessed data to be fed!
    np.random.seed(42)
    predictions = np.full_like(y_test, np.nan, dtype=float)
    clf = LogisticRegression(random_state=42, solver='liblinear')
    if do_SMOTE:
        X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
        clf.fit(X_balanced, y_balanced)
    else:
        clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)[:, 1]
    assert np.isnan(predictions).sum() == 0
    print(accuracy_score(np.rint(predictions), y_test))
    return predictions


continuous_cols = ['bw', 'momage', 'nnhealth', 'cigs', 'alcohol', 'ppvt.imp']
usek[continuous_cols] = preprocessing.scale(usek[continuous_cols])
X_train = usek.drop(columns=['iqsb.36', 'dose400'])
y_train = usek['dose400']
X_test = usek[usek['dose400'] == 1].drop(columns=['iqsb.36', 'dose400'])
y_test = usek.loc[usek['dose400'] == 1, 'dose400']

p = predict_g(X_train.values, y_train.values,
              X_test.values, y_test.values, False)
# accuracy 0.0149 wihtout SMOTE
# 0.7313 with SMOTE

p_smote = predict_g(X_train.values, y_train.values,
                    X_test.values, y_test.values, True)


bart_preds = pd.read_csv('../out/ihdp_bart/sensitivity_df.csv')

print(accuracy_score(np.rint(bart_preds['g'].values), bart_preds['t'].values))
# 0.00 bart
# replaced bart preds g with logreg preds
bart_preds['g'] = p_smote

bart_preds.to_csv('../out/ihdp_log_reg_bart/sensitivity_df.csv', index=False)
