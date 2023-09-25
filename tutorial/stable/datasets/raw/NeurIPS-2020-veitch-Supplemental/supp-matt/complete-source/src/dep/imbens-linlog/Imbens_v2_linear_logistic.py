import pathlib

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy as sp
from imblearn.over_sampling import SMOTE
from plotnine import *
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
GIT_PYTHON_REFRESH = 0

# region Functions
# HEADER


def predict_g(X, y, do_SMOTE, logs=False):
    # Preprocessing added!
    X = preprocessing.scale(X)
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
    if logs:
        mlflow.log_param('g_SMOTE', do_SMOTE)
        mlflow.log_metric('g_accuracy', accuracy_score(
            np.rint(predictions), y))
        column_names = ['age', 'education', 'black', 'hispanic',
                        'married', 'RE74', 'RE75', 'pos74', 'pos75']
        feature_importances = pd.DataFrame({'variables': column_names, 'score': np.squeeze(
            clf.coef_)}).sort_values(by='score', ascending=False)
        feature_importances.to_csv('temp_mlflow_files/g_feat_imp.csv')
        mlflow.log_artifact('temp_mlflow_files/g_feat_imp.csv')
    return predictions


def predict_Q(X, y, logs=False):
    # Preprocessing added!

    X = preprocessing.scale(X)

    np.random.seed(42)
    predictions = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        reg = LinearRegression()
        reg.fit(X[train_index], y[train_index])
        predictions[test_index] = reg.predict(X[test_index])
    assert np.isnan(predictions).sum() == 0
    if logs:
        mlflow.log_metric('q_rmse', np.sqrt(
            mean_squared_error(predictions, y)))
        column_names = ['treatment', 'age', 'education', 'black',
                        'hispanic', 'married', 'RE74', 'RE75', 'pos74', 'pos75']
        feature_importances = pd.DataFrame({'variables': column_names, 'score': np.squeeze(
            reg.coef_)}).sort_values(by='score', ascending=False)
        feature_importances.to_csv('temp_mlflow_files/q_feat_imp.csv')
        mlflow.log_artifact('temp_mlflow_files/q_feat_imp.csv')
    return predictions


def calc_delta(alpha, g, bias):
    alpha_term = (1/alpha)-1
    bias_term = sp.special.digamma(g*alpha_term+1) - sp.special.digamma(
        (1-g)*alpha_term) - sp.special.digamma(g*alpha_term) + sp.special.digamma((1-g)*alpha_term+1)
    delta = bias/np.mean(bias_term)
    return delta


def calc_beta_shapes(g, alpha):
    alpha_shape = g*((1/alpha)-1)
    beta_shape = (1-g)*((1/alpha)-1)
    return alpha_shape, beta_shape


def calc_Rsq(alpha, delta, calculated_params, treatment, response):
    alpha_shape, beta_shape = calc_beta_shapes(calculated_params['g'], alpha)
    Rsq_num = delta**2*np.mean(sp.special.polygamma(1, alpha_shape +
                                                    treatment) + sp.special.polygamma(1, beta_shape+(1-treatment)))
    Rsq_den = mean_squared_error(response, calculated_params['Q'])
    return Rsq_num/Rsq_den


def calc_Rsq_hat(covariate_col, response_col, df, Q):
    y = df[response_col].values
    Qhat = predict_Q(df.drop([response_col]+covariate_col, axis=1).values, y)
    Rsqhat = (mean_squared_error(y, Qhat)-mean_squared_error(y, Q)) / \
        (mean_squared_error(y, Qhat))
    if Rsqhat < 0:
        Rsqhat = 0
    return Rsqhat


def calc_alpha_hat(covariate_col, response_col, treatment_col, df, g, do_SMOTE):
    # call y t
    t = df[treatment_col].values
    ghat = predict_g(df.drop(
        [response_col, treatment_col]+covariate_col, axis=1).values, t, do_SMOTE)
#     ahat_numerator=np.var(g)-np.var(ghat)
#     ahat_denominator=np.mean(ghat*(1-ghat))
    ahat = 1-(np.mean(g*(1-g))/np.mean(ghat*(1-ghat)))
    # ahat=ahat_numerator/ahat_denominator
    if ahat < 0:
        ahat = 0
    return ahat


def plot_sensitivity_graph(df, treatment_col, response_col, covariate_cols, covariate_groups, bias, do_SMOTE):
    # Calculate g and Q
    g = predict_g(df.drop([response_col, treatment_col], axis=1).values,
                  df[treatment_col].values, do_SMOTE, logs=True)
    Q = predict_Q(df.drop([response_col], axis=1).values,
                  df[response_col].values, logs=True)
    calculated_params = pd.DataFrame({'g': g, 'Q': Q})

    # Calculate alpha, delta, Rsq
    alpha_series = pd.Series(np.arange(0.0001, 1, 0.0001))
    # alpha_series=pd.Series(np.arange(0.001,1,0.001))
    delta_series = alpha_series.apply(
        calc_delta, g=calculated_params['g'], bias=bias)
    sensitivity_df = pd.DataFrame(
        {'alphas': alpha_series, 'deltas': delta_series})
    Rsq = []
    for n in range(len(alpha_series)):
        Rsq.append(calc_Rsq(alpha_series[n], delta_series[n],
                            calculated_params, df[treatment_col], df[response_col]))
    sensitivity_df['Rsq'] = Rsq

    # Plot observed co-variates
    alpha_hat = []
    Rsq_hat = []

    for covar in covariate_cols:
        alpha_hat.append(calc_alpha_hat(covar, response_col,
                                        treatment_col, df, calculated_params['g'], do_SMOTE))
        Rsq_hat.append(calc_Rsq_hat(
            covar, response_col, df, calculated_params['Q']))

    variable_importances = pd.DataFrame(
        {'covariates': covariate_cols, 'alpha_hat': alpha_hat, 'Rsq_hat': Rsq_hat, 'groups': covariate_groups})

    variable_importances.to_csv('temp_mlflow_files/variable_importances.csv')
    mlflow.log_artifact('temp_mlflow_files/variable_importances.csv')
    treatment_Rsq = calc_Rsq_hat(
        ['treatment'], response_col, df, calculated_params['Q'])
    mlflow.log_metric('treatment_Rsq', treatment_Rsq)

    g = (ggplot(data=sensitivity_df, mapping=aes(x='alphas', y='Rsq'))
         + geom_line(color='#585858', size=1)
         + theme_light()
         + geom_point(data=variable_importances, mapping=aes(x='alpha_hat',
                                                             y='Rsq_hat', fill='groups'), color="black", alpha=0.6, size=2.5)
         + theme(figure_size=(3.5, 3.5), legend_key=element_blank(), axis_title=element_text(
             size=10), axis_text=element_text(color='black'), plot_title=element_text(size=10))
         + scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.01, 0.43))
         # +scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.02, min(0.75, max(Rsq+Rsq_hat))))
         + scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.02, 0.6))
         + labs(x=r'$\mathregular{\alpha}$ (treatment)',
                fill='', y='partial $R^2$ (outcome)')
         + scale_fill_brewer(type='qual', palette='Set1', labels=[
             'Individual covariates', 'Pre-program earnings', 'Recent earnings'])
         # +geom_label(data=variable_importances, mapping=aes(x='alpha_hat', y='Rsq_hat', label='labels'), size=8, nudge_y=0.03, adjust_text={'expand_text': (2, 2), 'expand_points': (2,2), 'force_text': (0.1, 0.25), 'force_points':(0.2, 0.5)})
         + annotate("text", x=0.35, y=0.065, label='Bias = $' + \
                    str(bias), size=8.5, color='#303030')
         )
    g.save('temp_mlflow_files/plot.png')
    mlflow.log_artifact('temp_mlflow_files/plot.png')
    return None

# endregion


imbens1 = pd.read_csv(pathlib.Path.cwd().parents[1].joinpath(
    'data', 'imbens-cleaned', 'imbens1.csv'))
imbens2 = pd.read_csv(pathlib.Path.cwd().parents[1].joinpath(
    'data', 'imbens-cleaned', 'imbens2.csv'))
imbens3 = pd.read_csv(pathlib.Path.cwd().parents[1].joinpath(
    'data', 'imbens-cleaned', 'imbens3.csv'))
imbens4 = pd.read_csv(pathlib.Path.cwd().parents[1].joinpath(
    'data', 'imbens-cleaned', 'imbens4.csv'))


covariate_cols = ['age', 'education', 'black', 'hispanic',
                  'married', 'RE74', 'RE75', 'pos74', 'pos75']
recent_earnings = ['RE75', 'pos75']
# what about "pre-program earnings"? Is that RE74?
pre_program_earnings = ['RE74', 'RE75', 'pos74', 'pos75']

covariates_for_graphs = [[x] for x in covariate_cols]
covariates_for_graphs.append(recent_earnings)
covariates_for_graphs.append(pre_program_earnings)

covariate_groups = ['individual_covariates'] * \
    len(covariate_cols)+['recent_earnings', 'pre_program_earnings']


mlflow.set_experiment('with_py_files_122819')
for data, name in [(imbens1, 'imbens1'), (imbens2, 'imbens2')]:
    for smote in [True, False]:
        BIAS = 1000
        GIT_PYTHON_REFRESH = 0
        mlflow.start_run()
        mlflow.log_param('data', name)
        mlflow.log_param('bias', BIAS)
        mlflow.log_param('model', 'linlog')
        plot_sensitivity_graph(data,
                               treatment_col='treatment',
                               response_col='RE78',
                               covariate_cols=covariates_for_graphs,
                               covariate_groups=covariate_groups,
                               bias=BIAS,
                               do_SMOTE=smote)

        mlflow.end_run()
