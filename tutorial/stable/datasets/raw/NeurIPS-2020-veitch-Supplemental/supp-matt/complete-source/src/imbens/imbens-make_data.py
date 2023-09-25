import pathlib
import pandas as pd


def make_data_imbens(df):
    df_new = df.drop(['nodegree'], axis=1)
    df_new['pos74'] = (df_new['RE74'] > 0).astype(int)
    df_new['pos75'] = (df_new['RE75'] > 0).astype(int)
    return df_new


col_names = ['treatment', 'age', 'education', 'black',
             'hispanic', 'married', 'nodegree', 'RE74', 'RE75', 'RE78']
control = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-raw', 'nswre74_control.txt'), header=None, sep=r"\s\s", names=col_names, engine='python')
treatment = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-raw', 'nswre74_treated.txt'), header=None, sep=r"\s\s", names=col_names, engine='python')
psid = pd.read_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-raw', 'psid_controls.txt'), header=None, sep=r"\s\s", names=col_names, engine='python')
print(control.shape, treatment.shape, psid.shape)

imbens1 = pd.concat([control, treatment]).reset_index(drop=True)
imbens1 = make_data_imbens(imbens1)
imbens1.to_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens1.csv'), index=False)

imbens2 = pd.concat([treatment, psid]).reset_index(drop=True)
imbens2 = make_data_imbens(imbens2)
imbens2.to_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens2.csv'), index=False)

imbens3 = pd.concat([treatment, psid]).reset_index(drop=True)
imbens3['change_earnings'] = imbens3['RE78']-imbens3['RE74']
imbens3 = make_data_imbens(imbens3)
imbens3 = imbens3.drop('RE78', axis=1)
imbens3.to_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens3.csv'), index=False)

imbens4 = pd.concat([treatment, psid]).reset_index(drop=True)
imbens4 = imbens4[~((imbens4['RE74'] >= 5000) | (imbens4['RE75'] >= 5000))]
imbens4 = imbens4.reset_index(drop=True)
# Does this match imbens' number of datapoints?
print(imbens4.groupby('treatment').count())
# yes!
imbens4 = make_data_imbens(imbens4)
imbens4.to_csv(pathlib.Path.cwd().parents[0].joinpath(
    'data', 'imbens-cleaned', 'imbens4.csv'), index=False)
