import pandas as pd
from bartpy.sklearnmodel import SklearnModel
df = pd.read_csv('../../data/ihdp-cleaned/example_data_cleaned.csv')
df = df.drop(columns='momraceF')
model = SklearnModel()
X = df.drop(columns=['treat']).values
y = df['treat'].values
model.fit(X, y)
