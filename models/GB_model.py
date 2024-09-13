import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df = pd.read_csv('temp_yield.csv')
df_dropped = df.drop(columns=['Country', 'Tmin', 'Tmax'])


X = df[['avg_temp', 'HWN', 'HWM', 'CWN', 'CWM', 'Country','year']]
y = df['Yield_Value']

numerical_features = ['avg_temp', 'HWN', 'HWM', 'CWN', 'CWM', 'year']
categorical_features = ['Country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=0, n_estimators = 500, max_depth = 7, learning_rate = 0.1, subsample = 0.7))
])

pipeline.fit(X, y)

y_pred = pipeline.predict(X)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
