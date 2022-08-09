from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor 
import pandas as pd 

df = pd.read_csv('Housing.csv')
to_convert = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
def to_map(x):
  return x.map({'yes':1,'no':0})
df[to_convert] = df[to_convert].apply(to_map)
status = pd.get_dummies(df['furnishingstatus'], drop_first=True)
df = pd.concat([df, status], axis=1)
df.drop(['furnishingstatus'], axis=1, inplace=True)
to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df[to_scale] = MinMaxScaler().fit_transform(df[to_scale])
y_train = df.pop('price')
X_train = df
reg = XGBRegressor().fit(X_train, y_train)