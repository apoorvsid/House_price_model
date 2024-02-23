from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor 
import pandas as pd 

df = pd.read_csv('Housing.csv')
to_convert = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
def to_map(x):
  return x.map({'yes':1,'no':0})
df[to_convert] = df[to_convert].apply(to_map)
status = pd.get_dummies(df['furnishingstatus'])
df = pd.concat([df, status], axis=1)
df.drop(['furnishingstatus'], axis=1, inplace=True)
df.drop(['bedrooms'], axis=1, inplace=True)
df.drop(['mainroad'], axis=1, inplace=True)
df.drop(['basement'], axis=1, inplace=True)
df.drop(['hotwaterheating'], axis=1, inplace=True)
df['price'] = df['price']/1.5
# to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
# df[to_scale] = MinMaxScaler().fit_transform(df[to_scale])
y_train = df.pop('price')
X_train = df

to_scale = ['area', 'bathrooms', 'stories', 'parking']
scaler = MinMaxScaler()
X_train_1[to_scale] = scaler.fit_transform(X_train[to_scale])

# Hyperparameter tuning using GridSearchCV for XGBoost model
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
}

xgb_model = XGBRegressor()
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, verbose=1)
grid_search.fit(X_train_1, y_train_1)

# Best hyperparameters
best_learning_rate = grid_search.best_params_['learning_rate']
best_max_depth = grid_search.best_params_['max_depth']
best_n_estimators = grid_search.best_params_['n_estimators']

# Train XGBoost model with best hyperparameters
xgb_model = XGBRegressor(learning_rate=best_learning_rate, max_depth=best_max_depth, n_estimators=best_n_estimators)
xgb_model.fit(X_train_1, y_train_1)

reg = LinearRegression().fit(X_train, y_train)
