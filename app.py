from typing import final
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor 
import streamlit as st
import pandas as pd 
import numpy as np

st.header("House price prediction model")
st.subheader("Input user parameters")

def user_input_features():
  area = st.number_input("Area of the house (in sq. ft.)", min_value=1500, max_value=16500, step=1)
  st.write("Min value :",1500, "Max value :",16500)
  bedrooms = st.number_input("Number of bedrooms in the house", min_value=1, max_value=6, step=1)
  st.write("Min value :",1, "Max value :",6)
  bathrooms = st.number_input("Number of bathrooms in the house", min_value=1, max_value=4, step=1)
  st.write("Min value :",1, "Max value :",4)
  stories = st.number_input("How many storeys the house has?", min_value=1, max_value=4, step=1)
  st.write("Min value :",1, "Max value :",4)
  mainroad = st.selectbox("Is the house close to the nearest main road?", ("Yes", "No"))
  guestroom = st.selectbox("Is there a guestroom?", ("Yes", "No"))
  basement = st.selectbox("Is there a basement?", ("Yes", "No"))
  hotwaterheating = st.selectbox("Does the house have a hot water heating system?", ("Yes", "No"))
  airconditioning = st.selectbox("Does the house have air conditioning?", ("Yes", "No"))
  parking = st.number_input("How many cars can be parked in the garage", min_value = 0, max_value=3, step=1)
  st.write("Min value :",0, "Max value :",3)
  prefarea = st.selectbox("Is there a swimmingpool in the premise?", ("Yes", "No"))
  furnishingstatus = st.selectbox("To what degree is the house furnished?", ("Furnished", "Semi-furnished", "Unfurnished"))

  data = {'AREA':area, 
          'BEDROOMS':bedrooms,
          'BATHROOMS':bathrooms,
          'STORIES':stories,
          'MAINROAD':mainroad,
          'GUESTROOM':guestroom,
          'BASEMENT':basement,
          'HOTWATERHEATING':hotwaterheating,
          'AIRCONDITIONING':airconditioning,
          'PARKING':parking,
          'PREFAREA':prefarea,
          'FURNISHINGSTATUS':furnishingstatus}
  features = pd.DataFrame(data, index=[0])
  return features

df_user = user_input_features()

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
df1 = pd.read_csv('Housing.csv')
maxa = df1['area'].max()
mina = df1['area'].min()
narea = (df_user['AREA'].iloc[0]-mina)/(maxa-mina)

maxb = df1['bedrooms'].max()
minb = df1['bedrooms'].min()
nbedrooms = (df_user['BEDROOMS'].iloc[0]-minb)/(maxb-minb)

maxbb = df1['bathrooms'].max()
minbb = df1['bathrooms'].min()
nbathrooms = (df_user['BATHROOMS'].iloc[0]-minbb)/(maxbb-minb)

maxs = df1['stories'].max()
mins = df1['stories'].min()
nstories = (df_user['STORIES'].iloc[0]-mins)/(maxs-mins)

maxp = df1['parking'].max()
minp = df1['parking'].min()
nparking = (df_user['PARKING'].iloc[0]-minp)/(maxp-minp)

nmainroad, nguestroom, nbasement, nhotwaterheating, nairconditioning, nprefarea, nsemi_furnish, nunfurnish = None, None, None, None, None, None, None, None
if df_user['MAINROAD'].iloc[0]=='Yes':
  nmainroad = 1
else:
  nmainroad = 0

if df_user['GUESTROOM'].iloc[0]=='Yes':
  nguestroom = 1
else:
  nguestroom = 0 

if df_user['BASEMENT'].iloc[0]=='Yes':
  nbasement=1
else:
  nbasement=0 

if df_user['HOTWATERHEATING'].iloc[0]=='Yes':
  nhotwaterheating = 1
else:
  nhotwaterheating = 0

if df_user['AIRCONDITIONING'].iloc[0]=='Yes':
  nairconditioning = 1
else:
  nairconditioning = 0 

if df_user['PREFAREA'].iloc[0]=='Yes':
  nprefarea = 1 
else:
  nprefarea = 0 

if df_user['FURNISHINGSTATUS'].iloc[0]=='Furnished': 
  nsemi_furnish = 0 
  nunfurnish = 0 
elif df_user['FURNISHINGSTATUS'].iloc[0]=='Semi-furnished': 
  nsemi_furnish = 1
  nunfurnish = 0 
elif df_user['FURNISHINGSTATUS'].iloc[0]=='Unfurnished':
  nsemi_furnish = 0 
  nunfurnish = 1

final_input = (narea, nbedrooms, nbathrooms, nstories, nmainroad, nguestroom, nbasement, nhotwaterheating, nairconditioning, nparking, nprefarea, nsemi_furnish, nunfurnish)
final_input = np.asarray(final_input)
final_input = final_input.reshape(1, -1)
price_predict = reg.predict(final_input)
maxpp = df1['price'].max()
minpp = df1['price'].min()
final_price_predict = (price_predict*(maxpp-minpp))+minpp

st.subheader("Predicted price of the house is :")
st.subheader("${:0,.2f}".format(float(final_price_predict)))
