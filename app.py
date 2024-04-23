import streamlit as st
import pandas as pd 
import numpy as np
import schedule
import requests
import time
from model import reg
import seaborn as sns

st.header("House price prediction model")
st.subheader("Input user parameters")

def ping_app():
    url = 'https://predictor-model-3avz.onrender.com/' 
    response = requests.get(url)
    print('App pinged:', response.status_code)

def user_input_features():
  area = st.slider("Area of the house (in sq. ft.)", min_value=1500, max_value=16500, step=10)
  bedrooms = st.slider("Number of bedrooms in the house", min_value=1, max_value=6, step=1)
  bathrooms = st.slider("Number of bathrooms in the house", min_value=1, max_value=4, step=1)
  stories = st.slider("How many storeys the house has?", min_value=1, max_value=4, step=1)
  parking = st.slider("How many cars can be parked in the garage", min_value = 0, max_value=3, step=1)
  mainroad = st.radio("Is the house close to the nearest main road?", ("Yes", "No"), horizontal=True)
  guestroom = st.radio("Is there a guestroom?", ("Yes", "No"), horizontal=True)
  basement = st.radio("Is there a basement?", ("Yes", "No"), horizontal=True)
  hotwaterheating = st.radio("Does the house have a hot water heating system?", ("Yes", "No"), horizontal=True)
  airconditioning = st.radio("Does the house have air conditioning?", ("Yes", "No"), horizontal=True)
  prefarea = st.radio("Is the house located in your preferred neighborhood?", ("Yes", "No"), horizontal=True)
  furnishingstatus = st.radio("To what degree is the house furnished?", ("Furnished", "Semi-furnished", "Unfurnished"), horizontal=True)

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

df1 = pd.read_csv('Housing.csv')
# maxa = df1['area'].max()
# mina = df1['area'].min()
# narea = (df_user['AREA'].iloc[0]-mina)/(maxa-mina)
narea = df_user['AREA'].iloc[0]
# maxb = df1['bedrooms'].max()
# minb = df1['bedrooms'].min()
# nbedrooms = (df_user['BEDROOMS'].iloc[0]-minb)/(maxb-minb)
nbedrooms = df_user['BEDROOMS'].iloc[0]
# maxbb = df1['bathrooms'].max()
# minbb = df1['bathrooms'].min()
# nbathrooms = (df_user['BATHROOMS'].iloc[0]-minbb)/(maxbb-minb)
nbathrooms = df_user['BATHROOMS'].iloc[0]
# maxs = df1['stories'].max()
# mins = df1['stories'].min()
# nstories = (df_user['STORIES'].iloc[0]-mins)/(maxs-mins)
nstories = df_user['STORIES'].iloc[0]
# maxp = df1['parking'].max()
# minp = df1['parking'].min()
# nparking = (df_user['PARKING'].iloc[0]-minp)/(maxp-minp)
nparking = df_user['PARKING'].iloc[0]
nmainroad, nguestroom, nbasement, nhotwaterheating, nairconditioning, nprefarea, nfurnish, nsemi_furnish, nunfurnish = None, None, None, None, None, None, None, None, None
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
  nfurnish = 1
  nsemi_furnish = 0 
  nunfurnish = 0 
elif df_user['FURNISHINGSTATUS'].iloc[0]=='Semi-furnished': 
  nfurnish = 0
  nsemi_furnish = 1
  nunfurnish = 0 
elif df_user['FURNISHINGSTATUS'].iloc[0]=='Unfurnished':
  nfurnish = 0
  nsemi_furnish = 0 
  nunfurnish = 1

final_input = (narea, nbathrooms, nstories, nguestroom, nairconditioning, nparking, nprefarea, nfurnish, nsemi_furnish, nunfurnish)
final_input = np.asarray(final_input)
final_input = final_input.reshape(1, -1)
price_predict = reg.predict(final_input)
# maxpp = df1['price'].max()
# minpp = df1['price'].min()
# final_price_predict = (price_predict*(maxpp-minpp))+minpp
st.write("\n")
st.write("\n")
if st.button("Predict the price"):
  st.write("\n")
  st.write("User input parameters")
  st.write(df_user)
  st.write("\n")
  st.subheader("Predicted price of the house is :")
  st.subheader("${:0,.2f}".format(float(price_predict)))
  st.write("\n")
  st.subheader("Statistics about the data")
  st.write("Correlation Heatmap")
  corr_matrix = df1.corr()
  st.write(corr_matrix)
  # Create heatmap using seaborn
  sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
  attributes = ['area', 'bedrooms', 'bathrooms', 'stories']
  for attr in attributes:
      st.write(f"Price vs {attr.capitalize()}")
      
      # Check if attribute is 'area' to determine plot type
      if attr == 'area':
          st.line_chart(df1.groupby(attr)['price'].mean(), use_container_width=True)
      else:
          st.bar_chart(df1.groupby(attr)['price'].mean(), use_container_width=True)

schedule.every(14).minutes.do(ping_app)

while True:
    schedule.run_pending()
    time.sleep(1)
