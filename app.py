from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd 
import numpy as np

st.write("House price prediction model")
st.header("Input user parameters")

def user_input_features():
  area = st.number_input("Area of the house", min_value=1500, max_value=16500, step=1)
  bedrooms = st.number_input("Number of bedrooms in the house", min_value=1, max_value=6, step=1)
  bathrooms = st.number_input("Number of bathrooms in the house", min_value=1, max_value=4, step=1)
  stories = st.number_input("How many storeys the house has?", min_values=1, max_value=4, step=1)
  mainroad = st.selectbox("Is the house close to the nearest main road?", ("Yes", "No"))
  guestroom = st.selectbox("Is there a guestroom?", ("Yes", "No"))
  basement = st.selectbox("Is there a basement?", ("Yes", "No"))
  hotwaterheating = st.selectbox("Does the house have a hot water heating system?", ("Yes", "No"))
  airconditioning = st.selectbox("Does the house have air conditioning?", ("Yes", "No"))
  parking = st.selectbox("Is parking available on the property?", ("Yes", "No"))
  prefarea = st.selectbox("Is there a swimmingpool in the premise?", ("Yes", "No"))
  furnishingstatus = st.selectbox("To what degree is the house furnished?", ("Furnished", "Semi-furnished", "Unfurnished"))

  data = {'Area of the house':area, 
          'Number of bedrooms in the house':bedrooms,
          'Number of bathrooms in the house':bathrooms,
          'How many storeys the house has?':stories,
          'Is the house close to the nearest main road?':mainroad,
          'Is there a guestroom?':guestroom,
          'Is there a basement?':basement,
          'Does the house have a hot water heating system?':hotwaterheating,
          'Does the house have air conditioning?':airconditioning,
          'Is parking available on the property?':parking,
          'Is there a swimmingpool?':prefarea,
          'To what degree is the house furnished?':furnishingstatus}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_features()
st.subheader("Your parameters")
st.write(df.to_dict())