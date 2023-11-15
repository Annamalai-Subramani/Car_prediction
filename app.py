import streamlit as st
import numpy as np
import pandas as pd
import datetime
import pickle
import sklearn




html_temp="""
     <div style = "background-color:lightblue;padding:16px">
     <h2 style="color:black;text-align:center;"> Car Price Prediction Using ML</h2>
     </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.markdown("##### Are you planning to sell your car !?\n##### So let's try evaluating the price..")

car = pd.read_csv('Clean_Car_data.csv')
model=pickle.load(open('car_predict.pkl','rb'))

with st.sidebar:
    st.subheader('car specs to predict price')

name = st.selectbox('Model name',(car['name'].unique()))
company = st.selectbox('company',(car['company'].unique()))
year = st.number_input('year', min_value = 2010,max_value = 2023, value=2010, step=1)
kms_driven = st.number_input('kms', min_value = 0,max_value = 200000, value=10000, step=500)
fuel_type = st.selectbox('fuel_type',(car['fuel_type'].unique()))


if st.button('predict'):

    df = pd.DataFrame({'name': [name], 'company': [company], 'year': [year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})
    st.table(df)

    result = model.predict(df)
    st.text('The Prediction Price of the Car is ' + str(np.round(result)))
   
  



    

   