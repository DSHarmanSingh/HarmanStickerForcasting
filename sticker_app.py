import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import datetime

model= joblib.load("xgboost_sales_model.pkl")
try:
    target_encodings= joblib.load("target_encodings.pkl")
except:
    target_encodings= {}

st.title("Sticker Sales Forcasting App")
st.subheader("Enter Sales Details")
date= st.date_input("Select Date", datetime.date.today())
country= st.text_input("Country")
store= st.text_input("Store")
product= st.text_input("Product")

if st.button("Predicted Sales"):
    input_data= pd.DataFrame([[date, country, store, product]], columns= ['date', 'country', 'store', 'product'])
    input_data['date']= pd.to_datetime(input_data['date'])
    input_data['year']= input_data['date'].dt.year
    input_data['month']= input_data['date'].dt.month
    input_data['weekday']= input_data['date'].dt.weekday
    le= LabelEncoder()
    input_data['weekday_encoded']= le.fit_transform(input_data['weekday'])
    
    categorical_cols= ['country', 'store', 'product']
    for  col in categorical_cols:
        if col in  target_encodings:
            input_data[col+ '_encoded']= input_data[col].map(target_encodings[col])
    input_data.drop(columns= ['date', 'store', 'product', 'country'], inplace= True)
    columns= ['year', 'month', 'country_encoded', 'store_encoded', 'product_encoded', 'weekday_encoded']
    input_data= input_data[columns]
    prediction= model.predict(input_data)[0]
    st.success(f"Predicted Sticker Sales: **{round(prediction, 2)}**")
    
    
