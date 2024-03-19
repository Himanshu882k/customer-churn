# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:34:56 2023

@author: himan
"""

import streamlit as st
import joblib
import catboost
from catboost import CatBoostClassifier
import pandas as pd

st.title('Churn Prediction App')

feature_cols = ['account.length', 'intl.mins', 'intl.calls', 'intl.charge', 'day.mins',
       'day.calls', 'day.charge', 'eve.mins', 'eve.calls', 'eve.charge',
       'night.mins', 'night.calls', 'night.charge', 'customer.calls',
       'intl.plan1', 'intl.plan2']

# Load the trained model
model = joblib.load(open("./catboost.pkl", "rb"))

# Define the Streamlit app
def main():
    input={'no':0 , 'yes':1}
    # Create input fields for user to input data
    account_length = st.number_input('Account Length', 0, 243, 0)
    int_plan1 = st.selectbox('Does not have International Plan', ['no', 'yes'])
    int_plan2 = st.selectbox('Has International Plan', ['no', 'yes'])
    input_fields=[]
    for i in [ 'intl.mins', 'intl.calls', 'intl.charge', 'day.mins',
       'day.calls', 'day.charge', 'eve.mins', 'eve.calls', 'eve.charge',
       'night.mins', 'night.calls', 'night.charge', 'customer.calls'
       ]:
        input_fields.append(st.number_input(i))
    
    user_input=[account_length,*input_fields,input[int_plan1],input[int_plan2]]

    # Make predictions
    if st.button('Predict'):
        prediction = model.predict([user_input])
        
        if prediction[0]==0:
            st.success('Prediction churn = No')
        else:
            st.success('prediction churn = Yes')

if __name__ == '__main__':
    main()
