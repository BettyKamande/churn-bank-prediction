# loading our libraries

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

st.title('Banking churn predictions.')
st.header('Predictions Data Analysis')

# loading our data
df = pd.read_csv("banking_churn.csv")


def main():
    st.title('Banking churn predictions.')
    st.header('Predictions Data Analysis')


gender_options = {
    0: 'Female',
    1: 'Male'
}
gender = st.selectbox(
    "", (0, 1), format_func=lambda x: gender_options.get(x))

geography_options = {
    0: 'France',
    1: 'Spain',
    2: 'Germany'
}
Geography = st.selectbox(
    "", (0, 1, 2), format_func=lambda x: geography_options.get(x))

age_bins_options = {
    0: 'Young',
    1: 'Adult',
    2: 'Old'
}
age_bins = st.selectbox(
    "", (0, 1, 2), format_func=lambda x: age_bins_options.get(x))

HasCrCard_options = {
    0: 'Yes',
    1: 'No'
}
HasCrCard = st.selectbox(
    "", (0, 1), format_func=lambda x: HasCrCard_options.get(x))

IsActiveMember_options = {
    0: 'No',
    1: 'Yes'
}

IsActiveMember = st.selectbox(
    "", (1, 0), key = "<uniquevalueofsomesort>" , format_func=lambda x: IsActiveMember_options.get(x))

NumOfProducts_options = {
    0: '1',
    1: '2',
    2: '3',
    3: '4'
}
NumOfProducts = st.selectbox(
    "", (0, 1), format_func=lambda x: NumOfProducts_options.get(x))

Tenure = st.slider('0,50')

CreditScore = st.slider('0,100000')

Balance = st.number_input('Balance')

EstimatedSalary = st.number_input('EstimatedSalary')

Age = st.number_input('Age')

file = open("model_random_forest_classifier_model_1.pkl", "rb")
loaded_pickle_model = pickle.load(file)

inputs = pd.DataFrame([[gender, Geography, age_bins, HasCrCard, Tenure, IsActiveMember,
                        NumOfProducts, CreditScore, Balance, EstimatedSalary, Age]])  # our inputs

prediction = loaded_pickle_model.predict(inputs)[0]
predict_probability = loaded_pickle_model.predict_proba(inputs)

if st.button('Predict'):  # making and printing our prediction
    result = loaded_pickle_model.predict(inputs)
    updated_res = result.flatten().astype(float)
    st.success('The Prediction {}'.format(updated_res))

if __name__ == '__main__':
    main()
