import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# model = tf.keras.models.load_model('model.keras') #load models
# model = tf.keras.models.load_model('E:\Anushka-playground\Gen_AI\notebooks\model.keras')#load models
model = tf.keras.models.load_model(r'E:\Anushka-playground\Gen_AI\notebooks\model.keras') #load models

with open(r'E:\Anushka-playground\Gen_AI\notebooks\Encoder.pkl', 'rb') as file: #load encoder and scaler
    Encoder = pickle.load(file)
with open(r'E:\Anushka-playground\Gen_AI\notebooks\onehot.pkl','rb') as file:
    onehot = pickle.load(file)
with open(r'E:\Anushka-playground\Gen_AI\notebooks\scaler.pickle', 'rb')as file:
    scaler = pickle.load(file)

#stream app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot.categories_[0])
gender = st.selectbox('Gender', Encoder.classes_)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_df = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [Encoder.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    "EstimatedSalary" : [estimated_salary]  
})

#onehot encoding of geography
geo_encoded = onehot.transform([[geography]]) #onehot encoder doing the job to convert it to array that is returning numpy array so toarray() not needed
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot.get_feature_names_out(['Geography']))

#one hot concat
input_df = pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_scaled = scaler.transform(input_df)

#Predict churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability:{prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')