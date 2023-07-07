import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
train = pd.read_csv("Train.csv")

# Data preprocessing function
def preprocess_data(df):
    columns_to_encode = ['NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'NAME_FAMILY_STATUS']
    for col in columns_to_encode:
        encoded_columns = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, encoded_columns], axis=1)
        df.drop(col, axis=1, inplace=True)

    df['NAME_EDUCATION_TYPE'].replace({'Secondary / secondary special': 2, 'Higher education': 3, 'Basic': 1}, inplace=True)
    df['NAME_INCOME_TYPE'].replace({'Pensioner': 2, 'Working': 3, 'Student': 1}, inplace=True)
    df['FLAG_OWN_CAR'].replace({'N': 0, 'Y': 1}, inplace=True)
    df['FLAG_OWN_REALTY'].replace({'N': 0, 'Y': 1}, inplace=True)
    df['DAYS_EMPLOYED_C'] = df['DAYS_EMPLOYED'] * (-1)
    df.loc[df['DAYS_EMPLOYED_C'] < 0, 'DAYS_EMPLOYED_C'] = 0
    df.drop('DAYS_EMPLOYED', axis=1, inplace=True)
    
    return df

# Preprocess the data
train = preprocess_data(train)

# Split the data into features (X) and target variable (y)
X = train.drop(["ID", "CARD_APPROVED"], axis='columns')
y = train['CARD_APPROVED']

# Train the Random Forest model
clf = RandomForestClassifier(n_estimators=50, class_weight='balanced')
clf.fit(X, y)

# Streamlit app
st.title("Credit Score Model Deployment")
st.write("This app predicts credit card approval using a Random Forest model.")

# Sidebar inputs
st.sidebar.header("User Inputs")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.text_input(col, value=str(X[col].median()))

# Make predictions
input_df = pd.DataFrame([input_data])
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Show prediction results
st.header("Prediction")
if prediction[0] == 0:
    st.write("The credit card application is **not approved**.")
else:
    st.write("The credit card application is **approved**.")
st.write(f"Probability of approval: {prediction_proba[0][1]:.2%}")
