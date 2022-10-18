import pickle
import streamlit as st
import pandas as pd
from joblib import dump
from joblib import load


# dump(model, 'filename.joblib')
# from sklearn.externals import joblib
# N,P,K,temperature,humidity,ph,rainfall,label
df = pd.read_csv("./Crop_recommendation.csv")
classess = df["label"].unique()
print(df)

f=open('finalized_model.sav', 'rb')

# Title
st.header("Streamlit Machine Learning App")

# Input bar 1
N = st.number_input("Enter N")

# Input bar 2
P = st.number_input("Enter P")


K = st.number_input("Enter K")

temperature = st.number_input("Enter temp")

humidity = st.number_input("Enter humidity")

ph = st.number_input("Enter ph")

rainfall = st.number_input("Enter rainfall")


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = pickle.load(f) 
    
    
    X=clf.predict([[N,P,K,temperature,humidity,ph,rainfall]])

        # Output prediction
    st.text(f"YOU SHOULD GROW {X[0]} UNDER THE MENTIONED CONDITIONS.")