
import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Ensure project root is on `sys.path` so package `src` can be imported
# when unpickling the saved pipeline that references `src.*` modules.
sys.path.append(str(Path(__file__).resolve().parents[1]))

st.title("Titanic Survival Predictor")

model=joblib.load(Path(__file__).parent.parent / "models/pipeline.joblib")
df=pd.read_csv(Path(__file__).parent.parent / "data/train.csv")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Data Exploration", "Insights", "Prediction", "Model Performance"])

if page == "Home":
    st.write("Welcome to the **Titanic Survival Predictor**.")
    st.write("This app uses machine learning to predict whether a passenger would survive the Titanic disaster.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", caption="RMS Titanic")

elif page == "Data Exploration":
    st.subheader("Data Exploration")
    st.write("Here are the first few rows of the training dataset:")
    st.write(df.head())
    st.write("Survival Count:")
    st.bar_chart(df["Survived"].value_counts())

elif page == "Insights":
    st.subheader("Insights")
    st.write("Passenger Class Distribution:")
    st.bar_chart(df["Pclass"].value_counts())
    st.write("Embarked Location Distribution:")
    st.bar_chart(df["Embarked"].value_counts())

elif page == "Prediction":
    st.subheader("Prediction")
    pclass=st.selectbox("Pclass",[1,2,3])
    name=st.text_input("Name","John Doe")
    sex=st.selectbox("Sex",["male","female"])
    age=st.number_input("Age",0,100,30)
    sib=st.number_input("SibSp",0,10,0)
    par=st.number_input("Parch",0,10,0)
    fare=st.number_input("Fare",0.0,600.0,7.25)
    emb=st.selectbox("Embarked",["S","C","Q"])
    if st.button("Predict"):
        inp=pd.DataFrame([{
            "Pclass":pclass,"Name":name,"Sex":sex,"Age":age,
            "SibSp":sib,"Parch":par,"Fare":fare,"Embarked":emb
        }])
        pred=model.predict(inp)[0]
        st.success("Survived" if pred==1 else "Not Survived")

elif page == "Model Performance":
    st.subheader("Model Performance")
    st.write("The model used is a **Random Forest Classifier**.")
    st.write("Training Accuracy: **~80%**")
    st.info("The model was trained on the Titanic training dataset using scikit-learn.")

