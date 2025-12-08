
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

tab1,tab2,tab3 = st.tabs(["Overview","Model","Predict"])

with tab1:
    st.write(df.head())
    st.bar_chart(df["Survived"].value_counts())

with tab2:
    st.write("Model loaded successfully.")

with tab3:
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
