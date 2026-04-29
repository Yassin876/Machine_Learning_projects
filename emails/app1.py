import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


model=joblib.load(r"model_emails.pkl")
original_col=pd.read_csv(r"emails.csv").drop(['Email No.','Prediction'],axis=1).columns

def ready_email(email):
    list_of_words=email.split()
    count_words=Counter(list_of_words)
    data={col:[count_words.get(col,np.nan)] for col in original_col}
    df=pd.DataFrame(data)
    df=df.fillna(value=0)

    return df

class Email(BaseModel):
    email: str

@app.post("/predict")
async def predict(email: Email):
    df = ready_email(email.email)
    pred = model.predict(df)
    if pred[0] == 1:
        return {"prediction": "spam"}
    return {"prediction": "not spam"}