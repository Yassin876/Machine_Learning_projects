import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from collections import Counter
model=joblib.load(r"model_emails.pkl")
original_col=pd.read_csv(r"emails.csv").drop(['Email No.','Prediction'],axis=1).columns
st.title("spam or not spam ")

list_of_words=st.text_input("your email : ").lower().split()

count_words=Counter(list_of_words)
data={col:[count_words.get(col,np.nan)] for col in original_col}
df=pd.DataFrame(data)
df=df.fillna(value=0)


button=st.button("Check 🤖")

if button :
    pred=model.predict(df)

    if pred==0:
        st.write("thes is email is not spam 📩  ")
        st.image(r"not_spam.jpg",width=500)

    else :
        st.write("thes is email is spam ⚠️ ")
        st.image(r"spam.jpg",width=500)
