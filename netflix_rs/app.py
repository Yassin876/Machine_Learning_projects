import pandas as pd
import numpy as np 
import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

vectorizer=joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix=joblib.load('tfidf_matrix.pkl')
df=pd.read_csv("preprocessing_data.csv")

st.title(" Netflix Recomendation Movie And Tv Show 🎞️")

user_input=st.text_input("write eny information abuot movie or tv show 🎞️")
button=st.button("عرض المقترحات ")
if button:
    if user_input.strip() == "":
        st.warning("من فضلك ادخل اي معلومات ")
    else :
        user_input_vectorizer=vectorizer.transform([user_input])
        scores=cosine_similarity(user_input_vectorizer,tfidf_matrix)
        top_endix=np.argsort(scores,axis=1)[:,-5:].flatten()[::-1]
        the_top=df.iloc[top_endix]


        st.subheader("المقترحات ")
        for idx,row in the_top.iterrows():
            st.markdown(f"title : {row['title']}")
            st.markdown(f"type : {row['type']}")
            st.markdown(f"director : {row['director']}")
            st.markdown(f"cast : {row['cast']}")
            st.markdown(f"country : {row['country']}")
            st.markdown(f"release year : {row['release_year']}")
            st.markdown(f"rating : {row['rating']}")
            st.markdown(f"duration : {row['duration']}")
            st.markdown(f"category : {row['listed_in']}")
            st.markdown(f"description : {row['description']}")
            st.markdown("----")
