import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# تحميل الموديلات المحفوظة
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
df = pd.read_csv("processed_data.csv")

# استخدام عادي:
st.title("🎓 نظام التوصية للكورسات - Coursera Recommender")

user_input = st.text_input("اكتب المجلات الذي من اهتماماتك :")

if st.button("اعرض الكورسات المقترحة"):
    if user_input.strip() == "":
        st.warning("من فضلك أدخل بعض الاهتمامات.")
    else:
        user_vector = tfidf.transform([user_input])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
        top_indices = np.argsort(similarity_scores, axis=1)[:, -5:].flatten()[::-1]
        recommendations = df.iloc[top_indices]

        st.subheader("📚 الكورسات المقترحة:")
        for idx, row in recommendations.iterrows():
            st.markdown(f"### [{row['Course Name']}]({row['Course URL']})")
            st.markdown(f"**الجامعة:** {row['University']}")
            st.markdown(f"**التقييم:** {round(row['Course Rating'], 2)} ⭐")
            st.markdown(f"**المهارات:** {row['Skills']}")
            st.markdown(f"**الوصف:** {row['Course Description']}")
            st.markdown("---")
