# 🎓 Coursera Recommender System

A web-based interactive system built with Streamlit that recommends Coursera courses to users based on their interests using Natural Language Processing (NLP) techniques and content-based recommendation algorithms.

---

## Project Idea
The system allows the user to enter their interests (fields or keywords), then displays the best matching courses from a large database collected from Coursera, with direct links to the courses and their details (university, rating, skills, description).

---

## Features
- Simple interface via Streamlit.
- Smart recommendations using TF-IDF and Cosine Similarity.
- Displays the top 5 most relevant courses with details.
- Automatic data cleaning and preprocessing for courses.

---

## How to Run
1. **Install requirements:**
   - Python 3.7 or higher.
   - Required packages (install via):
     ```
     pip install streamlit scikit-learn pandas numpy joblib
     ```

2. **Run the app:**
   ```
   streamlit run app.py
   ```

3. Enter your interests in the text box and click "Show Recommended Courses".

---

## Main Files
- `app.py` : Main Streamlit application.
- `Coursera.csv` : Original courses database.
- `notebook_RS.ipynb` : Jupyter notebook showing data processing and recommender system building steps.
- `processed_data.csv` : Courses data after cleaning and preprocessing.
- `tfidf_vectorizer.pkl` & `tfidf_matrix.pkl` : Saved model files.

---

## How does it work?
1. **Data processing:** Cleans the data, handles missing values, and combines text columns into one document.
2. **Text transformation:** Uses TF-IDF to convert text to numeric vectors.
3. **Similarity calculation:** When the user enters their interests, they are vectorized and compared to all courses.
4. **Display results:** Courses are ranked by similarity and the top 5 are shown to the user.

---

## Notes
- All data is local; nothing is sent to external servers.
- You can modify the database or retrain the model using the notebook.

--- 
