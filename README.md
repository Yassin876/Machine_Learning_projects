# Machine Learning Projects Overview

This repository contains several machine learning projects, each focused on a different real-world application. Below is a brief description of each project:

---

## 1. courses_recommendation_system
- **Description:**
  - A recommendation system for online courses (e.g., Coursera).
  - Uses course data to suggest relevant courses to users.
- **Key Files:**
  - `app.py`: Main application code.
  - `Coursera.csv`: Dataset of courses.
  - `notebook_RS.ipynb`: Jupyter notebook for data analysis and model development.
  - `processed_data.csv`: Preprocessed data for the model.

---

## 2. defualt_classfier
- **Description:**
  - A classifier to predict loan default risk.
  - Helps in identifying customers likely to default on loans.
- **Key Files:**
  - `app.py`: Main application code.
  - `Loan_Default.csv`: Dataset for loan default prediction.
  - `default_notebook.ipynb`: Jupyter notebook for data analysis and modeling.

---

## 3. diabetes
- **Description:**
  - A machine learning model to predict diabetes.
  - Uses patient data to assess diabetes risk.
- **Key Files:**
  - `app.py`: Main application code.
  - `diabetes.csv`: Dataset for diabetes prediction.
  - `diabetes_notebook.ipynb`: Jupyter notebook for data analysis and modeling.

---

## 4. emails
- **Description:**
  - An email classification project.
  - Can be used for spam detection or categorizing emails.
- **Key Files:**
  - `app.py`, `app1.py`: Application code.
  - `emails.csv`: Dataset of emails.

---

## 5. netflix_rs
- **Description:**
  - A recommendation system for Netflix titles.
  - Suggests movies or shows based on user preferences.
- **Key Files:**
  - `app.py`: Main application code.
  - `netflix_titles.csv`: Dataset of Netflix titles.
  - `preprocessing_data.csv`: Preprocessed data for the model.
  - `RSMovie.ipynb`: Jupyter notebook for data analysis and model development.

## 6. Sentiment Analysis for Mental Health

- **Description:**
   * A machine learning model to classify mental health status from text.
   * Detects Depression, Stress, and Normal states from social media expressions.
- **Key Files:**
   * `app.py`: Main application code.
   * `Sentiment_Analysis_for_Mental_Health.ipynb`: Jupyter notebook for data analysis and modeling.
   * `models/`: Saved model, TF-IDF vectorizer, and label encoder.
---

## 7. Sentiment Analysis for Mental Health updated

- **Description:**
  * A classical machine learning system that classifies text statements 
  into five mental health conditions: Anxiety, Depression, Normal, 
  Stress, and Suicidal.
  
  * The pipeline includes text preprocessing (lemmatization, TF-IDF), 
  data augmentation via synonym replacement to handle class imbalance, 
  and hyperparameter tuning using RandomizedSearchCV with StratifiedKFold. 
  Four models were evaluated; Random Forest achieved the best performance 
  (77.96% accuracy, Macro F1: 0.80) and is deployed via a Streamlit web app.
- **Key Files:**
   * `app.py`: Main application code.
   * `Sentiment_Analysis_for_Mental_Health.ipynb`: Jupyter notebook for data analysis and modeling.

Each project includes code, datasets, and notebooks for experimentation and analysis. You can explore each folder for more details and to run the applications or notebooks.
