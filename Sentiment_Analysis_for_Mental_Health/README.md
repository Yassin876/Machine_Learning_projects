
# Sentiment Analysis for Mental Health

## Dataset

- **Source:** Kaggle — Sentiment Analysis for Mental Health
- **Size:** (27639, 3)
- **Classes:**
   - Depression (includes Suicidal)
   - Normal
   - Stress (includes Anxiety)
- **Link:** https://drive.google.com/file/d/1jwK34dhkgB9qO4o_568m79jKgKmsG4ty/view?usp=sharing

## Problem

Mental health issues are often expressed through text on social media and online forums. Automatically detecting the sentiment and mental health status from these texts can help in early intervention and support for individuals in need.

## Why this is hard

- Mental health expressions are subtle and context-dependent.
- Text data is noisy, unstructured, and may contain slang or misspellings.
- Imbalanced classes (some mental health statuses are less frequent).
- Requires robust preprocessing and feature extraction to capture meaningful patterns.

## My approach

1. **Data Cleaning & Preprocessing:**
   - Removed duplicates and missing values.
   - Cleaned text using regular expressions, stopword removal, and stemming.
2. **Exploratory Data Analysis (EDA):**
   - Analyzed class distribution and visualized data.
3. **Label Simplification:**
   - Merged similar classes for better model performance.
   - **Justification:**
     - *Suicidal* was merged into *Depression* because both represent severe negative mental health states and often require similar intervention strategies. Combining them helps address class imbalance and improves model reliability.
     - *Anxiety* was merged into *Stress* as both share overlapping symptoms and textual expressions, making them difficult to distinguish reliably in short text. This also helps balance the dataset and enhances classification performance.
4. **Feature Extraction:**
   - Used TF-IDF vectorization for text features.
5. **Model Training:**
   - Trained Random Forest, Naive Bayes, and XGBoost classifiers.
   - Handled class imbalance with sample weights.
6. **Evaluation:**
   - Compared models using accuracy and classification reports.

## Why I chose this model

- **Random Forest:** Robust to overfitting, handles high-dimensional data well, and provides good baseline performance.
- **Naive Bayes:** Fast and effective for text classification tasks.
- **XGBoost:** Powerful gradient boosting model, often achieves state-of-the-art results in classification tasks.

## Challenges

- Handling imbalanced data and rare classes.
- Ensuring text preprocessing captures the nuances of mental health expressions.
- Selecting the right features and models for optimal performance.
- Managing dependencies and compatibility issues in the Python environment.


## Results

Below are the detailed results for each model on the test set:


### Random Forest

**Accuracy:** 0.9493

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Depression |   0.96    |  0.98  |   0.97   |  4070   |
| Normal     |   0.91    |  0.91  |   0.91   |  1264   |
| Stress     |   0.96    |  0.61  |   0.75   |   174   |

**Macro avg:** Precision 0.94, Recall 0.83, F1-score 0.88
**Weighted avg:** Precision 0.95, Recall 0.95, F1-score 0.95

---

### Naive Bayes

**Accuracy:** 0.8862

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Depression |   0.90    |  0.99  |   0.94   |  4070   |
| Normal     |   0.98    |  0.56  |   0.71   |  1264   |
| Stress     |   0.45    |  0.75  |   0.56   |   174   |

**Macro avg:** Precision 0.77, Recall 0.77, F1-score 0.74
**Weighted avg:** Precision 0.90, Recall 0.89, F1-score 0.88

---

### XGBoost

**Accuracy:** 0.9443

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Depression |   0.99    |  0.95  |   0.97   |  4070   |
| Normal     |   0.86    |  0.96  |   0.90   |  1264   |
| Stress     |   0.76    |  0.80  |   0.78   |   174   |

**Macro avg:** Precision 0.87, Recall 0.90, F1-score 0.88
**Weighted avg:** Precision 0.95, Recall 0.94, F1-score 0.95

---

The final model and preprocessing pipeline were saved for deployment.


## User Interface

The project uses **Streamlit** to provide an interactive web-based user interface for sentiment prediction. Users can input text and receive real-time mental health sentiment analysis results in a simple, accessible format.

## Project Structure


```
ml_school_task/
├── models/
│   ├── label_encoder.pkl
│   ├── Mental_Health.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── README.md
├── requirements.txt
└── Sentiment_Analysis_for_Mental_Health.ipynb
```

- **models/**: Contains serialized models and preprocessing objects (such as Mental_Health.pkl, tfidf_vectorizer.pkl, label_encoder.pkl) used for inference and deployment. This folder is not tracked by git.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```


## Demo (video)

- https://drive.google.com/file/d/1XOI48Nu3muYoY9QbUiKiNlTp6DOQJigc/view?usp=sharing