# Mental Health Sentiment Analysis

This project develops a sentiment analysis tool to predict mental health conditions based on text statements. It utilizes classical machine learning techniques, natural language processing (NLP), and data augmentation to handle class imbalances.

## Problem
The goal is to provide a predictive model that can classify text data into five categories: **Anxiety, Depression, Normal, Stress, and Suicidal**. This helps in early identification of emotional states and mental health risks from digital footprints or user input.

## Why this is hard
*   **Semantic Overlap**: Words used in expressing anxiety often overlap with those for depression or stress, making classification subtle.
*   **Class Imbalance**: Original datasets often have sparse data for specific conditions like 'Stress' compared to 'Normal' or 'Depression'.
*   **Lexical Complexity**: Emotional expressions involve nuances that simple keyword matching cannot capture.

## My approach
1.  **Lemmatization**: I used NLTK's `WordNetLemmatizer` instead of the *Porter Stemmer*. Lemmatization is superior because it understands the linguistic context and reduces words to their meaningful dictionary root (lemma), whereas stemming simply chops off suffixes.
2.  **TF-IDF Vectorization**: I chose TF-IDF (Term Frequency-Inverse Document Frequency) to transform text into numerical representations. It reflects how important a word is to a document relative to a corpus, which is ideal for highlighting unique emotional indicators.
3.  **Classic Machine Learning**: No modern word embeddings (like Word2Vec) were used as I was restricted to classic machine learning. TF-IDF is more suitable for these models as it provides high-dimensional scalar representations that models like SVC and Logistic Regression handle well.

## Why I chose these models
*   **Logistic Regression**: Excellent for multi-class problems using the **One-vs-Rest (OvR)** strategy. It is computationally efficient and provides clear probabilistic outputs.
*   **Linear SVC**: Performs exceptionally well with high-dimensional data (TF-IDF features) where the boundary between classes is relatively linear.
*   **Random Forest & XGBoost**: Robust ensemble methods used to reduce overfitting and capture non-linear patterns through decision trees and gradient boosting.

## Challenges & Solutions (Handling Imbalance)
The dataset initially suffered from a severe imbalance between classes like **Stress** and **Anxiety**. To solve this, I implemented a multi-stage process:
1.  **Dataset Merging**: I integrated an external dataset with similar labels to increase sample counts.
2.  **Data Augmentation**: I used `nltk` to perform synonym replacement, generating new sentences with the same meaning but different words.
3.  **Class Weights**: I applied the `class_weight='balanced'` parameter in model initialization. This forces the model to penalize errors on minority classes (like Stress) more heavily, improving their recall.

## Results
Below are the results evaluated on the final test set (total support: **7976**).

### Model Accuracy Summary
*   **Random Forest**: 77.96% **(Best Overall — Used in Production)**
*   **Linear SVC**: 77.13%
*   **Logistic Regression**: 76.18%
*   **XGBoost**: 76.71% (CPU Training)

### Detailed Classification Reports

**Label Mapping**: `0: Anxiety`, `1: Depression`, `2: Normal`, `3: Stress`, `4: Suicidal`

> [!WARNING]
> **XGBoost Performance Note**: The model initially showed degraded performance (**22.79%**) when trained on GPU. This was caused by an incompatibility between GPU-accelerated XGBoost (`device='cuda'`) and sparse matrices from TF-IDF inside an sklearn Pipeline, leading to class collapse. The results below reflect the stable performance achieved using **CPU training**.

#### 1. Random Forest (Optimized - Best Performance)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0: Anxiety** | 0.91 | 0.93 | 0.92 | 1634 |
| **1: Depression** | 0.67 | 0.66 | 0.67 | 2072 |
| **2: Normal** | 0.83 | 0.93 | 0.88 | 1263 |
| **3: Stress** | 0.95 | 0.82 | 0.88 | 1009 |
| **4: Suicidal** | 0.67 | 0.66 | 0.67 | 1998 |

#### 2. Linear SVC
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0: Anxiety** | 0.90 | 0.91 | 0.91 | 1634 |
| **1: Depression** | 0.68 | 0.63 | 0.65 | 2072 |
| **2: Normal** | 0.86 | 0.93 | 0.89 | 1263 |
| **3: Stress** | 0.79 | 0.83 | 0.81 | 1009 |
| **4: Suicidal** | 0.68 | 0.67 | 0.68 | 1998 |

#### 3. Logistic Regression
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0: Anxiety** | 0.89 | 0.85 | 0.87 | 1634 |
| **1: Depression** | 0.70 | 0.63 | 0.66 | 2072 |
| **2: Normal** | 0.83 | 0.94 | 0.88 | 1263 |
| **3: Stress** | 0.71 | 0.79 | 0.75 | 1009 |
| **4: Suicidal** | 0.70 | 0.69 | 0.70 | 1998 |

#### 4. XGBoost (CPU Training)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0: Anxiety** | 0.82 | 0.89 | 0.85 | 1634 |
| **1: Depression** | 0.72 | 0.67 | 0.69 | 2072 |
| **2: Normal** | 0.79 | 0.94 | 0.86 | 1263 |
| **3: Stress** | 0.92 | 0.49 | 0.64 | 1009 |
| **4: Suicidal** | 0.71 | 0.80 | 0.75 | 1998 |

## Datasets

### Primary Dataset
The main training dataset used in this project is sourced from Kaggle:

**[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)**

The original dataset file is also available directly via Google Drive:

**[Download Original Dataset](https://drive.google.com/file/d/1jwK34dhkgB9qO4o_568m79jKgKmsG4ty/view?usp=sharing)**

### Augmented Dataset
An additional dataset was merged with the original to address class imbalance (particularly for the **Stress** class), increasing overall sample counts and improving model generalization.

## Trained Models

All trained model files (`.pkl`) are available for download from Google Drive:

**[Download All Models](https://drive.google.com/file/d/1sbcgoQyGFcws1be0lImgyFAldmH-mL89/view?usp=sharing)**

The archive includes:
*   `best_Random_Forest.pkl` — Primary production model (best accuracy: **77.96%**)
*   `label_encoder.pkl` — Label mapping (0–4 to class names)
*   `other_models/` — XGBoost, Logistic Regression, and Linear SVC weights

> [!NOTE]
> The **Random Forest** model is the recommended model for production use. It achieves the highest overall accuracy and the best balance of precision and recall across all five classes.

## Project Structure
```text
├── app.py                            # Streamlit web application
├── best_Random_Forest.pkl             # Trained Random Forest model (Primary)
├── label_encoder.pkl                 # Label mapping (0-4 to text)
├── requirements.txt                  # Project dependencies
├── sentiment-analysis-for-mental-health.ipynb # Training & EDA notebook
├── other_models/                     # Optimized weights for other models (XGBoost, Logistic Regression, Linear SVC)
└── README.md                         # Project documentation
```

## Demo

### Video Demo
A live demonstration of the application in action is available on Google Drive:

**[Watch Demo Video](https://drive.google.com/file/d/1JIrlMyw_MzJwaZs2lSErvpGVID9eMzfF/view?usp=drive_link)**

### Run Locally
The project includes a Streamlit web interface.
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app:
```bash
streamlit run app.py
```
3.  The application uses the **Random Forest** model for real-time predictions.
