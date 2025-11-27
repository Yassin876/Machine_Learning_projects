# Loan Default Classifier

This project is an interactive web application for predicting loan default risk using a machine learning model.

## Overview

The app allows users to input various loan and applicant details, then predicts whether the specified case is likely to **Default** or **Not Default**. The backend is built using a trained classifier (scikit-learn), and the frontend uses Streamlit for an easy-to-use UI.

## Features
- Predict loan default for custom applicant and loan data
- All model features are available as input via dropdowns or number fields
- Automatic value validation and type checking (no manual feature engineering needed)
- Clear explanations/hints for every input field

## Technologies Used
- Python 3
- scikit-learn (machine learning, model training)
- joblib (model serialization)
- pandas (data processing)
- Streamlit (web UI)

## Main Files
- `app.py`: Main Streamlit application (UI and prediction logic)
- `default_model_full.pkl`: Pre-trained model file (includes classifier, scaler, and label encoders)
- `README.md`: Project documentation

## How It Works
- The app loads the trained model (`default_model_full.pkl`) on startup
- The user fills all required loan/applicant data in the web form
- Input fields match exactly the modeled features (with dropdowns for categorical columns and number inputs for numerical ones)
- Each input is type-checked and explained with a contextual help message
- After clicking **Predict**, the model returns if the case is likely "Default" or "Not Default"

## How To Run
1. Install the required packages:
    ```bash
    pip install streamlit scikit-learn pandas joblib
    ```

2. Place all necessary files (`app.py`, `default_model_full.pkl`, `README.md`) in the same folder.

3. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Access the app in your browser (default: http://localhost:8501) and enter your loan/applicant data for prediction.

## Prerequisites
- Python 3 (recommended 3.8+)
- The following libraries: `streamlit`, `scikit-learn`, `pandas`, `joblib`
- The trained model file (`default_model_full.pkl`) present in the project directory

## Notes
- Input fields provide hints to explain each data entry required.
- You can retrain or update the model in Jupyter and export a new `default_model_full.pkl`.
- This app is for demonstration and prototyping. For production, review security and input validation in more depth.

---

**Author:** [Yassin Ahmed]
