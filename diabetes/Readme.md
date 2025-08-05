# Diabetes Prediction App

A simple web application using [Streamlit](https://streamlit.io/) to predict diabetes based on user input, powered by a pre-trained XGBoost model.

---

## Features

- User-friendly web interface for diabetes prediction
- Uses a machine learning model trained on the Pima Indians Diabetes Dataset
- Data preprocessing and model training steps are available in the included Jupyter Notebook

---

## Requirements

- Python 3.11
- numpy
- pandas
- scikit-learn
- xgboost
- joblib
- streamlit
- matplotlib, seaborn (for EDA in the notebook)

Install all requirements with:
```sh
pip install -r requirements.txt
```
Or manually:
```sh
pip install numpy pandas scikit-learn xgboost joblib streamlit matplotlib seaborn
```

---

## How to Run

1. Make sure the following files are in the same directory:
   - `app.py`
   - `scaler.pkl`
   - `model.pkl` or `diabetes_xgb_model.pkl` (as exported from the notebook)

2. Start the Streamlit app:
```sh
streamlit run app.py
```

3. Open the local URL provided in your terminal (usually: http://localhost:8501)

---

## Model Training

- The notebook `diabetes_notebook.ipynb` contains all steps for data cleaning, EDA, model training, and exporting the model and scaler.
- You can retrain the model or adjust hyperparameters as needed.

---

## Notes

- If you encounter library errors (especially with numpy or pandas), make sure all scientific libraries are updated together.
- This app is for educational purposes only and should not be used for real medical diagnosis.

---

## Author

Developed for