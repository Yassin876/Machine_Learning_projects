# Email Spam Classifier API

This is an API service built with FastAPI that classifies emails into "spam" or "not spam" using a pre-trained machine learning model.

## Project Files
- `app.py`: The FastAPI application containing the API endpoint and logic to preprocess text and predict its class.
- `model_emails.pkl`: The serialized (joblib) pre-trained machine learning model used for prediction.
- `emails.csv`: The original dataset. Its columns are used to map the words of incoming text to the feature space the model expects.
- `spam.jpg` / `not_spam.jpg`: Image assets related to the classification outputs.

## How it Works
The application exposes a `POST` endpoint `/predict`. When an email text is submitted:
1. The text is split into words and the word frequencies are counted.
2. This frequency map is transformed into a dataframe aligning with the original features (columns from `emails.csv`).
3. The dataset is passed to the loaded model (`model_emails.pkl`) for prediction.
4. The service returns a JSON response indicating whether the email is `spam` or `not spam`.

## Usage

### Prerequisites
Make sure you have the following packages installed:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

### Running the API
To start the FastAPI server, run this command in your terminal:
```bash
uvicorn app:app --reload
```
By default, the API will be hosted on `http://127.0.0.1:8000`.

### Making a Prediction
You can test the API by sending a POST request to `/predict`. An example using `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "Congratulations! You have won a free ticket. Click here to claim your prize."
}'
```

**Example Response:**
```json
{
  "prediction": "spam"
}
```

## Note
- The application loads the `.pkl` and `.csv` files into memory globally when the script is run. This keeps inference fast but requires sufficient memory for `emails.csv`.
- The `emails.csv` file should remain in the root folder alongside the script.
