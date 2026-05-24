import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:    
    model = joblib.load('best_Random_Forest.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'best_Random_Forest.pkl' and 'label_encoder.pkl' are in the same directory as the app.")
    st.stop()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Function to clean the input text
def clean_text(text):
  text=re.sub('[^a-zA-Z]',' ',text)
  text=text.lower()
  text=word_tokenize(text)
  text=[lemmatizer.lemmatize(word) for word in text if not word in stop_words]
  text=' '.join(text)
  return text

# Streamlit app
def main():
    st.title("Mental Health Prediction")
    st.write("Enter your text to predict the mental health condition.")

    user_input = st.text_area("Enter your text here:")
# Predict button
    if st.button("Predict"):
        if user_input:#validate that the user input is not empty
            cleaned_input = clean_text(user_input)#apply the same cleaning process to the user input
            prediction = model.predict([cleaned_input])# Predict the mental health condition using the pre-trained model
            predicted_label = label_encoder.inverse_transform(prediction)[0]# Decode the predicted label
            st.success(f"Predicted Mental Health Condition: {predicted_label}")
        else: # If the user input is empty, show a warning message
            st.warning("Please enter some text to predict.")


if __name__ == "__main__":
    main()