import streamlit as st
import re
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the pre-trained model and other necessary components
model = joblib.load('models/Mental_Health.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

nltk.download('stopwords')
ps=PorterStemmer()


# Function to clean the input text
def clean_text(text):
  text=re.sub('[^a-zA-Z]',' ',text)
  text=text.lower()
  text=text.split()
  text=[ps.stem(word) for word in text if not word in stopwords.words('english')] #apply stemming and remove stop words
  text=' '.join(text)#back to string    
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
            input_vector = tfidf_vectorizer.transform([cleaned_input])# Transform the cleaned input using the same TF-IDF vectorizer
            prediction = model.predict(input_vector)# Predict the mental health condition using the pre-trained model
            print(prediction) # Debugging: Print the raw prediction output
            predicted_label = label_encoder.inverse_transform(prediction)[0]# Decode the predicted label
            st.success(f"Predicted Mental Health Condition: {predicted_label}")
        else: # If the user input is empty, show a warning message
            st.warning("Please enter some text to predict.")


if __name__ == "__main__":
    main()