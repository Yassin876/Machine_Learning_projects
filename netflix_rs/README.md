# Netflix Movie & TV Show Recommendation System

## Project Overview

This project is a recommendation system for Netflix movies and TV shows using AI (Natural Language Processing). You can enter any information about a movie or TV show (name, actor, director, genre, etc.), and the system will suggest the top 5 similar titles based on your input.

## Features
- Interactive interface using Streamlit.
- Uses TF-IDF and Cosine Similarity to generate recommendations.
- Supports searching with any available information about the title.

## Main Files
- `app.py`: The main application code.
- `preprocessing_data.csv`: Preprocessed Netflix data.
- `tfidf_vectorizer.pkl` & `tfidf_matrix.pkl`: Saved model files.
- `netflix_titles.csv`: The original Netflix dataset.
- `RSMovie.ipynb`: Notebook showing the analysis, preprocessing, and model building steps.

## How to Run
1. Make sure you have Python 3.7 or newer installed.
2. Install the required packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the link that appears in your browser.

## Main Data Columns
- title: Title name
- type: Type (Movie/TV Show)
- director: Director
- cast: Cast
- country: Country
- release_year: Release year
- rating: Age rating
- duration: Duration
- listed_in: Genre/Category
- description: Description

## Example Usage
- Type: "Will Smith" or "action 2019" or a movie name.
- Click the "Show Recommendations" button to see the top suggestions.

## Notes
- The more accurate your input, the better the recommendations.
- The system works locally and does not require an internet connection after downloading the files.

---

This project was developed for educational and experimental purposes. 