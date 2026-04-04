# Yelp Review Sentiment Analysis App

A Streamlit app for analyzing Yelp reviews with NLP and logistic regression.

## Features
- Upload a CSV of Yelp reviews
- Supports binary or three-class sentiment prediction
- Text preprocessing with Python and TextBlob
- TF-IDF + logistic regression using scikit-learn
- Predict sentiment for new reviews in the browser

## Expected CSV format
Your file must contain these columns:
- `text`
- `stars`

Example:
```csv
text,stars
"The food was excellent and service was great",5
"Terrible experience, very rude staff",1
"It was average, nothing special",3
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Click **Create app**.
4. Select your GitHub repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Tech stack
- Python
- Streamlit
- TextBlob
- scikit-learn
