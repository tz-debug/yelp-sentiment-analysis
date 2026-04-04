# Yelp Review Sentiment Analysis App

A Streamlit app for Yelp review sentiment analysis using NLP.

## Features
- Upload a CSV with `text` and `stars` columns
- Use a built-in sample dataset for quick testing
- Supports binary and three-class sentiment modes
- Uses TextBlob features, TF-IDF, and logistic regression
- Predicts sentiment for new review text in the browser

## Files
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `sample_yelp_reviews.csv` - built-in sample dataset

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit
1. Push these files to a GitHub repository
2. Go to Streamlit Community Cloud
3. Create a new app
4. Select your repository and set the main file to `app.py`

## Dataset format
Your uploaded CSV should contain:
- `text`
- `stars`

## Notes
The app includes safer train/test splitting and a larger sample dataset to avoid small-dataset errors.
