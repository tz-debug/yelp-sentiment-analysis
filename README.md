# Yelp Review Sentiment Analysis

A deployed machine learning application for analyzing Yelp reviews and predicting sentiment using natural language processing techniques.

---

## Live Demo

https://yelp-sentiment-analysis-project.streamlit.app/

---

## Overview

This project presents an interactive sentiment analysis system built with Python and deployed using Streamlit. The application allows users to upload review datasets or input custom text to classify sentiment based on learned patterns in customer feedback.

Sentiment analysis is widely used to understand customer opinions, where text is categorized into positive, negative, or neutral sentiment based on linguistic patterns. ([LinkedIn][1])

---

## Features

* Upload Yelp-style datasets with review text and ratings

* Predict sentiment using machine learning models

* Support for:

  * Binary classification (positive / negative)
  * Multi-class classification (positive / neutral / negative)

* Natural language preprocessing:

  * Text cleaning
  * Token normalization
  * Feature extraction

* Model pipeline:

  * TF-IDF vectorization
  * Logistic Regression classifier

* Interactive interface:

  * Real-time predictions
  * User input for custom reviews

---

## Approach

The system follows a standard NLP pipeline:

1. Text preprocessing (cleaning, normalization)
2. Feature extraction using TF-IDF
3. Sentiment classification using Logistic Regression
4. Evaluation using accuracy and classification metrics

Streamlit is used to create an interactive interface that allows real-time sentiment prediction and visualization. ([Medium][2])

---

## Results

* Achieved approximately **87% accuracy** on sentiment classification
* Demonstrates strong performance for a lightweight machine learning model
* Provides fast and interpretable predictions suitable for real-time applications

---

## Tech Stack

* Python
* scikit-learn
* pandas, numpy
* TextBlob
* Streamlit

---

## Use Cases

* Customer feedback analysis
* Review monitoring for businesses
* Educational demonstration of NLP pipelines
* Rapid prototyping of sentiment analysis systems

---

## Limitations

* Performance depends on dataset quality and preprocessing
* Limited contextual understanding compared to deep learning models
* May misclassify sarcasm or ambiguous language

---

## Future Improvements

* Integration of transformer-based models (e.g., BERT)
* Advanced text preprocessing (lemmatization, embeddings)
* Model comparison and hyperparameter tuning
* Visualization dashboard for sentiment trends

---

## Author

Developed as part of a machine learning and NLP portfolio project, demonstrating end-to-end model development, deployment, and interactive analysis.

[1]: https://www.linkedin.com/pulse/project-yelp-reviews-sentiment-analysis-tanja-ad%C5%BEi%C4%87?utm_source=chatgpt.com "project | Yelp Reviews Sentiment Analysis"
[2]: https://medium.com/%40kayamericengin/from-raw-reviews-to-real-insights-building-a-sentiment-analysis-app-with-python-streamlit-d54dd4fb6c89?utm_source=chatgpt.com "Building a Sentiment Analysis App with Python & Streamlit"
