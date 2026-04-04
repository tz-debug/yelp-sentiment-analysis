import re
import string
from io import StringIO

import pandas as pd
import streamlit as st
from textblob import TextBlob
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Yelp Review Sentiment Analysis", page_icon="⭐", layout="wide")


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_textblob_features(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def prepare_data(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "text" not in df.columns or "stars" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'stars' columns.")

    df = df[["text", "stars"]].dropna().copy()
    df["clean_text"] = df["text"].apply(clean_text)
    feats = df["clean_text"].apply(get_textblob_features)
    df["polarity"] = feats.apply(lambda x: x[0])
    df["subjectivity"] = feats.apply(lambda x: x[1])

    if mode == "Binary":
        df = df[df["stars"] != 3].copy()
        df["sentiment"] = df["stars"].apply(lambda s: "negative" if s <= 2 else "positive")
    else:
        def label(stars: int) -> str:
            if stars <= 2:
                return "negative"
            if stars == 3:
                return "neutral"
            return "positive"
        df["sentiment"] = df["stars"].apply(label)
    return df


def train_model(df: pd.DataFrame):
    X = df[["clean_text", "polarity", "subjectivity"]]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "clean_text"),
            ("num", StandardScaler(), ["polarity", "subjectivity"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, report, cm, X_test, y_test, y_pred


def predict_reviews(model, reviews):
    temp = pd.DataFrame({"text": reviews})
    temp["clean_text"] = temp["text"].apply(clean_text)
    feats = temp["clean_text"].apply(get_textblob_features)
    temp["polarity"] = feats.apply(lambda x: x[0])
    temp["subjectivity"] = feats.apply(lambda x: x[1])
    X_new = temp[["clean_text", "polarity", "subjectivity"]]
    temp["prediction"] = model.predict(X_new)
    return temp[["text", "prediction", "polarity", "subjectivity"]]


st.title("Yelp Review Sentiment Analysis")
st.write(
    "Upload a CSV with `text` and `stars` columns, train a logistic regression sentiment model, and test new reviews in the browser."
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Classification mode", ["Binary", "Three-class"], help="Binary drops 3-star reviews and often yields higher accuracy.")
    st.markdown("**Expected CSV columns**: `text`, `stars`")

uploaded_file = st.file_uploader("Upload Yelp review CSV", type=["csv"])

sample_csv = """text,stars
The food was excellent and service was great,5
Terrible experience and very rude staff,1
It was average and nothing special,3
Amazing burgers and lovely ambience,5
I waited forever and the food was cold,1
Pretty decent place for a quick lunch,4
"""

if uploaded_file is None:
    st.info("No file uploaded yet. You can use the sample data below to test the app.")
    if st.button("Use sample dataset"):
        uploaded_file = StringIO(sample_csv)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        processed = prepare_data(df, mode)

        st.subheader("Dataset Preview")
        st.dataframe(processed.head(10), use_container_width=True)

        if processed["sentiment"].nunique() < 2:
            st.error("The dataset does not contain enough classes to train a model.")
        else:
            model, acc, report, cm, X_test, y_test, y_pred = train_model(processed)

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.2%}")
            c2.metric("Reviews used", len(processed))
            c3.metric("Classes", processed["sentiment"].nunique())

            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(cm)
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("Try New Reviews")
            default_text = "The food was amazing and the staff were friendly.\nService was slow and the order was wrong."
            user_reviews = st.text_area("Enter one review per line", value=default_text, height=150)
            if st.button("Predict sentiment"):
                reviews = [line.strip() for line in user_reviews.splitlines() if line.strip()]
                if reviews:
                    preds = predict_reviews(model, reviews)
                    st.dataframe(preds, use_container_width=True)
                else:
                    st.warning("Please enter at least one review.")

            st.subheader("What this app does")
            st.markdown(
                "- Cleans review text\n"
                "- Extracts TextBlob polarity and subjectivity\n"
                "- Builds TF-IDF features\n"
                "- Trains a logistic regression classifier\n"
                "- Predicts sentiment for new reviews"
            )

    except Exception as e:
        st.error(f"Error: {e}")
