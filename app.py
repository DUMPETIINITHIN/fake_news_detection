import streamlit as st
import pandas as pd
import re
import string
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection")
st.write("Machine Learning based Fake News Classifier with Real-Time News")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv", engine="python", on_bad_lines="skip")
    true = pd.read_csv("True.csv", engine="python", on_bad_lines="skip")

    fake["label"] = 0   # Fake
    true["label"] = 1   # Real

    df = pd.concat([fake, true], ignore_index=True)
    df = df[["text", "label"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

df = load_data()

# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer

model, vectorizer = train_model(df)

# --------------------------------------------------
# Real-Time News Fetch (NewsAPI)
# --------------------------------------------------
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

def fetch_live_news():
    if not NEWS_API_KEY:
        return None

    url = (
        "https://newsapi.org/v2/top-headlines?"
        "language=en&pageSize=1&apiKey=" + NEWS_API_KEY
    )
    response = requests.get(url)
    data = response.json()

    if data.get("status") == "ok" and data.get("articles"):
        article = data["articles"][0]
        text = article["title"]
        if article["description"]:
            text += ". " + article["description"]
        return text
    return None


# --------------------------------------------------
# Manual News Input
# --------------------------------------------------
st.subheader("📝 Enter News Text")

news_text = st.text_area(
    "Paste news article or headline",
    height=180
)

if st.button("🔍 Detect"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        if prediction == 1:
            st.success(f"✅ REAL News (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.error(f"🚨 FAKE News (Confidence: {probability[0]*100:.2f}%)")

# --------------------------------------------------
# Real-Time News Section
# --------------------------------------------------
st.markdown("---")
st.subheader("🌐 Real-Time News Detection")

if st.button("📡 Fetch Live News"):
    live_news = fetch_live_news()

    if live_news:
        st.text_area("Live News Article", live_news, height=150)

        cleaned = clean_text(live_news)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        if prediction == 1:
            st.success(f"✅ Predicted REAL (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.error(f"🚨 Predicted FAKE (Confidence: {probability[0]*100:.2f}%)")
    else:
        st.warning("Unable to fetch live news.")

# --------------------------------------------------
# Disclaimer
# --------------------------------------------------
st.markdown("---")
st.info(
    "⚠️ Disclaimer: This system predicts news authenticity based on historical patterns "
    "and should not be considered a fact-checking authority."
)
