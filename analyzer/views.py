from django.shortcuts import render

# Create your views here.
import joblib
from django.shortcuts import render
import re
import string

# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def predict_tweet(tweet):
    cleaned = clean_text(tweet)
    X_tweet = vectorizer.transform([cleaned])
    prediction = model.predict(X_tweet)[0]

    if prediction == 1:
        return "Positive Sentiment", "positive"
    elif prediction == -1:
        return "Negative Sentiment", "negative"
    else:
        return "Neutral Sentiment", "neutral"

def home(request):
    sentiment = None
    category = None

    if request.method == "POST":
        text = request.POST.get("tweet")
        sentiment, category = predict_tweet(text)

    return render(request, "analyzer/index.html", {"sentiment": sentiment, "category": category})
