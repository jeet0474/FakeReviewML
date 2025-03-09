import re
import time
import random
import requests
import joblib
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "ok"})

# Load ML models
logistic_model = joblib.load("./Models/logistic_regression_model.joblib")
random_forest_model = joblib.load("./Models/random_forest_model.joblib")
svm_model = joblib.load("./Models/svm_model.joblib")
stacking_model = joblib.load("./Models/xgb_stacking_model.joblib")

# Flipkart product reviews sorting orders
sorting_orders = [
    "",  # Default order
    "&aid=overall&certifiedBuyer=false&sortOrder=MOST_HELPFUL"
]

# Function to get rating distribution
def get_rating_distribution(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"5_star": 0, "4_star": 0, "3_star": 0, "2_star": 0, "1_star": 0}

    soup = BeautifulSoup(response.content, 'html.parser')
    ratings = soup.find_all('div', class_='x2IWCo')
    counts = soup.find_all('div', class_='BArk-j')

    if len(ratings) < 5 or len(counts) < 5:
        return {"5_star": 0, "4_star": 0, "3_star": 0, "2_star": 0, "1_star": 0}

    return {
        f"{5-i}_star": int(counts[i].get_text(strip=True).replace(',', '')) for i in range(5)
    }

# Function to scrape reviews
def get_flipkart_reviews(url):
    if not url:
        return []

    # Convert product page URL to reviews URL (Flipkart-specific)
    if "/p/" in url:
        url = url.replace('/p/', '/product-reviews/')

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    all_reviews = []
    seen_reviews = set()

    page_number = 1
    max_pages = 5  # Limit scraping to 5 pages to avoid being blocked

    while page_number <= max_pages:
        page_url = f"{url}&page={page_number}"
        
        try:
            response = requests.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an error if request fails
        except requests.exceptions.RequestException as e:
            print(f"Error fetching reviews: {e}")
            break  # Stop if request fails

        soup = BeautifulSoup(response.text, 'html.parser')

        reviews = soup.find_all('div', class_='col EPCmJX Ma1fCG')  # Flipkart review container
        if not reviews:
            break  # Stop if no more reviews found

        for review in reviews:
            # Extract review content
            review_content = review.find('div', class_='ZmyHeo')
            review_text = review_content.get_text(strip=True) if review_content else ""

            # Extract rating
            rating_div = review.find('div', class_=re.compile(r'^XQDdHH.*Ga3i8K$'))
            rating = None
            if rating_div:
                try:
                    rating = float(rating_div.get_text(strip=True))
                except ValueError:
                    rating = None

            # Ensure unique reviews
            review_pair = (review_text, rating)
            if review_pair not in seen_reviews:
                all_reviews.append({"text": review_text, "rating": rating})
                seen_reviews.add(review_pair)

        page_number += 1
        time.sleep(random.uniform(1, 3))  # Random delay to prevent getting blocked

    return all_reviews 


# Predict AI-generated reviews
def predict_fake_reviews(reviews):
    print(f"Received Reviews: {len(reviews)}")
    df = pd.DataFrame(reviews)
    if df.empty:
        print("❌ DataFrame is empty. Returning []")
        return []

    try:
        logistic_pred = logistic_model.predict_proba(df[['text', 'rating']])[:, 1]
        rf_pred = random_forest_model.predict_proba(df[['text', 'rating']])[:, 1]
        svm_pred = svm_model.predict_proba(df[['text', 'rating']])[:, 1]

        X_stack = pd.DataFrame({'logistic_pred': logistic_pred, 'rf_pred': rf_pred, 'svm_pred': svm_pred})
        df['prediction'] = stacking_model.predict_proba(X_stack)[:, 1] * 100

        return df.to_dict(orient="records") 
    except Exception as e:
        print(f"Prediction Error: {e}")
        return []

# Perform sentiment analysis
# def analyze_sentiment(reviews, ratings_distribution):
#     analyzer = SentimentIntensityAnalyzer()
#     sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

#     for review in reviews:
#         sentiment = analyzer.polarity_scores(review["text"])["compound"]
#         category = "positive" if sentiment >= 0.05 else "neutral" if sentiment > -0.05 else "negative"
#         sentiment_counts[category] += 1

#     total = sum(sentiment_counts.values())

#     if total == 0:
#         return {"positive": 0, "neutral": 0, "negative": 0}

#     return {key: round((value / total) * 100, 2) for key, value in sentiment_counts.items()}

def analyze_sentiment(reviews, ratings_distribution):
    analyzer = SentimentIntensityAnalyzer()

    # Custom lexicon updates for domain-specific words
    custom_lexicon = {
        # Strong Negative Sentiments
        "abysmal": -3.5, "appalling": -3.7, "atrocious": -3.9, "awful": -3.0, "terrible": -3.5,
        "horrible": -3.8, "dreadful": -3.6, "disastrous": -3.9, "pathetic": -3.2, "lousy": -3.0,
        "outrageous": -3.4, "unacceptable": -3.3, "inferior": -3.1, "subpar": -2.8, "unreliable": -3.0,
        "incompetent": -3.5, "negligent": -3.6, "untrustworthy": -3.7, "deceptive": -3.8, "scandalous": -3.9,

        # Moderate Negative Sentiments
        "disappointing": -2.5, "unsatisfactory": -2.7, "mediocre": -2.0, "poor": -2.8, "substandard": -2.9,
        "flawed": -2.6, "problematic": -2.4, "deficient": -2.7, "lacking": -2.3, "unimpressive": -2.2,
        "underwhelming": -2.1, "regrettable": -2.5, "unfortunate": -2.4, "disturbing": -2.8, "troubling": -2.6,
        "concerning": -2.3, "questionable": -2.5, "dissatisfactory": -2.9, "unsuitable": -2.7, "undesirable": -2.6,

        # Mild Negative Sentiments
        "inconvenient": -1.5, "annoying": -1.7, "frustrating": -1.8, "bothersome": -1.6, "irritating": -1.7,
        "unpleasant": -1.9, "displeasing": -1.8, "uncomfortable": -1.6, "awkward": -1.5, "unfavorable": -1.7,
        "unappealing": -1.8, "unattractive": -1.6, "uninviting": -1.5, "unwelcoming": -1.7, "unfriendly": -1.9,
        "cold": -1.5, "distant": -1.6, "aloof": -1.7, "apathetic": -1.8, "indifferent": -1.9,

        # Neutral Sentiments
        "average": 0.0001, "mediocre": 0.0001, "fair": 0.0001, "moderate": 0.0001, "satisfactory": 0.0001,
        "adequate": 0.0001, "sufficient": 0.0001, "acceptable": 0.0001, "tolerable": 0.0001, "passable": 0.0001,
        "standard": 0.0001, "ordinary": 0.0001, "common": 0.0001, "typical": 0.0001, "regular": 0.0001,
        "normal": 0.0001, "unremarkable": 0.0001, "plain": 0.0001, "simple": 0.0001, "basic": 0.0001,

        # Mild Positive Sentiments
        "pleasant": 1.5, "nice": 1.7, "agreeable": 1.6, "enjoyable": 1.8, "delightful": 1.9,
        "pleasing": 1.7, "satisfying": 1.6, "gratifying": 1.8, "refreshing": 1.9, "comforting": 1.7,
        "welcoming": 1.8, "friendly": 1.9, "warm": 1.6, "cordial": 1.7, "amiable": 1.8,
        "affable": 1.9, "genial": 1.6, "kind": 1.7, "considerate": 1.8, "thoughtful": 1.9,

        # Moderate Positive Sentiments
        "good": 2.5, "great": 2.7, "excellent": 2.9, "fantastic": 3.0, "wonderful": 2.8,
        "superb": 2.9, "outstanding": 3.0, "remarkable": 2.7, "impressive": 2.6, "exceptional": 2.8,
        "brilliant": 2.9, "amazing": 3.0, "marvelous": 2.8, "fabulous": 2.7, "terrific": 2.9,
        "splendid": 2.6, "magnificent": 2.8, "phenomenal": 3.0, "extraordinary": 2.9, "stellar": 2.7,

        # Strong Positive Sentiments
        "awesome": 3.5, "incredible": 3.7, "spectacular": 3.9, "breathtaking": 4.0, "astonishing": 3.8,
        "stunning": 3.9, "awe-inspiring": 4.0, "mind-blowing": 3.8, "unbelievable": 3.7, "unparalleled": 3.9,
        "unmatched": 3.8, "unrivaled": 3.7, "peerless": 3.9, "unsurpassed": 4.0, "transcendent": 3.8,
        "divine": 3.9, "heavenly": 4.0, "sublime": 3.8, "glorious": 3.9, "majestic": 4.0
    }

    analyzer.lexicon.update(custom_lexicon)

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    for review in reviews:
        text = review["text"].lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        text = re.sub(r"\b(not|never|no)\b[\w\s]+", lambda match: match.group().replace(" ", "_"), text)  # Handle negations

        sentiment = analyzer.polarity_scores(text)["compound"]
        category = "positive" if sentiment >= 0.05 else "neutral" if sentiment > -0.05 else "negative"
        sentiment_counts[category] += 1

    total = sum(sentiment_counts.values())
    
    if total == 0:
        return {"positive": 0, "neutral": 0, "negative": 0}

    # Normalize text-based sentiment
    text_sentiments = {key: value / total for key, value in sentiment_counts.items()}

    # Calculate rating-based sentiment distribution
    total_ratings = sum(ratings_distribution.values())
    
    if total_ratings > 0:
        rating_weights = {
            "positive": (ratings_distribution.get('5_star', 0) + ratings_distribution.get('4_star', 0)) / total_ratings,
            "neutral": ratings_distribution.get('3_star', 0) / total_ratings,
            "negative": (ratings_distribution.get('2_star', 0) + ratings_distribution.get('1_star', 0)) / total_ratings,
        }
    else:
        rating_weights = {"positive": 0, "neutral": 0, "negative": 0}

    # Merge text-based sentiment with rating-based sentiment
    final_sentiments = {
        key: round(((text_sentiments[key] * 0.5) + (rating_weights[key] * 0.5)) * 100, 2)
        for key in sentiment_counts
    }

    return final_sentiments


# Django API view
def predict_fake_reviews_api(request):
    try:
        url = request.GET.get("url")
        # print(f"Received URL: {url}")
        if not url:
            return JsonResponse({"error": "URL parameter is required"}, status=400)

        ratings_distribution = get_rating_distribution(url)
        print(f"Ratings: {ratings_distribution}")
        
        reviews = get_flipkart_reviews(url)
        # Remove "READ MORE" from all reviews
        for review in reviews:
            review["text"] = review["text"].replace("READ MORE", "").strip()
        # print(f"Fetched Reviews: {reviews}")

        if not reviews:
            return JsonResponse({
                "text": [],
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "rating_distribution": ratings_distribution
            }, safe=False)

        reviews_with_prediction = predict_fake_reviews(reviews)
        print(f"Predicted Fake Reviews: {len(reviews_with_prediction)}")

        sentiment_distribution = analyze_sentiment(reviews_with_prediction, ratings_distribution)
        print(f"Sentiment: {sentiment_distribution}")

        response_data = {
            "text": reviews_with_prediction,
            "sentiment_distribution": sentiment_distribution,
            "rating_distribution": ratings_distribution
        }

        return JsonResponse(response_data, safe=False)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
