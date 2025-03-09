import re
import csv
import time
import random
import requests
import joblib
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Base URL for Flipkart reviews
base_url = "https://www.flipkart.com/nutrabay-pure-100-whey-protein-concentrate-raw/p/itmd400ebaf7afc0?pid=PSLFHNGCZFFAPT83&lid=LSTPSLFHNGCZFFAPT83AFE6CU&marketplace=FLIPKART&q=protien+powder+nutrabay&store=hlc%2Fetg%2F1rx&srno=s_1_10&otracker=search&otracker1=search&fm=Search&iid=4967777d-ac69-4f2d-a455-0d75348fb831.PSLFHNGCZFFAPT83.SEARCH&ppt=sp&ppn=sp&ssid=sxqlt9ojo00000001741276363969&qH=7ca282fc242ad4e3"
base_url = base_url.replace('/p/', '/product-reviews/')

# Sorting orders for different review pages
sorting_orders = [
    "",  # Default (unsorted)
    "&aid=overall&certifiedBuyer=false&sortOrder=MOST_HELPFUL",
    "&aid=overall&certifiedBuyer=false&sortOrder=MOST_RECENT",
    "&aid=overall&certifiedBuyer=false&sortOrder=POSITIVE_FIRST",
    "&aid=overall&certifiedBuyer=false&sortOrder=NEGATIVE_FIRST"
]

# Function to scrape ratings distribution with retry mechanism
def get_rating_distribution(url, retries=4):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            sleep(1)  # Wait for a second before making the request

            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                rating_distribution = []

                # Extracting ratings count for each rating
                ratings = soup.find_all('div', class_='x2IWCo')
                counts = soup.find_all('div', class_='BArk-j')

                for i in range(5):
                    rating = int(ratings[i].get_text(strip=True).split('★')[0])
                    count = int(counts[i].get_text(strip=True).replace(',', ''))
                    rating_distribution.append(count)

                # Print the scraped ratings distribution for verification
                print("Scraped Rating Distribution:")
                for i, count in enumerate(rating_distribution):
                    print(f"{5-i}★: {count} reviews")

                return rating_distribution  # Return if successful
            
            else:
                print(f"Failed to retrieve rating distribution. Status Code: {response.status_code}")
                raise Exception("Failed to fetch ratings")

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == retries:
                print("Max retries reached. Could not retrieve ratings. Returning all zeros.")
                return [0, 0, 0, 0, 0]  # Return all zeros if failed after 4 attempts
            sleep(random.uniform(1, 3))  # Sleep for a random time between retries to avoid hitting the server too quickly

    return [0, 0, 0, 0, 0]  # Return all zeros if failed after all attempts


# Generate review URLs
def generate_review_urls(base_url, sorting_orders):
    return [base_url + order for order in sorting_orders]

# List of URLs to scrape
urls = generate_review_urls(base_url, sorting_orders)

# List to hold reviews
all_reviews = []
seen_reviews = set()

# Function to scrape Flipkart reviews
def get_flipkart_reviews(url):
    total_pages = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    page_number = 1
    while True:
        # Append page number to the URL
        page_url = f"{url}&page={page_number}"
        response = requests.get(page_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to retrieve page {page_number} from {url}")
            break

        # Parse the page content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all reviews
        reviews = soup.find_all('div', class_='col EPCmJX Ma1fCG')

        if not reviews:
            print(f"No more reviews found on page {page_number} from {url}. Moving to the next URL.")
            break

        # Extract review text and rating
        for review in reviews:
            # Extract review text
            review_content = review.find('div', class_='ZmyHeo').get_text(separator=' ', strip=True)

            # Extract the rating (star rating) using regex for dynamic class names
            rating_div = review.find('div', class_=re.compile(r'^XQDdHH.*Ga3i8K$'))
            rating = None
            if rating_div:
                rating = rating_div.get_text(strip=True)
                try:
                    rating = float(rating)
                except ValueError:
                    rating = None

            # Check if "READ MORE" is present in the review
            read_more = review.find('span', class_='wTYmpv')
            if read_more:
                # Replace ' READ MORE' with a period ('.')
                review_content = review_content.split('READ MORE')[0].strip() + "."

            # If the (review_content, rating) pair is unique, add it to the list
            review_pair = (review_content, rating)
            if review_pair not in seen_reviews:
                all_reviews.append([review_content, rating])  # Add to list
                seen_reviews.add(review_pair)  # Add the review pair to the set
                print(f"Review: {review_content}\nRating: {rating}\n")
            else:
                print(f"Duplicate review found and skipped.")

        # Go to the next page
        page_number += 1
        total_pages += 1

        # Sleep after every 5 pages to avoid overloading the server
        if page_number % 2 == 0:
            sleep(1.5)

    print(f"Total pages scraped from {url}: {total_pages}")
    
# Scrape ratings distribution
ratings_distribution = get_rating_distribution(base_url)

# Scrape reviews from all URLs
for url in urls:
    get_flipkart_reviews(url)

# Load Machine Learning Models
def load_model(path, name):
    start_time = time.time()
    model = joblib.load(path)
    print(f"{name} loaded in {time.time() - start_time:.4f} seconds")
    return model

logistic_model = load_model(r"backend\Models\logistic_regression_model.joblib", "Logistic Regression Model")
random_forest_model = load_model(r"backend\Models\random_forest_model.joblib", "Random Forest Model")
svm_model = load_model(r"backend\Models\svm_model.joblib", "SVM Model")
stacking_model = load_model(r"backend\Models\xgb_stacking_model.joblib", "Stacking Model")

# Convert reviews to DataFrame
df_test = pd.DataFrame(all_reviews, columns=['text', 'rating'])

# Make predictions
X_test = df_test[['text', 'rating']]
logistic_pred = logistic_model.predict_proba(X_test)[:, 1]
rf_pred = random_forest_model.predict_proba(X_test)[:, 1]
svm_pred = svm_model.predict_proba(X_test)[:, 1]

# Stacking Model Prediction
X_stack = pd.DataFrame({'logistic_pred': logistic_pred, 'rf_pred': rf_pred, 'svm_pred': svm_pred})
df_test['final_prediction'] = stacking_model.predict_proba(X_stack)[:, 1] * 100

# Save to CSV
df_test[['text', 'rating', 'final_prediction']].to_csv('predicted_reviews.csv', index=False)
print("Predictions saved to 'predicted_reviews.csv'.")


#----------------------------------------------------------------------------------------
# SENTIMENT ANALYSIS

# Initialize VADER SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Custom lexicon adjustments for domain-specific terms
custom_lexicon = {
    # Negative Sentiments
    'fraud': -4.0,  
    'scam': -4.5,
    'defective': -4.5,
    'poor': -3.5,
    'bad': -3.0,
    'disappointing': -3.5,
    'hidden': -3.0,
    'charges': -2.0,
    
    # Positive Sentiments
    'excellent': 3.5,
    'responsive': 2.5,
    'quality': 3.0,
    'good': 2.5,
    'worth': 4.0,
    'best': 4.5,
    'love': 4.0,
    
    # Neutral Sentiments
    'okay': 0.005,
    'average': 0.005,
    'satisfactory': 0.005
}

analyzer.lexicon.update(custom_lexicon)

def load_and_preprocess_data(filepath):
    """Load and preprocess data with enhanced cleaning"""
    df_test = pd.read_csv(filepath)
    
    # Enhanced text cleaning
    df_test['cleaned_text'] = df_test['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))
    df_test['cleaned_text'] = df_test['cleaned_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    return df_test

# Load and preprocess data
df_test = load_and_preprocess_data('predicted_reviews.csv')

# Filter reviews for analysis
analysis_df = df_test[df_test['final_prediction'] < 60].copy()

# Enhanced sentiment scoring with negation handling
def enhanced_sentiment(text):
    # Handle negations explicitly
    text = re.sub(r"\b(not|never|no)\b[\w\s]+", lambda match: match.group().replace(" ", "_"), text, flags=re.IGNORECASE)
    sentiment = analyzer.polarity_scores(text)
    
    # Add intensity weighting
    intensity = 1.0 + abs(sentiment['compound']) * 0.5
    return sentiment['compound'] * intensity

analysis_df['sentiment_score'] = analysis_df['cleaned_text'].apply(enhanced_sentiment)

# Normalize to 0-100 scale
analysis_df['normalized_sentiment'] = (analysis_df['sentiment_score'] + 1) * 50

# Scraped rating distribution (use actual data here)
# ratings_distribution = np.array([3058, 1350, 573, 204, 403])  # Example: [5-star, 4-star, 3-star, 2-star, 1-star]

# Normalize the rating distribution to create **weights**
total_reviews = sum(ratings_distribution)
normalized_distribution = [x / total_reviews for x in ratings_distribution]

# Convert Sentiment Scores to Ratings (1-5 Scale)
analysis_df['rating'] = pd.cut(analysis_df['normalized_sentiment'], bins=5, labels=[5, 4, 3, 2, 1]).astype(int)

# Adjust Sentiment Dynamically Based on Ratings
def adjust_sentiment(row):
    sentiment = row['normalized_sentiment']
    rating_index = 5 - row['rating']  # Convert rating (1-5) to index (0-4)
    
    # Apply **non-linear weighting**: Give **higher impact to lower ratings**
    weighted_sentiment = sentiment * (1 + 0.3 * (1 - normalized_distribution[rating_index]))

    return np.clip(weighted_sentiment, 0, 100)

# Apply Adjusted Sentiment Calculation
analysis_df['adjusted_sentiment'] = analysis_df.apply(adjust_sentiment, axis=1)

# Sentiment categorization
def categorize_sentiment(score):
    if score >= 70: return 'Positive'
    elif score >= 40: return 'Neutral'
    else: return 'Negative'

analysis_df['sentiment_category'] = analysis_df['adjusted_sentiment'].apply(categorize_sentiment)

# ----------------------------------------------------------------------------------------
# CALCULATE & PRINT SENTIMENT PERCENTAGES

# Get counts for each sentiment category
sentiment_counts = analysis_df['sentiment_category'].value_counts(normalize=True) * 100

# Extract individual percentages
positive_percentage = sentiment_counts.get('Positive', 0)
neutral_percentage = sentiment_counts.get('Neutral', 0)
negative_percentage = sentiment_counts.get('Negative', 0)

# Print the final sentiment breakdown
print(f"Sentiment Analysis Breakdown:")
print(f"Positive Sentiment: {positive_percentage:.2f}%")
print(f"Neutral Sentiment: {neutral_percentage:.2f}%")
print(f"Negative Sentiment: {negative_percentage:.2f}%")
