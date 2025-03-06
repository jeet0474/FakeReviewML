import re
import csv
import time
import requests
import joblib
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    
    page_number = 1
    while True:
        page_url = f"{url}&page={page_number}"
        response = requests.get(page_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page_number}")
            break
        
        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = soup.find_all('div', class_='col EPCmJX Ma1fCG')
        
        if not reviews:
            print(f"No more reviews on page {page_number}. Moving to next URL.")
            break
        
        for review in reviews:
            review_text = review.find('div', class_='ZmyHeo').get_text(separator=' ', strip=True)
            rating_div = review.find('div', class_=re.compile(r'^XQDdHH.*Ga3i8K$'))
            rating = float(rating_div.get_text(strip=True)) if rating_div else None
            
            # Remove "READ MORE" if present
            read_more = review.find('span', class_='wTYmpv')
            if read_more:
                review_text = review_text.split('READ MORE')[0].strip() + "."

            review_pair = (review_text, rating)
            if review_pair not in seen_reviews:
                all_reviews.append([review_text, rating])
                seen_reviews.add(review_pair)
        
        page_number += 1

        # Sleep every 5 pages to avoid server overload
        if page_number % 5 == 0:
            sleep(1.5)

# Scrape reviews from all URLs
for url in urls:
    get_flipkart_reviews(url)

# Load Machine Learning Models
def load_model(path, name):
    start_time = time.time()
    model = joblib.load(path)
    print(f"{name} loaded in {time.time() - start_time:.4f} seconds")
    return model

logistic_model = load_model(r".\Models\logistic_regression_model.joblib", "Logistic Regression Model")
random_forest_model = load_model(r".\Models\random_forest_model.joblib", "Random Forest Model")
svm_model = load_model(r".\Models\svm_model.joblib", "SVM Model")
stacking_model = load_model(r".\Models\xgb_stacking_model.joblib", "Stacking Model")

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

# Load the predicted reviews CSV
# df_test = pd.read_csv('predicted_reviews.csv')

# Initialize VADER SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(review):
    sentiment = analyzer.polarity_scores(str(review))
    return sentiment['compound'] 

# Apply sentiment analysis only to reviews with prediction < 60%
df_test['sentiment_score'] = df_test['text'].apply(lambda x: get_sentiment_score(x) if x else 0)

# Normalize the sentiment scores to a 0-100 scale
df_test['normalized_sentiment'] = (df_test['sentiment_score'] + 1) * 50 

# Calculate the average sentiment score only for reviews with prediction < 60%
filtered_sentiments = df_test.loc[df_test['final_prediction'] < 60, 'normalized_sentiment']

if not filtered_sentiments.empty:
    average_sentiment = filtered_sentiments.mean()
    print(f"Overall Sentiment Score (for reviews with <60% probability): {average_sentiment:.2f}")
else:
    print("No reviews found with probability < 60%.")
