# FakeReviewML
A project to extract reviews from supported website links and analyze them to determine which reviews are computer generated and which are genuine, providing insights into review authenticity.

## **Steps Involved**  

### **1. Dataset Preparation**  
- The `divide_dataset_in_parts.py` script splits the dataset into multiple parts, ensuring balance between real and fake reviews.  
- The dataset is divided into **training** and **testing** sets.  

### **2. Model Training & Hyperparameter Tuning**  
Each model undergoes **hyperparameter tuning** using GridSearchCV, followed by final training with optimized parameters.  

#### **Base Models**  
- **SVM** → `Gridsearch/SVM.py` (tuning) → `Training/SVM.py` (final training)  
- **Random Forest** → `Gridsearch/Random_forest.py` → `Training/Random_forest.py`  
- **Logistic Regression** → `Gridsearch/Logistic_Regression.py` → `Training/Logistic_Regression.py`  

#### **Stacking Model (XGBoost)**  
- Combines predictions from **SVM, Random Forest, and Logistic Regression**.  
- **XGBoost** assigns weightage to each model’s predictions to improve accuracy.  
- `Gridsearch/XGBoost.py` (tuning) → `Training/XGBoost.py` (final training).  

### **3. Fake Review Detection & Sentiment Analysis**  
- The `Scraping_Identifying.py` script **scrapes reviews from Flipkart**.  
- Uses trained models to **predict authenticity** of reviews.  
- If a review has **less than 60% authenticity probability**, it undergoes **sentiment analysis** to understand its tone.  

## **How to Run the Project**  
1. **Prepare the Dataset** → Run `divide_dataset_in_parts.py` to split data.  
2. **Train Models** → Use `Gridsearch` scripts for tuning, then `Training` scripts to finalize models.  
3. **Detect Fake Reviews** → Run `Scraping_Identifying.py` to scrape and analyze reviews.  

## Requirements

- Python 3.x
- pandas
- scikit-learn 1.2.2
- xgboost
- joblib
- BeautifulSoup
- vaderSentiment

Install the required packages using:

```bash
pip install pandas scikit-learn xgboost joblib beautifulsoup4 vaderSentiment
```

## **Conclusion**  
This project **scrapes and analyzes online reviews** using machine learning. It leverages **ensemble learning (stacking)** to improve accuracy, helping users identify **authentic reviews** before making purchase decisions.

## Project Structure

```
FakeReviewML/
│
├── Dataset/
│   ├── divide_dataset_in_parts.py
│   ├── part1.csv
│   ├── part2.csv
│   ├── part3.csv
│   ├── part4.csv
│   ├── test.csv
│
├── Gridearch/
│   ├── SVM.py
│   ├── Random_forest.py
│   ├── Logistic_Regression.py
│   ├── XGBoost.py
│
├── Training/
│   ├── SVM.py
│   ├── Random_forest.py
│   ├── Logistic_Regression.py
│   ├── XGBoost.py
│
├── Models/
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── svm_model.joblib
│   ├── stacking_model.joblib
│
├── Scraping_Identifying.py
├── README.md
```