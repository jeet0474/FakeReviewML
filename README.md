# **FakeReviewML**  
A project to extract reviews from supported website links and analyze them to determine which reviews are computer generated and which are genuine, providing insights into review authenticity.  

---

## **How to Run the Project**  
### **First, clone this repository.**  

Then follow these steps:  

1. **Start the Backend:**  
   ```bash
   cd ./backend
   python manage.py runserver
   ```  

2. **Start the Frontend:**  
   ```bash
   cd ./frontend
   python app.py
   ```  

3. **Access the Web Interface:**  
   - Open **http://127.0.0.1:5000** in your browser.  
   - Paste any **Flipkart product link** in the input field.  
   - Wait for **1-2 minutes** while the system scrapes reviews and processes them.  
   - The system will display **sentiments and the probability of reviews being AI-generated.**  

---

## **Steps Involved**  

### **1. Dataset Preparation**  
- The `divide_dataset_in_parts.py` script splits the dataset into multiple parts, ensuring a balance between real and fake reviews.  
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

---

## **Project Structure**  

```
D:.
├───backend
│   ├───backend
│   │   └───__pycache__
│   ├───Models
│   └───process
│       ├───migrations
│       │   └───__pycache__
│       └───__pycache__
├───Dataset
├───frontend
│   ├───static
│   └───templates
├───Gridearch
├───Models
└───Training
```

---

## **Requirements**  

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

---

## **Conclusion**  
This project **scrapes and analyzes online reviews** using machine learning. It leverages **ensemble learning (stacking)** to improve accuracy, helping users identify **authentic reviews** before making purchase decisions. 🚀  

---