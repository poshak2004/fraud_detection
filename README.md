# Fraud Detection using Machine Learning

This project implements machine learning techniques to detect fraudulent credit card transactions.  
The dataset is highly imbalanced, and the goal is to build robust models that can identify fraud with high accuracy while minimizing false positives.  

---

## Features
- Data preprocessing and feature scaling  
- Handling class imbalance using resampling techniques (SMOTE, undersampling)  
- Training multiple ML models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost / LightGBM  
- Model evaluation with precision, recall, F1-score, ROC-AUC  
- Jupyter notebooks for EDA & experimentation  
- Modular Python scripts for training and evaluation  

---

## Tech Stack
- Python 3.9+  
- NumPy, Pandas → data manipulation  
- Matplotlib, Seaborn → visualization  
- Scikit-learn → machine learning models  
- Imbalanced-learn → handling class imbalance  
- XGBoost / LightGBM → gradient boosting models  

---

## Project Structure
```text
fraud_detection/
│── data/                # (dataset ignored in git, >100MB excluded)
│── notebooks/           # Jupyter notebooks with EDA & experiments
│── src/                 # Python scripts for training & evaluation
│── models/              # Saved trained models
│── results/             # Plots, metrics & reports
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
