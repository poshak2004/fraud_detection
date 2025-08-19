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

fraud_detection/
│── data/                # (dataset ignored in git, >100MB excluded)
│── notebooks/           # Jupyter notebooks with EDA & experiments
│── src/                 # Python scripts for training & evaluation
│── models/              # Saved trained models
│── results/             # Plots, metrics & reports
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation

## Installation & Setup
-Clone the repository:

git clone https://github.com/poshak2004/fraud_detection.git
cd fraud_detection

-Create a virtual environment & activate it:

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

-Install dependencies:
pip install -r requirements.txt

-Place the dataset (creditcard.csv) inside the data/ folder.
Note: The dataset is ignored in version control due to its size.

-Usage
Run EDA and experiments using notebooks in the notebooks/ folder.
Train models using scripts in src/. Example:
python src/train_model.py
Saved models and results will be stored in models/ and results/.

-Results
Models are evaluated using:
Precision
Recall
F1-score
ROC-AUC
Detailed results and plots are available in the results/ directory.

-License
This project is licensed under the MIT License. See LICENSE for details.
