# 🛡️ Fraud Detection using Machine Learning

## 📌 Problem Statement
Financial fraud leads to significant monetary losses every year. The objective of this project is to build a **robust fraud detection system** that accurately identifies fraudulent transactions from historical data while effectively handling **class imbalance** and maximizing **recall**, which is critical in fraud-sensitive applications.

---

## 🗂️ Dataset
- **Source:** Public credit card transaction dataset  
- **Data Type:** Transaction-level numerical features  
- **Target Variable:** `Class`  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction  
- **Key Challenge:** Extreme class imbalance (fraud cases are rare)

---

## 🔍 Exploratory Data Analysis (EDA)
The following EDA steps were performed:
- Class distribution analysis to understand imbalance  
- Feature distribution visualization  
- Correlation analysis  
- Outlier inspection  

### Key Insights
- Severe class imbalance required specialized handling techniques  
- Certain features showed strong discriminative power for fraud detection  
- Accuracy alone is misleading for fraud detection problems  

---

## 🧠 Modeling Approach

### 🔹 Data Preprocessing
- Feature scaling using `StandardScaler`  
- Stratified train-test split  
- Techniques to address class imbalance  

### 🔹 Models Implemented
- Logistic Regression (baseline model)  
- Random Forest Classifier  
- Gradient Boosting / XGBoost  

### 🔹 Evaluation Strategy
- Stratified cross-validation  
- Threshold tuning to prioritize recall  
- Model comparison using multiple performance metrics  

---

## 📊 Results & Performance

Models were evaluated using **stratified cross-validation** with a strong focus on **recall** and **ROC-AUC**, as missing fraudulent transactions is significantly more costly than false positives.

### 📈 Model Performance Summary

| Model | ROC-AUC | Precision | Recall | F1-Score |
|------|--------|-----------|--------|----------|
| Logistic Regression (Baseline) | 0.94 | 0.82 | 0.76 | 0.79 |
| Random Forest | 0.97 | 0.89 | 0.83 | 0.86 |
| XGBoost | **0.99** | **0.92** | **0.88** | **0.90** |

> ⚠️ *Metrics shown are representative and should be replaced with final evaluated values from model runs.*

---

### 🧮 Evaluation Metrics Used
- **ROC-AUC** – overall discriminative ability  
- **Precision** – proportion of predicted frauds that are correct  
- **Recall** – ability to correctly identify fraudulent transactions  
- **F1-Score** – balance between precision and recall  
- **Confusion Matrix** – detailed error analysis  

---
📌 **Final Model Selection:**  
The final model was selected based on **high recall with acceptable precision**, prioritizing fraud detection over minimizing false positives.

---

## 📈 Visualizations
- ROC Curves  
- Confusion Matrices  
- Feature Importance Plots  

(All visualizations are stored in the `results/figures/` directory.)

---

## 🚀 Deployment

### 🔹 Streamlit Application
An interactive Streamlit app allows users to:
- Input transaction features  
- Receive real-time fraud probability predictions  

### 🔹 API (Optional Extension)
The trained model can be served using **FastAPI** for real-time inference in production environments.

---

## 📦 Project Structure

```text
fraud_detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/
├── results/
│   ├── metrics.csv
│   └── figures/
├── app.py
├── requirements.txt
└── README.md
```
## ▶️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/poshak2004/fraud_detection.git
cd fraud_detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
## 🧪 Tools & Technologies

### 🧠 Programming & Data Handling
- **Python** – core language for data processing and modeling  
- **NumPy**, **Pandas**, **SciPy** – numerical computing, data manipulation, and scientific operations  
- **SQL** – querying and preprocessing structured data  

---

### 🤖 Machine Learning & Modeling
- **scikit-learn** – preprocessing, modeling, pipelines, and evaluation  
- **XGBoost / Gradient Boosting** – high-performance ensemble models  
- **Imbalanced-learn** – handling class imbalance in fraud detection  
- Feature engineering & feature selection techniques  

---

### 📊 Data Analysis & Visualization
- **Matplotlib** – static visualizations  
- **Seaborn** – statistical data visualization  
- **Plotly** – interactive plots for deeper analysis  

---

### 📈 Model Evaluation & Experimentation
- **ROC-AUC**, Precision, Recall, F1-score  
- Confusion Matrix & Precision–Recall curve analysis  
- **MLflow** – experiment tracking, model comparison, and reproducibility  

---

### 🚀 Deployment & Applications
- **Streamlit** – interactive web app for model inference  
- **FastAPI** – REST APIs for real-time prediction services (extension-ready)  

---

### 🛠️ Development & Workflow
- **Git & GitHub** – version control and collaboration  
- **Jupyter Notebook** – EDA and rapid experimentation  
- **VS Code** – development environment  
- Modular, reproducible project structure following ML best practices  

---

## 📌 Key Takeaways
- Fraud detection requires **metric-driven evaluation**, not accuracy alone  
- Handling **class imbalance** is essential for real-world performance  
- Recall-focused optimization helps minimize costly false negatives  
- Well-structured pipelines improve reproducibility and deployment readiness  

---

## 🔮 Future Improvements
- Hyperparameter optimization using **Optuna**  
- Advanced resampling techniques (SMOTE variants)  
- Model explainability using **SHAP** and feature importance analysis  
- Model monitoring and data drift detection  
- Dockerized deployment for production environments  

---

## 📎 Notes
This project is designed to reflect **real-world data science workflows**, emphasizing:
- End-to-end ownership  
- Strong evaluation and validation practices  
- Practical, deployment-ready machine learning solutions  


