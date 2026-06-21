# 🔮 ChurnSense — Customer Churn Prediction System

A machine learning web application that predicts customer churn using behavioral, demographic, and transactional data. Built with Scikit-learn and deployed via a professional Streamlit interface with a dark analytics-grade UI.

---

## 📌 Problem Statement

Customer churn is one of the most critical challenges in retail and e-commerce. Retaining an existing customer costs **5× less** than acquiring a new one. This project builds an end-to-end ML pipeline that identifies at-risk customers **before** they leave — enabling proactive, data-driven retention strategies.

---

## 🖥️ App Preview

> *Dark-themed professional dashboard with sidebar input controls, live metric cards, feature importance visualization, and contextual retention recommendations.*

![Dashboard Screenshot](Screenshots/dashboard.png)

---

## 🗂️ Project Structure

```
churnsense-ml/
│
├── app_churn.py              # Streamlit web app (UI + prediction)
├── train_churn.py            # Model training pipeline
├── churn_model.pkl           # Saved model (best estimator + scaler + features)
├── churn_prediction.csv      # Dataset (5,200 customer records)
├── requirements.txt          # Python dependencies
├── Screenshots/              # App screenshots
└── README.md
```

---

## 📊 Dataset Overview

| Attribute | Details |
|---|---|
| **Records** | 5,200 customers |
| **Features** | 19 input features + 1 target |
| **Target** | `Churn` — Yes / No (binary) |
| **Class Balance** | Yes: 2,627 · No: 2,573 (nearly balanced) |

### Feature Categories

| Category | Features |
|---|---|
| Demographics | Age, Gender, Income |
| Location | City, State, Country |
| Purchase Behaviour | SpendingScore, PurchaseAmount, ProductCategory, PaymentMethod |
| Engagement | Returns, DiscountUsed, ReviewScore, SessionTime, Browser, Device |
| Temporal | LastPurchaseDate → DaysSinceLastPurchase (engineered) |

---

## ⚙️ ML Pipeline

```
Raw CSV  →  Data Cleaning  →  Feature Engineering  →  Encoding  →  Scaling  →  Model Training  →  Evaluation  →  Best Model Saved
```

### Steps in `train_churn.py`

1. **Data Collection** — Load `churn_prediction.csv`
2. **Data Cleaning** — Remove duplicates, handle nulls (forward-fill + back-fill), fix formats
3. **Feature Engineering** — Convert `LastPurchaseDate` → `DaysSinceLastPurchase` (recency signal)
4. **Encoding** — Label Encoding on all categorical columns
5. **EDA** — Churn distribution, Income vs Churn, Spending vs Churn
6. **Train/Test Split** — 80/20 split, `random_state=42`
7. **Model Training** — 6 classifiers benchmarked
8. **Best Model Selection** — Ranked by F1 Score
9. **Model Serialization** — `pickle.dump` saves `(model, scaler, features)`

### Models Benchmarked

| Model | Metric Used |
|---|---|
| Logistic Regression | F1 Score |
| K-Nearest Neighbors | F1 Score |
| Naive Bayes | F1 Score |
| Support Vector Machine | F1 Score |
| Decision Tree | F1 Score |
| ✅ Random Forest *(best)* | F1 Score |

> F1 Score was used as the selection criterion to balance Precision and Recall, avoiding bias from class distribution.

---

## 🖥️ App Features (`app_churn.py`)

- **Sidebar input panel** — 16 customer attributes organized into 4 sections
- **Live metric cards** — Income, Spending Score, Review Score update in real time
- **Customer snapshot strip** — 8-field summary grid for at-a-glance review
- **Feature importance visualization** — Horizontal bar chart of top churn drivers
- **Prediction output** — Color-coded result card (🚨 Churn Risk / ✅ Retained)
- **Contextual action panels** — Automated retention or nurture recommendations

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chowdri-Furkhan07/churnsense-ml.git
cd churnsense-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_churn.py
```

This generates `churn_model.pkl` in the project root.

### 4. Launch the app

```bash
streamlit run app_churn.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

> See `requirements.txt` for pinned versions.

---

## 📈 Results

| Metric | Value |
|---|---|
| Best Model | Random Forest Classifier |
| Selection Criterion | F1 Score |
| Dataset Size | 5,200 records |
| Features Used | 19 (+ 1 engineered) |
| Train / Test Split | 80% / 20% |

---

## 🧠 Key Insights from EDA

- Customers with **lower spending scores** show higher churn rates
- **High return counts** (3+) correlate strongly with churn
- **Short session times** paired with low review scores are strong churn signals
- Income alone is not a reliable predictor — **behaviour patterns dominate**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Framework | Scikit-learn |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Storage | Pickle |

---

## 👤 Author

**Chowdri Furkhan**
AIML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/chowdri-furkhan/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/Chowdri-Furkhan07)

---

## 📄 License

This project is for academic and portfolio purposes.

---
