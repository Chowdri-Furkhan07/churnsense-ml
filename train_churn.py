# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# STEP 2: DATA COLLECTION

df = pd.read_csv("churn prediction.csv")

print("Initial Shape:", df.shape)

# STEP 3: DATA CLEANING


# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle null values & missing data
df = df.ffill()
df = df.bfill()
df.fillna(0, inplace=True)

# Fix formats
df['Country'] = df['Country'].str.strip().str.title()

# Convert date
df['LastPurchaseDate'] = pd.to_datetime(
    df['LastPurchaseDate'],
    dayfirst=True,
    errors='coerce'
)
# Handle missing dates
df['LastPurchaseDate'] = df['LastPurchaseDate'].ffill()
df['LastPurchaseDate'] = df['LastPurchaseDate'].bfill()

# STEP 4: FEATURE ENGINEERING


# Convert date to days
df['DaysSinceLastPurchase'] = (
    pd.Timestamp.now() - df['LastPurchaseDate']
).dt.days

df.drop('LastPurchaseDate', axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# STEP 5: EDA


# Who churns more
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Income vs Churn
sns.boxplot(x='Churn', y='Income', data=df)
plt.title("Income vs Churn")
plt.show()

# Spending vs Churn
sns.boxplot(x='Churn', y='SpendingScore', data=df)
plt.title("Spending vs Churn")
plt.show()

# STEP 6: SPLIT DATA


X = df.drop('Churn', axis=1)
y = df['Churn']

# SAVE FEATURE NAMES BEFORE SCALING
features = X.columns.tolist()

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# STEP 7: MODEL TRAINING


models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = f1

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

# STEP 8: BEST MODEL SELECTION

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# STEP 9: SAVE MODEL

pickle.dump((best_model, scaler, features), open("churn_model.pkl", "wb"))

print("Model saved successfully!")