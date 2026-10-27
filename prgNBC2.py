
# ============================================================
# NAIVE BAYES CLASSIFICATION - AIR QUALITY PREDICTION
# Dataset: Real_Combine.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv('Data/Real-Data/Real_Combine.csv')
df = df.dropna()

print("Dataset loaded successfully")
print(df.head())

# ============================================================
# 2. CONVERT PM2.5 INTO CATEGORIES
# ============================================================

def categorize_pm25(value):
    if value <= 50:
        return 0       # Low
    elif value <= 100:
        return 1       # Medium
    else:
        return 2       # High

df['PM2.5_Category'] = df['PM2.5'].apply(categorize_pm25)

# ============================================================
# 3. SELECT INPUT FEATURES AND TARGET
# ============================================================

X = df.drop(columns=['PM2.5', 'PM2.5_Category']).values
y = df['PM2.5_Category'].values

# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y
)

# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 6. CREATE NAIVE BAYES CLASSIFIER
# ============================================================

nb = GaussianNB()

# ============================================================
# 7. TRAIN NAIVE BAYES MODEL
# ============================================================

nb.fit(X_train, y_train)

print("\nNaive Bayes Model Trained Successfully")

# ============================================================
# 8. PREDICT TEST DATA
# ============================================================

y_pred = nb.predict(X_test)

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

conf_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(conf_matrix)

# ============================================================
# 10. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low', 'Medium', 'High'],
    yticklabels=['Low', 'Medium', 'High']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')
plt.tight_layout()
plt.show()

# ============================================================
# 11. ACCURACY SCORE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print(f'\nNaive Bayes Accuracy Score: {accuracy * 100:.2f}%')

# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nNaive Bayes Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=['Low', 'Medium', 'High'],
        zero_division=0
    )
)

# ============================================================
# END
# ============================================================

