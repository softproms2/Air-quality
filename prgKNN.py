import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
df = pd.read_csv('Data/Real-Data/Real_Combine.csv')
df = df.dropna()

# Convert PM2.5 to categorical (classification problem)
def categorize_pm25(value):
    if value <= 50:
        return 0  # Low
    elif value <= 100:
        return 1  # Medium
    else:
        return 2  # High

df['PM2.5_Category'] = df['PM2.5'].apply(categorize_pm25)

# Independent and dependent features
X = df.iloc[:, :-2].values  # all columns except PM2.5 and new category
y = df['PM2.5_Category'].values  # Classification target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune n_neighbors
knn.fit(X_train, y_train)



# Predictions
y_pred = knn.predict(X_test)

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy * 100:.2f}%')
