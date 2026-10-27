import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)
# Predict on test set
y_pred = dt_model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy * 100:.2f}%')

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.show()

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy * 100:.2f}%')

values = input("Enter 8 comma-separated values: ").split(",")



if len(values) != 8:
    print("Error: Please enter exactly 8 values.")
else:
    record = [[float(value.strip()) for value in values]]

    # Scale the input using the same fitted scaler
    record_scaled = scaler.transform(record)

    # Predict using the trained model
    prediction =dt_model.predict(record_scaled)

    print("Predicted output:", prediction[0])
     # Convert numeric prediction to string
    categories = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    print("Predicted PM2.5 Category:", categories[int(prediction[0])])