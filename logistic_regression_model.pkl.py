# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 23:39:21 2025

@author: lalit
"""
# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load your dataset (replace with your CSV file)
train_df=pd.read_csv("C:\\Users\\lalit\\OneDrive\\Documents\\titanic_train.csv")

# 2. Features (X) and Target (y)  -> adjust column names
X = train_df.drop("Survived", axis=1)  
y =train_df["Survived"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save the trained model
joblib.dump(model, "logistic_regression_model.pkl")
print("✅ Model saved as logistic_regression_model.pkl")

