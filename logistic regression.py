# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 22:38:37 2025

@author: lalit
"""
import streamlit as st

st.title("🚀 Logistic Regression App")
st.write("Welcome! If you see this, the app is working.")

#  Import libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")


#  Preview data
print(train_df.head())

#  Survival distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Survived')
plt.title("Survival Distribution")
plt.show()

#  Survival by class
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Pclass', hue='Survived')
plt.title("Survival by Passenger Class")
plt.show()

#  Age distribution
plt.figure(figsize=(6,4))
sns.histplot(train_df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

#  Age vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title("Age vs Survival")
plt.show()

#  Handle missing values
train_df.fillna(method='ffill', inplace=True)
test_df.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
for col in train_df.select_dtypes(include=['object']).columns:
    train_df[col] = label_enc.fit_transform(train_df[col].astype(str))
    test_df[col] = label_enc.fit_transform(test_df[col].astype(str))

#  Define features and target
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

#Evaluation
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

#  Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.show()

#  Predict on test data
test_predictions = model.predict(test_df)
print("Test Predictions (first 10):", test_predictions[:10])



