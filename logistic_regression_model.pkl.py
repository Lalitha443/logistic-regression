# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load your dataset (ensure it's in the same directory)
try:
    train_df = pd.read_csv("titanic_train.csv")
except FileNotFoundError:
    print("Error: 'titanic_train.csv' not found. Make sure the file is in the same directory.")
    exit()

# 2. Data Preprocessing
# Fill missing values
train_df.fillna(method='ffill', inplace=True)
# Encode categorical features
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked'] = le.fit_transform(train_df['Embarked'].astype(str))

# 3. Features (X) and Target (y)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Save the trained model
joblib.dump(model, "logistic_regression_model.pkl")
print("✅ Model saved as logistic_regression_model.pkl")


