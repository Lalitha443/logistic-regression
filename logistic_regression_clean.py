#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_df=pd.read_csv("C:\\Users\\lalit\\OneDrive\\Documents\\titanic_train.csv")
train_df
test_df=pd.read_csv("C:\\Users\\lalit\\OneDrive\\Documents\\titanic_test.csv")
test_df

train_df.describe()
test_df.describe()

train_df.info()
test_df.info()

train_df['PassengerId'].hist()
test_df['PassengerId'].hist()

train_df['Survived'].hist()

train_df['Pclass'].hist()
test_df['Pclass'].hist()

train_df['Age'].hist()
test_df['Age'].hist()

train_df['SibSp'].hist()
test_df['SibSp'].hist()

train_df['Parch'].hist()
test_df['Parch'].hist()

train_df['Fare'].hist()
test_df['Fare'].hist()


# Bivariate (numerical vs categorical)
import seaborn as sns
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title("Boxplot of Age by Survival")
plt.show()

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

#  Handle missing values
train_df.fillna(method='ffill', inplace=True)
test_df.fillna(method='ffill', inplace=True)

# DATA TRANSFORMATION for train data
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

train_df["Name"] = LE.fit_transform(train_df["Name"])
train_df["Sex"] = LE.fit_transform(train_df["Sex"])
train_df["Ticket"] = LE.fit_transform(train_df["Ticket"])
train_df["Cabin"] = LE.fit_transform(train_df["Cabin"])
train_df["Embarked"] = LE.fit_transform(train_df["Embarked"])
train_df.tail()






from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

test_df["Name"] = LE.fit_transform(test_df["Name"])
test_df["Sex"] = LE.fit_transform(test_df["Sex"])
test_df["Ticket"] = LE.fit_transform(test_df["Ticket"])
test_df["Cabin"] = LE.fit_transform(test_df["Cabin"])
test_df["Embarked"] = LE.fit_transform(test_df["Embarked"])
test_df.tail()


X_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y_train = train_df['Survived']

X_test = test_df
Y_test = train_df['Survived']



#Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)



#Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)


#Model Testing
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("✅ Accuracy:", accuracy_score(Y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred))




X_final = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]




test_predictions = model.predict(X_final)




test_df
test_df.drop(columns=['Y_pred'], inplace=True)
test_df

from sklearn.metrics import recall_score,precision_score,f1_score
r_score = recall_score(Y_test,y_pred)
print("sensitivity score:" , np.round(r_score,2))

p_score = precision_score(Y_test,y_pred)
print("precision score:" , np.round(p_score,2))

f1_score = f1_score(Y_test,y_pred)
print("F1 score:" , np.round(f1_score,2))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix

# Extract TN and FP from the matrix
# cm.ravel() flattens the array for easy unpacking
tn, fp, fn, tp = cm.ravel()

# Calculate specificity manually
if (tn + fp) > 0:
    specificity = tn / (tn + fp)
    print(f"\nSpecificity Score: {specificity:.4f}")
else:
    print("\nSpecificity cannot be calculated as there are no actual negative cases (non-survivors) in the test set.")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", cm)

# unpack if it's 2x2
TN, FP, FN, TP = cm.ravel()
print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
TN = cm[0,0]
FP = cm[1,0]
TNR = TN/(TN + FP)
print("Specificity score:" , np.round(TNR,2))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:,1]  # probability of class 1 (Survived)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)

# Compute AUC
roc_auc = roc_auc_score(Y_test, y_pred_proba)

# Plot ROC Curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')  # baseline (random model)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Titanic Survival Prediction")
plt.legend()
plt.show()




# ===============================
# Titanic Survival Prediction
# Random Forest Classifier
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score,log_loss,confusion_matrix,roc_curve,roc_auc_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_samples=0.8, max_features=0.7)
rf_model.fit(X_train, Y_train)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Probabilities for log loss & ROC
y_pred_train_proba = rf_model.predict_proba(X_train)
y_pred_test_proba = rf_model.predict_proba(X_test)

print("\n✅ Random Forest Results")
print("Training Accuracy:", accuracy_score(Y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(Y_test, y_pred_test))
print("Training Log Loss:", log_loss(Y_train, y_pred_train_proba))
print("Test Log Loss:", log_loss(Y_test, y_pred_test_proba))



y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_test_proba[:, 1])
roc_auc = roc_auc_score(Y_test, y_pred_test_proba[:, 1])

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Titanic Survival Prediction")
plt.legend()
plt.show()




# Feature Importance
feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(6,4), color="skyblue")
plt.title("Feature Importance - Random Forest")
plt.show()

