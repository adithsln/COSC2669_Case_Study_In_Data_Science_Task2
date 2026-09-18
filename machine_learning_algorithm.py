import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate

from sklearn.metrics import accuracy_score
import os

file_path = "digital_marketing_campaign_dataset - digital_marketing_campaign_dataset.csv.csv"
target_col = "Conversion"

print("Loading and processing file!")

df = pd.read_csv(file_path)

df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sensitive_features= X_test['Gender']

metric_dict = {
    "accuracy": accuracy_score,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate
}

numeric_features = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object','category','bool']).columns.tolist()

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numeric_features])

X_test_num_scaled = scaler.transform(X_test[numeric_features]) 

encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_cat_encoded = encoder.transform(X_test[categorical_features])

X_train_processed = np.hstack((X_train_num_scaled,X_train_cat_encoded))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_encoded))



print("\n--- Training LightGBM")
lgbm = LGBMClassifier(n_estimators=100, random_state=42, is_unbalance=True, learning_rate=0.1, max_depth=100)
lgbm.fit(X_train_processed, y_train)
y_pred = lgbm.predict(X_test_processed)
print(f"LightGBM Acuracy: {accuracy_score(y_test,y_pred)}")
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred))
print("LightGBM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

mf_lgbm = MetricFrame(
    metrics=metric_dict,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)
print("\n--- LightGBM Overall Model Metrics ---")
print(mf_lgbm.overall)
print("\n--- LightGBM Model Metrics by Gender ---")
print(mf_lgbm.by_group)

print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
rf.fit(X_train_processed, y_train)
y_pred = rf.predict(X_test_processed)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

mf_rf = MetricFrame(
    metrics=metric_dict,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("\n--- Random Forest Overall Model Metrics ---")
print(mf_rf.overall)
print("\n--- Random Forest Metrics by Gender ---")
print(mf_rf.by_group)

file_path = "marketing_AB.csv"
target_col = "converted"

print("Loading and processing file!")

df = pd.read_csv(file_path)

df = pd.read_csv(file_path)
df = df.drop(columns=['Unnamed: 0', 'user id'], errors='ignore')
df[target_col] = df[target_col].astype(int)
df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object','category','bool']).columns.tolist()

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numeric_features])

X_test_num_scaled = scaler.transform(X_test[numeric_features]) 

encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_cat_encoded = encoder.transform(X_test[categorical_features])

X_train_processed = np.hstack((X_train_num_scaled,X_train_cat_encoded))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_encoded))

print("\n--- Training LightGBM")
lgbm = LGBMClassifier(n_estimators=100, random_state=42, is_unbalance=True, learning_rate=0.1, max_depth=100)
lgbm.fit(X_train_processed, y_train)
y_pred = lgbm.predict(X_test_processed)
print(f"LightGBM Acuracy: {accuracy_score(y_test,y_pred)}")
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred))
print("LightGBM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
rf.fit(X_train_processed, y_train)
y_pred = rf.predict(X_test_processed)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
