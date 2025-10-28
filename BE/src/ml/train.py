import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, target):
    features = [col for col in df.columns if col not in ['Num', 'subject_ID', target]]
    X = df[features]
    y = df[target]
    return X, y

def balance_data(X_train, y_train):
    sm = SMOTE(random_state=42, sampling_strategy='auto')
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_res, y_res = balance_data(X_train, y_train)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_res, y_res)

    return model, X_test, y_test

def save_model(model, model_path):
    joblib.dump(model, model_path)

def main():
    data_filepath = 'path/to/your/data.csv'  # Update with your data file path
    model_path = os.path.join('models', 'random_forest.joblib')

    df = load_data(data_filepath)
    X, y = preprocess_data(df, target='Hypertension')
    model, X_test, y_test = train_model(X, y)
    save_model(model, model_path)

if __name__ == "__main__":
    main()