# training_pipeline.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, data_path, target_col='size'):
        self.data_path = data_path
        self.target_col = target_col
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC()
        }
        self.best_model = None
        self.best_score = 0
        self.results = {}

    def load_data(self):
        print("📥 Loading data...")
        df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def prepare_data(self, df):
        print("🔧 Preparing data for training...")
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Save encoder for inference
        joblib.dump(self.label_encoder, "label_encoder.pkl")

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        print(f"✅ Split complete: {X_train.shape[0]} train / {X_test.shape[0]} test")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("🚀 Training models...\n")
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.results[name] = acc
            print(f"{name}: Accuracy = {acc:.4f}")
            
            if acc > self.best_score:
                self.best_score = acc
                self.best_model = model
        
        print("\n🏆 Best model:", type(self.best_model).__name__)
        print(f"✅ Accuracy: {self.best_score:.4f}")
        return self.best_model

    def save_best_model(self, filename="best_model.pkl"):
        if self.best_model:
            joblib.dump(self.best_model, filename)
            print(f"💾 Best model saved as {filename}")

    def full_training_pipeline(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        best_model = self.train_and_evaluate(X_train, X_test, y_train, y_test)
        self.save_best_model()
        print("\n📊 Classification Report:")
        print(classification_report(y_test, best_model.predict(X_test)))
