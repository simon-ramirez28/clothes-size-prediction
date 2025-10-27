# training_pipeline_v2.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Optional advanced models
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    ADVANCED_MODELS = True
except ImportError:
    ADVANCED_MODELS = False


class ModelTrainerV2:
    def __init__(self, data_path, target_col='size'):
        self.data_path = data_path
        self.target_col = target_col
        self.best_model = None
        self.best_score = 0
        self.results = {}

        # Define candidate models and their hyperparameter grids
        self.param_grids = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=2000),
                "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]}
            },
            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {"n_estimators": [100, 200], "max_depth": [5, 10, 15]}
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": [3, 5, 7]}
            },
            "SVM": {
                "model": SVC(),
                "params": {"C": [0.5, 1, 5], "kernel": ["rbf", "linear"]}
            }
        }

        if ADVANCED_MODELS:
            self.param_grids.update({
                "XGBoost": {
                    "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                    "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
                },
                "LightGBM": {
                    "model": LGBMClassifier(),
                    "params": {"n_estimators": [100, 200], "num_leaves": [31, 63]}
                },
                "CatBoost": {
                    "model": CatBoostClassifier(verbose=0),
                    "params": {"iterations": [100, 200], "learning_rate": [0.05, 0.1]}
                }
            })

    def load_data(self):
        print("📥 Loading data...")
        df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def prepare_data(self, df):
        print("🔧 Preparing data...")
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        joblib.dump(self.label_encoder, "label_encoder.pkl")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Apply SMOTE to balance classes
        print("⚖️ Applying SMOTE balancing...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"✅ Data balanced: {X_train_bal.shape[0]} samples (was {X_train.shape[0]})")

        return X_train_bal, X_test, y_train_bal, y_test

    def train_and_tune_models(self, X_train, X_test, y_train, y_test):
        print("🚀 Training and tuning models...\n")

        for name, config in self.param_grids.items():
            print(f"🔍 {name} - Running GridSearchCV...")
            grid = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring="accuracy",
                cv=3,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            self.results[name] = acc
            print(f"✅ {name}: Accuracy = {acc:.4f} (Best params: {grid.best_params_})")

            if acc > self.best_score:
                self.best_score = acc
                self.best_model = best_model

        print("\n🏆 Best model:", type(self.best_model).__name__)
        print(f"✅ Best accuracy: {self.best_score:.4f}")
        return self.best_model

    def save_best_model(self, filename="best_model_v2.pkl"):
        if self.best_model:
            joblib.dump(self.best_model, filename)
            print(f"💾 Model saved as {filename}")

    def full_training_pipeline(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        best_model = self.train_and_tune_models(X_train, X_test, y_train, y_test)
        self.save_best_model()
        print("\n📊 Classification Report:")
        print(classification_report(y_test, best_model.predict(X_test)))