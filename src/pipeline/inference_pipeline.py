# inference_pipeline.py

import pandas as pd
import numpy as np
import joblib

class InferencePipeline:
    def __init__(self, model_path="best_model.pkl", encoder_path="label_encoder.pkl", scaler_path=None):
        print("🧠 Initializing inference pipeline...")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        print("✅ Model and encoder loaded successfully.")

    def create_features(self, data: pd.DataFrame):
        """Replicates the feature engineering pipeline used during training."""
        df = data.copy()

        # Derived features (as in the feature pipeline)
        df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
        df["weight_age_ratio"] = df["weight"] / df["age"]
        df["height_age_ratio"] = df["height"] / df["age"]

        # Apply same scaling if available
        if self.scaler:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def predict(self, input_data):
        """
        input_data: dict or DataFrame with columns ['weight', 'age', 'height']
        Example:
            {'weight': 72, 'age': 30, 'height': 178}
        """
        if isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            data = input_data
        else:
            raise ValueError("Input must be a dictionary or pandas DataFrame")

        print("📦 Creating features...")
        features = self.create_features(data)

        print("Making prediction...")
        prediction_encoded = self.model.predict(features)
        prediction = self.label_encoder.inverse_transform(prediction_encoded)

        print(f"✅ Predicted size: {prediction[0]}")
        return prediction[0]