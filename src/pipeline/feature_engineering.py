"""
feature_engineering.py
Feature engineering pipeline for Clothes Size Prediction dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, target_col: str = "Size", out_dir: str = "results/features"):
        self.df = df.copy()
        self.target_col = target_col
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    # -------------------------
    # Numeric feature creation
    # -------------------------
    def create_bmi(self):
        """Create Body Mass Index feature."""
        self.df["BMI"] = self.df["weight"] / ((self.df["height"] / 100) ** 2)
        print("🧮 Feature 'BMI' created.")
        return self

    def create_interactions(self):
        """Optional: add some simple feature interactions."""
        self.df["weight_age_ratio"] = self.df["weight"] / (self.df["age"] + 1)
        self.df["height_age_ratio"] = self.df["height"] / (self.df["age"] + 1)
        print("⚙️ Interaction features created: weight_age_ratio, height_age_ratio.")
        return self

    # -------------------------
    # Encoding target or categorical vars
    # -------------------------
    def encode_target(self):
        """Encode the Size column into numerical labels."""
        if self.target_col in self.df.columns:
            self.df[f"{self.target_col}_encoded"] = self.encoder.fit_transform(self.df[self.target_col])
            print(f"🔢 Encoded target column '{self.target_col}' as '{self.target_col}_encoded'.")
        else:
            print(f"⚠️ Target column '{self.target_col}' not found.")
        return self

    # -------------------------
    # Scaling numeric features
    # -------------------------
    def scale_numeric_features(self):
        """Scale numeric features (excluding target)."""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != f"{self.target_col}_encoded"]
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        print(f"📏 Scaled numeric features: {numeric_cols}")
        return self

    # -------------------------
    # Export
    # -------------------------
    def save_features(self, filename: str = "features_processed.csv"):
        """Save the final dataset with new features."""
        path = os.path.join(self.out_dir, filename)
        self.df.to_csv(path, index=False)
        print(f"✅ Saved feature-engineered dataset to {path}")
        return path

    # -------------------------
    # Run all steps
    # -------------------------
    def run_all(self, scale: bool = True, interactions: bool = True):
        """Run full feature engineering pipeline."""
        print("🚀 Running full Feature Engineering pipeline...")
        self.create_bmi()
        if interactions:
            self.create_interactions()
        self.encode_target()
        if scale:
            self.scale_numeric_features()
        print("✅ Feature engineering pipeline completed.")
        return self.df