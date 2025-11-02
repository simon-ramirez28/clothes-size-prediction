"""
feature_engineering.py
Pre-processing pipeline: Scaling and Target Encoding.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib # -> A must to save models

# Definición de rutas (ajusta si es necesario)
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, target_col: str = "size"):
        self.df = df.copy()
        self.target_col = target_col
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        # Definimos las features a usar, sin el BMI ni interacciones
        self.features = ['weight', 'height', 'age'] 
        
    # ------------------------------------------------
    # 🚫 ELIMINATED Functions: create_bmi and create_interactions
    # ------------------------------------------------

    def encode_target(self):
        """Encode the Size column into numerical labels and SAVE the encoder."""
        if self.target_col in self.df.columns:
            # 1. Codificate
            self.df[f"{self.target_col}_encoded"] = self.encoder.fit_transform(self.df[self.target_col])
            print(f"🔢 Encoded target column '{self.target_col}'.")
            
            # 2. Save the LabelEncoder for Inference
            joblib.dump(self.encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
            print("💾 LabelEncoder saved.")
        else:
            print(f"⚠️ Target column '{self.target_col}' not found.")
        return self

    def scale_numeric_features(self):
        """Scale numeric features (weight, height, age) and SAVE the scaler."""
        
        # 1. Escalar
        # Usamos self.features para asegurar que solo se escalen las que queremos
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])
        print(f"📏 Scaled numeric features: {self.features}")
        
        # 2. Guardar el Scaler para Inferencia
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
        print("💾 StandardScaler saved.")
        return self

    # ------------------------------------------------
    # Ejecución Principal
    # ------------------------------------------------
    def run_all_preprocessing(self):
        """Run essential pre-processing pipeline: encoding and scaling."""
        print("🚀 Running essential Pre-processing pipeline (Scaling & Encoding)...")
        self.encode_target()
        self.scale_numeric_features()
        print("✅ Pre-processing pipeline completed.")
        
        # Devolvemos el DataFrame con las features escaladas y el target codificado
        return self.df