"""
data.py
Initial data processing and inspection pipeline.

This script loads a DataFrame and performs basic validations:
- Displays general information and descriptive statistics.
- Detects null values ​​by column.
- Checks for date columns (by name).
- Suggests format conversions if necessary.

Usage:
from src.pipelines.data import DataProcessor

processor = DataProcessor("data/raw/clothes_info.csv")
df = processor.run_pipeline()
"""

import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data_path: str):
        """
        Initializes the pipeline with the path to the dataset.

        Args:
        data_path (str): Path to the CSV file.
        """
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Loads data from CSV."""
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data uploaded correctly. {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        return self.df

    def show_basic_info(self):
        """Displays general information about the DataFrame."""
        print("\n📊 General information about the dataset:\n")
        print(self.df.info())
        print("\n📈 Descriptive statistics:\n")
        print(self.df.describe(include="all"))

    def check_missing_values(self):
        """Checks for columns with null values."""
        print("\n🔍 Checking for missing values...\n")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("✅ No missing values detected in any column.")
        else:
            print("⚠️ Columns with missing values detected:\n")
            print(missing)

    def validate_date_columns(self):
        """Detects date columns by name and checks format."""
        print("\n🕒 Checking for date columns...\n")
        date_cols = [col for col in self.df.columns if col.lower().startswith(("date", "fecha"))]

        if not date_cols:
            print("✅ No date columns detected.")
            return

        print(f"🗓️ Columns detected with possible date information: {date_cols}")

        for col in date_cols:
            try:
                # Attempt to convert to datetime
                self.df[col] = pd.to_datetime(self.df[col], errors="raise")
                print(f"✅ Column '{col}' is already in a valid date format.")
            except Exception:
                print(f"⚠️ Column '{col}' requires conversion to date format (e.g., YYYY-MM-DD).")

    def run_pipeline(self):
        """Executes all stages of the pipeline."""
        self.load_data()
        self.show_basic_info()
        self.check_missing_values()
        self.validate_date_columns()
        print("\n🚀 Data processing pipeline completed.\n")
        return self.df
