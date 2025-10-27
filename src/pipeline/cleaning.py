"""
cleaning.py
Data Cleaning Pipeline

Handles missing values, duplicates, and basic data consistency checks.
Designed to run after data inspection pipeline (data.py).

Usage:
from src.pipelines.cleaning import DataCleaner

cleaner = DataCleaner(df)
cleaned_df = cleaner.run_pipeline()
"""

import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the DataFrame to clean.

        Args:
            df (pd.DataFrame): Raw or preprocessed dataframe.
        """
        self.df = df.copy()

    def handle_missing_values(self):
        """Drop rows with missing values and report the change."""
        print("\n🧹 Handling missing values...")
        before = self.df.shape
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("✅ No missing values found.")
        else:
            print("⚠️ Columns with missing values detected:\n", missing)
            self.df.dropna(inplace=True)
            after = self.df.shape
            print(f"➡️ Before dropna: {before}")
            print(f"✅ After dropna:  {after}")

    def handle_duplicates(self):
        """Detect and remove duplicate rows."""
        print("\n📑 Checking for duplicates...")
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("✅ No duplicate rows found.")
        else:
            print(f"⚠️ {duplicates} duplicate rows detected. Removing them...")
            self.df.drop_duplicates(inplace=True)
            print(f"✅ Remaining rows after drop: {self.df.shape[0]}")

    def check_data_types(self):
        """Warn about inconsistent data types (e.g., numeric columns stored as objects)."""
        print("\n🔍 Checking data types consistency...")
        for col in self.df.columns:
            if self.df[col].dtype == object:
                # Check if this column should be numeric
                numeric_like = self.df[col].str.replace('.', '', 1).str.isdigit().mean()
                if numeric_like > 0.5:
                    print(f"⚠️ Column '{col}' looks numeric but is stored as object.")
        print("✅ Data type check complete.")

    def handle_outliers(self, z_thresh=3, action="report"):
        """
        Detect and optionally handle outliers.
        action:
            - "report": just print counts (default)
            - "remove": drop rows containing outliers
            - "cap": replace with percentile caps
        """
        print("\n📈 Checking for outliers (Z-score method)...")
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        outlier_counts = {}

        for col in numeric_cols:
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            outliers = (abs(z_scores) > z_thresh)
            count = outliers.sum()

            if count > 0:
                outlier_counts[col] = int(count)
                if action == "remove":
                    self.df = self.df[~outliers]
                elif action == "cap":
                    lower_cap = self.df[col].quantile(0.01)
                    upper_cap = self.df[col].quantile(0.99)
                    self.df[col] = np.clip(self.df[col], lower_cap, upper_cap)

        if outlier_counts:
            print("⚠️ Possible outliers detected:\n", outlier_counts)
            if action == "remove":
                print("🧹 Outlier rows removed.")
            elif action == "cap":
                print("🧱 Outliers capped to 1st–99th percentile.")
        else:
            print("✅ No strong outliers detected.")


    def run_pipeline(self):
        """Run all cleaning steps."""
        print("\n🚀 Starting Data Cleaning Pipeline...\n")
        self.handle_missing_values()
        self.handle_duplicates()
        self.check_data_types()
        self.handle_outliers()
        print("\n✨ Data cleaning completed. Ready for feature engineering.")
        return self.df