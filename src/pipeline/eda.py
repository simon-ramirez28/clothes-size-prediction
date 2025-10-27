"""
eda.py
Exploratory Data Analysis pipeline.

Usage:
from src.pipelines.eda import EDAProcessor
eda = EDAProcessor(df, out_dir="results/eda")
eda.run_all(show_plots=False)

Methods are modular so you can call them one-by-one in a notebook:
- eda.summary()
- eda.missing_values_report()
- eda.plot_numeric_distributions()
- eda.plot_boxplots()
- eda.plot_correlation_heatmap()
- eda.plot_scatter_matrix()
"""

import os
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

plt.rcParams.update({"figure.max_open_warning": 0})  # avoid warnings with many figures


class EDAProcessor:
    def __init__(self, df: pd.DataFrame, out_dir: str = "results/eda"):
        self.df = df.copy()
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # -------------------------
    # Summary / textual reports
    # -------------------------
    def summary(self) -> Dict[str, Any]:
        """Return info(), describe() and basic shapes. Print summary too."""
        print("📋 DataFrame shape:", self.df.shape)
        print("\n---- info() ----")
        self.df.info()
        print("\n---- describe() ----")
        print(self.df.describe(include="all").T)

        result = {
            "shape": self.df.shape,
            "info": self.df.dtypes.to_dict(),
            "describe": self.df.describe(include="all")
        }
        return result

    def missing_values_report(self) -> pd.Series:
        """Return and print count and percentage of missing values per column."""
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        report = pd.concat([missing_count, missing_pct], axis=1)
        report.columns = ["missing_count", "missing_pct"]
        report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)

        if report.empty:
            print("✅ No missing values detected.")
        else:
            print("⚠️ Missing values report (count and %):")
            print(report)

        # Save CSV
        report_path = os.path.join(self.out_dir, "missing_values_report.csv")
        report.to_csv(report_path)
        print(f"Saved missing values report to {report_path}")
        return report

    # -------------------------
    # Plots for numeric columns
    # -------------------------
    def plot_numeric_distributions(self, numeric_cols: Optional[list] = None, bins: int = 30, show: bool = False):
        """Plot histograms for numeric columns and save them."""
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(self.df[col].dropna(), bins=bins)
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            fig.tight_layout()
            path = os.path.join(self.out_dir, f"dist_{col}.png")
            fig.savefig(path)
            if show:
                plt.show()
            plt.close(fig)
        print(f"Saved histograms for numeric columns to {self.out_dir}")

    def plot_boxplots(self, numeric_cols: Optional[list] = None, show: bool = False):
        """Plot boxplots for numeric columns (one by one)."""
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.boxplot(self.df[col].dropna(), vert=False)
            ax.set_title(f"Boxplot: {col}")
            path = os.path.join(self.out_dir, f"boxplot_{col}.png")
            fig.tight_layout()
            fig.savefig(path)
            if show:
                plt.show()
            plt.close(fig)
        print(f"Saved boxplots for numeric columns to {self.out_dir}")

    # -------------------------
    # Categorical / target plots
    # -------------------------
    def plot_target_distribution(self, target_col: str, show: bool = False):
        """Bar plot for a categorical target column (counts and percentages)."""
        counts = self.df[target_col].value_counts(dropna=False)
        pct = self.df[target_col].value_counts(normalize=True, dropna=False) * 100
        summary = pd.concat([counts, pct], axis=1)
        summary.columns = ["count", "percent"]
        print(f"\nTarget distribution for '{target_col}':\n", summary)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(summary.index.astype(str), summary["count"])
        ax.set_title(f"Counts per {target_col}")
        ax.set_ylabel("count")
        ax.set_xlabel(target_col)
        fig.tight_layout()
        path = os.path.join(self.out_dir, f"target_dist_{target_col}.png")
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"Saved target distribution plot to {path}")
        return summary

    # -------------------------
    # Correlation and relationships
    # -------------------------
    def plot_correlation_heatmap(self, numeric_cols: Optional[list] = None, show: bool = False):
        """Plot and save correlation heatmap for numeric columns using matplotlib imshow."""
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        corr = self.df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(4, len(numeric_cols) * 0.4)))
        cax = ax.imshow(corr, interpolation="nearest", cmap="RdBu_r")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols)
        fig.colorbar(cax, ax=ax, fraction=0.025)
        ax.set_title("Correlation heatmap")
        fig.tight_layout()
        path = os.path.join(self.out_dir, "correlation_heatmap.png")
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"Saved correlation heatmap to {path}")
        return corr

    def plot_scatter_matrix(self, numeric_cols: Optional[list] = None, sample: Optional[int] = 1000, show: bool = False):
        """
        Save a scatter matrix (pairwise scatter) for numeric_cols.
        Use `sample` to limit size (avoid plotting entire large dataset).
        """
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        df_plot = self.df[numeric_cols].dropna()
        if sample is not None and len(df_plot) > sample:
            df_plot = df_plot.sample(sample, random_state=42)

        fig = scatter_matrix(df_plot, figsize=(len(numeric_cols) * 2, len(numeric_cols) * 2), diagonal="hist")
        # scatter_matrix returns numpy array of axes; we need to save figure
        plt.suptitle("Scatter matrix (sampled)", y=0.92)
        path = os.path.join(self.out_dir, "scatter_matrix.png")
        plt.tight_layout()
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
        print(f"Saved scatter matrix (sampled) to {path}")

    # -------------------------
    # Run everything
    # -------------------------
    def run_all(self, target_col: Optional[str] = None, show_plots: bool = False, scatter_sample: int = 1000) -> Dict[str, Any]:
        """
        Run all EDA steps and save artifacts.
        Returns a dict with key reports and file locations.
        """
        print("🚀 Running full EDA pipeline...")
        summary = self.summary()
        missing = self.missing_values_report()
        self.plot_numeric_distributions(show=show_plots)
        self.plot_boxplots(show=show_plots)

        if target_col is not None:
            target_summary = self.plot_target_distribution(target_col, show=show_plots)
        else:
            target_summary = None

        corr = self.plot_correlation_heatmap(show=show_plots)
        self.plot_scatter_matrix(sample=scatter_sample, show=show_plots)

        report = {
            "summary": summary,
            "missing_report": missing,
            "target_summary": target_summary,
            "correlation": corr,
            "out_dir": self.out_dir
        }
        print("✅ EDA pipeline finished.")
        return report