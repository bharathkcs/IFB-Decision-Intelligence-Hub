"""
AI-driven Revenue Leakage Detection and Forecasting System

This module provides a production-ready implementation for detecting and forecasting
revenue leakages across multiple dimensions including discounting, profitability,
product performance, and regional trends.

Architecture:
    - RevenueLeakageAnalyzer: Core analytics and ML logic
    - LLMInsightGenerator: LLM-based narrative insights
    - Streamlit UI functions: User interface layer

Author: IFB Decision Intelligence Hub
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RevenueLeakageConfig:
    """Configuration for revenue leakage analysis."""

    # Target margins
    target_margin: float = 0.15  # 15% default target margin
    category_margins: Dict[str, float] = field(default_factory=dict)

    # Thresholds
    high_discount_threshold: float = 0.20  # 20%
    low_margin_threshold: float = 0.05  # 5%
    anomaly_percentile: float = 0.95  # 95th percentile for anomalies

    # Forecasting
    forecast_horizon: int = 6  # months
    train_test_split: float = 0.75

    # Model parameters
    n_estimators: int = 100
    random_state: int = 42

    # Feature engineering
    lag_periods: List[int] = field(default_factory=lambda: [1, 3])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6])

    # Required/optional columns
    required_columns: List[str] = field(default_factory=lambda: ["Sales", "Profit"])
    optional_columns: List[str] = field(
        default_factory=lambda: [
            "Discount",
            "Category",
            "Sub Category",
            "Region",
            "State",
            "Customer Segment",
            "Order Date",
            "Order_Date",
        ]
    )


# ============================================================================
# Core Analytics Engine
# ============================================================================


class RevenueLeakageAnalyzer:
    """
    Core analytics engine for revenue leakage detection and forecasting.

    This class contains all business logic for:
    - Data preparation and validation
    - Feature engineering
    - Leakage metrics computation
    - Time-series forecasting
    - Anomaly detection
    - Risk scoring

    No Streamlit UI code is included in this class.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[RevenueLeakageConfig] = None,
    ) -> None:
        """
        Initialize the analyzer with data and configuration.

        Args:
            df: Input DataFrame containing sales/transaction data.
            config: Configuration object (uses defaults if None).

        Raises:
            ValueError: If required columns are missing.
        """
        self.df: pd.DataFrame = df.copy()
        self.config: RevenueLeakageConfig = config or RevenueLeakageConfig()
        self.date_column: Optional[str] = None
        self.prepared: bool = False

        self._validate_schema()

    def _validate_schema(self) -> None:
        """
        Validate that required columns exist in the dataset.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_cols = [
            col for col in self.config.required_columns if col not in self.df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(self.df.columns)}"
            )

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare and clean the dataset with comprehensive feature engineering.

        This method:
        - Validates and converts data types
        - Normalizes discount values
        - Computes leakage metrics
        - Creates time-based features

        Returns:
            Prepared DataFrame with all engineered features.
        """
        if self.prepared:
            return self.df

        # Convert numeric fields
        numeric_fields = ["Sales", "Profit", "Discount", "Quantity"]
        for field in numeric_fields:
            if field in self.df.columns:
                self.df[field] = pd.to_numeric(self.df[field], errors="coerce")

        # Detect and convert date columns
        self._detect_and_convert_dates()

        # Normalize discount logic
        if "Discount" in self.df.columns:
            self._normalize_discounts()

        # Add leakage metrics
        self._compute_leakage_metrics()

        # Add time-based features if date column exists
        if self.date_column:
            self._add_time_features()

        self.prepared = True
        return self.df

    def _detect_and_convert_dates(self) -> None:
        """Detect and convert date columns to datetime and pick a primary date column."""
        for col in self.df.columns:
            # Already datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                if self.date_column is None:
                    self.date_column = col
                continue

            # Column name suggests date
            if "date" in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    if self.date_column is None:
                        self.date_column = col
                except Exception:
                    # If conversion fails, just skip
                    pass

    def _normalize_discounts(self) -> None:
        """
        Normalize discount values to fractions and handle edge cases.

        Logic:
        - If max(|discount|) > 1.0, assume values are percentages and divide by 100.
        - Clip to [0, 0.9].
        - Replace negative/invalid values with 0.
        """
        if self.df["Discount"].abs().max() > 1.0:
            self.df["Discount"] = self.df["Discount"] / 100.0

        self.df["Discount"] = self.df["Discount"].clip(0, 0.9)
        self.df.loc[self.df["Discount"] < 0, "Discount"] = 0.0
        self.df.loc[self.df["Discount"] > 0.9, "Discount"] = 0.9

    def _compute_leakage_metrics(self) -> None:
        """
        Compute comprehensive leakage metrics.

        Creates:
            - Profit_Margin: Actual profit margin (%) with safe division.
            - Expected_Profit: Expected profit based on target margin (category-aware).
            - Margin_Leakage: Shortfall from expected profit (>= 0).
            - Discount_Leakage: Revenue lost to discounts (Sales * Discount).
            - Total_Leakage: Margin_Leakage + Discount_Leakage.
            - Leakage_Rate: Total_Leakage / Sales * 100 with safe division.
        """
        # Profit margin (%)
        self.df["Profit_Margin"] = np.where(
            self.df["Sales"] > 0,
            (self.df["Profit"] / self.df["Sales"]) * 100,
            0.0,
        )

        # Target margin (category-specific if provided)
        if "Category" in self.df.columns and self.config.category_margins:
            self.df["Target_Margin"] = (
                self.df["Category"]
                .map(self.config.category_margins)
                .fillna(self.config.target_margin)
            )
        else:
            self.df["Target_Margin"] = self.config.target_margin

        # Expected profit
        self.df["Expected_Profit"] = self.df["Sales"] * self.df["Target_Margin"]

        # Margin leakage (only shortfall)
        self.df["Margin_Leakage"] = np.maximum(
            self.df["Expected_Profit"] - self.df["Profit"], 0.0
        )

        # Discount leakage (simple model: Sales * Discount)
        if "Discount" in self.df.columns:
            self.df["Discount_Leakage"] = self.df["Sales"] * self.df["Discount"]
        else:
            self.df["Discount_Leakage"] = 0.0

        # Total leakage
        self.df["Total_Leakage"] = (
            self.df["Margin_Leakage"] + self.df["Discount_Leakage"]
        )

        # Leakage rate (% of sales)
        self.df["Leakage_Rate"] = np.where(
            self.df["Sales"] > 0,
            (self.df["Total_Leakage"] / self.df["Sales"]) * 100,
            0.0,
        )

    def _add_time_features(self) -> None:
        """Add time-based features for temporal analysis."""
        if not self.date_column or self.date_column not in self.df.columns:
            return

        date_col = self.df[self.date_column]

        self.df["Year"] = date_col.dt.year
        self.df["Month"] = date_col.dt.month
        self.df["Quarter"] = date_col.dt.quarter
        self.df["YearMonth"] = date_col.dt.to_period("M")
        self.df["DayOfWeek"] = date_col.dt.dayofweek

    def compute_leakage_metrics(self) -> Dict[str, float]:
        """
        Compute overall leakage metrics summary.

        Returns:
            Dictionary containing key leakage metrics.
        """
        if not self.prepared:
            self.prepare_data()

        total_sales = float(self.df["Sales"].sum())
        total_profit = float(self.df["Profit"].sum())

        avg_profit_margin = (
            float(total_profit / total_sales * 100) if total_sales > 0 else 0.0
        )

        return {
            "total_sales": total_sales,
            "total_profit": total_profit,
            "avg_profit_margin": avg_profit_margin,
            "total_discount_leakage": float(self.df["Discount_Leakage"].sum()),
            "total_margin_leakage": float(self.df["Margin_Leakage"].sum()),
            "total_leakage": float(self.df["Total_Leakage"].sum()),
            "avg_leakage_rate": float(self.df["Leakage_Rate"].mean()),
            "negative_profit_count": int((self.df["Profit"] < 0).sum()),
            "negative_profit_amount": float(
                self.df.loc[self.df["Profit"] < 0, "Profit"].sum()
            ),
        }

    def build_time_series(
        self,
        level: str = "overall",
        groupby_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build time series aggregation at specified level.

        Args:
            level: Aggregation level ("overall", "region", "category").
            groupby_col: Optional explicit column to group by.

        Returns:
            Time series DataFrame with YearMonth and aggregated metrics.

        Raises:
            ValueError: If no date column is available.
        """
        if not self.date_column:
            raise ValueError("No date column available for time series analysis")

        if not self.prepared:
            self.prepare_data()

        # Determine grouping columns
        if groupby_col:
            group_cols = ["YearMonth", groupby_col]
        elif level == "region" and "Region" in self.df.columns:
            group_cols = ["YearMonth", "Region"]
        elif level == "category" and "Category" in self.df.columns:
            group_cols = ["YearMonth", "Category"]
        else:
            group_cols = ["YearMonth"]

        ts_data = (
            self.df.groupby(group_cols)
            .agg(
                {
                    "Sales": "sum",
                    "Profit": "sum",
                    "Total_Leakage": "sum",
                    "Discount_Leakage": "sum",
                    "Margin_Leakage": "sum",
                }
            )
            .reset_index()
        )

        ts_data["Leakage_Rate"] = np.where(
            ts_data["Sales"] > 0,
            (ts_data["Total_Leakage"] / ts_data["Sales"]) * 100,
            0.0,
        )

        return ts_data

    def _add_lag_rolling_features(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag and rolling window features to a time series dataframe.

        Args:
            ts_df: Time series DataFrame with 'Total_Leakage' column.

        Returns:
            DataFrame with additional lag and rolling features.
        """
        df = ts_df.copy()

        # Lag features
        for lag in self.config.lag_periods:
            df[f"Leakage_Lag{lag}"] = df["Total_Leakage"].shift(lag)

        # Rolling mean features
        for window in self.config.rolling_windows:
            df[f"Rolling_{window}M_Leakage"] = (
                df["Total_Leakage"].rolling(window=window, min_periods=1).mean()
            )

        return df

    def forecast_leakage(
        self,
        forecast_periods: int = 6,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Forecast future revenue leakage using a tree-based regressor.

        Steps:
        - Aggregate data to monthly level.
        - Engineer lag and rolling features.
        - Train GradientBoostingRegressor with time-based split.
        - Compute MAE, RMSE, MAPE (if possible).
        - Iteratively forecast future periods.

        Args:
            forecast_periods: Number of months to forecast.

        Returns:
            (forecast_df, metrics_dict)
                forecast_df: DataFrame with Date, Historical_Leakage, Forecasted_Leakage.
                metrics_dict: Model performance and summary metrics.

        Raises:
            ValueError: If insufficient data for forecasting.
        """
        if not self.date_column:
            raise ValueError("No date column available for forecasting")

        if not self.prepared:
            self.prepare_data()

        # Build monthly time series at overall level
        monthly_data = self.build_time_series(level="overall")

        # Remove rows without YearMonth (just in case)
        monthly_data = monthly_data.dropna(subset=["YearMonth"])

        # Convert to datetime index
        monthly_data["Date"] = monthly_data["YearMonth"].dt.to_timestamp()
        monthly_data = monthly_data.set_index("Date").sort_index()

        if len(monthly_data) < 6:
            raise ValueError(
                f"Insufficient data for forecasting. Need at least 6 months, "
                f"got {len(monthly_data)}"
            )

        # Add lag/rolling features
        monthly_data = self._add_lag_rolling_features(monthly_data)

        # Drop initial NaNs from lag/rolling
        monthly_data = monthly_data.dropna()

        if len(monthly_data) < 4:
            raise ValueError("Insufficient data after feature engineering")

        # Calendar features
        monthly_data["Month"] = monthly_data.index.month
        monthly_data["Quarter"] = monthly_data.index.quarter
        monthly_data["Year"] = monthly_data.index.year

        feature_cols: List[str] = ["Month", "Quarter", "Year"]

        for lag in self.config.lag_periods:
            col = f"Leakage_Lag{lag}"
            if col in monthly_data.columns:
                feature_cols.append(col)

        for window in self.config.rolling_windows:
            col = f"Rolling_{window}M_Leakage"
            if col in monthly_data.columns:
                feature_cols.append(col)

        # Build X, y and clean NaNs / infs
        X = monthly_data[feature_cols].copy()
        y = monthly_data["Total_Leakage"].copy()

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        valid_mask = (~X.isna().any(axis=1)) & (~y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        monthly_data = monthly_data.loc[valid_mask]

        if len(X) < 4:
            raise ValueError("Insufficient clean data for forecasting after NaN removal")

        # Time-based split
        split_idx = int(len(X) * self.config.train_test_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            learning_rate=0.1,
            max_depth=4,
        )
        model.fit(X_train, y_train)

        # Evaluate on test set
        metrics: Dict[str, float] = {}
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            # MAPE (avoid division by zero)
            mask = y_test != 0
            if mask.sum() > 0:
                mape = (
                    np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                )
                metrics["mape"] = float(mape)

        # Future forecasting
        last_date = monthly_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq="MS",
        )

        future_predictions: List[float] = []
        forecast_df = monthly_data.copy()

        for future_date in future_dates:
            future_row: Dict[str, float] = {
                "Month": float(future_date.month),
                "Quarter": float(future_date.quarter),
                "Year": float(future_date.year),
            }

            # Lag features from recent history / forecasts
            for lag in self.config.lag_periods:
                lag_idx = len(forecast_df) - lag
                if lag_idx >= 0:
                    prev_row = forecast_df.iloc[lag_idx]

                    if (
                        "Forecast_Leakage" in prev_row.index
                        and not pd.isna(prev_row["Forecast_Leakage"])
                    ):
                        value = float(prev_row["Forecast_Leakage"])
                    elif not pd.isna(prev_row.get("Total_Leakage", 0.0)):
                        value = float(prev_row["Total_Leakage"])
                    else:
                        value = 0.0
                else:
                    value = 0.0
                future_row[f"Leakage_Lag{lag}"] = value

            # Rolling features mixing historical + forecasted
            for window in self.config.rolling_windows:
                window_data = forecast_df.tail(window)
                values: List[float] = []
                for _, row in window_data.iterrows():
                    if (
                        "Forecast_Leakage" in row.index
                        and not pd.isna(row["Forecast_Leakage"])
                    ):
                        values.append(float(row["Forecast_Leakage"]))
                    elif not pd.isna(row.get("Total_Leakage", 0.0)):
                        values.append(float(row["Total_Leakage"]))
                    else:
                        values.append(0.0)
                future_row[f"Rolling_{window}M_Leakage"] = (
                    float(np.mean(values)) if values else 0.0
                )

            # Predict
            X_future = pd.DataFrame([future_row])[feature_cols]
            X_future = X_future.apply(pd.to_numeric, errors="coerce")
            X_future = X_future.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            prediction = float(model.predict(X_future)[0])
            future_predictions.append(prediction)

            # Append to forecast_df for subsequent lags/rolling
            new_row = pd.DataFrame(
                {
                    "Total_Leakage": [0.0],
                    "Forecast_Leakage": [prediction],
                },
                index=[future_date],
            )
            forecast_df = pd.concat([forecast_df, new_row])

        # Build output dataframe
        result_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Forecasted_Leakage": future_predictions,
            }
        )

        # Historical leakage
        historical_df = pd.DataFrame(
            {
                "Date": monthly_data.index,
                "Historical_Leakage": monthly_data["Total_Leakage"].values,
            }
        )

        # Combine
        full_forecast = (
            pd.concat(
                [
                    historical_df.set_index("Date"),
                    result_df.set_index("Date"),
                ],
                axis=1,
            )
            .reset_index()
            .rename(
                columns={
                    "index": "Date",
                }
            )
        )
        full_forecast.columns = ["Date", "Historical_Leakage", "Forecasted_Leakage"]

        # Summary metrics
        metrics["total_forecasted_leakage"] = float(sum(future_predictions))
        metrics["avg_monthly_forecast"] = float(np.mean(future_predictions))
        metrics["historical_avg"] = float(monthly_data["Total_Leakage"].mean())

        return full_forecast, metrics

    def detect_anomalies(
        self,
        contamination: float = 0.05,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect anomalous transactions using regression residuals and IsolationForest.

        Steps:
        - Train regression model to predict expected profit.
        - Compute residuals (expected - actual).
        - Mark residual-based anomalies (high residual & low/negative profit).
        - Train IsolationForest on feature space.
        - Flag final anomalies where either rule or IsolationForest triggers.

        Args:
            contamination: Expected proportion of outliers (0–1).

        Returns:
            (anomalies_df, summary_dict)
        """
        if not self.prepared:
            self.prepare_data()

        feature_cols: List[str] = [
            "Sales",
            "Profit",
            "Profit_Margin",
            "Margin_Leakage",
            "Total_Leakage",
        ]

        if "Discount" in self.df.columns:
            feature_cols.append("Discount")

        df_encoded = self.df.copy()

        # Simple label-encoding of categoricals
        for col in ["Category", "Sub Category", "Region", "Customer Segment"]:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f"{col}_Encoded"] = le.fit_transform(
                    df_encoded[col].astype(str)
                )
                feature_cols.append(f"{col}_Encoded")

        df_features = df_encoded[feature_cols].copy()
        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_features) < 100:
            raise ValueError(
                "Insufficient data for anomaly detection (need >= 100 records)"
            )

        # Regression model: predict Profit
        X = df_features.drop(["Profit"], axis=1, errors="ignore")
        y = df_features["Profit"]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, _ = y.iloc[:split_idx], y.iloc[split_idx:]

        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=self.config.random_state,
            max_depth=4,
        )
        model.fit(X_train, y_train)

        # Predict on full feature set
        y_pred_all = model.predict(X)
        residuals = y_pred_all - y.values

        df_features["Residual"] = residuals
        df_features["Expected_Profit_Model"] = y_pred_all

        # Residual-based anomalies
        residual_threshold = np.percentile(
            residuals, self.config.anomaly_percentile * 100
        )
        df_features["Is_Anomaly_Residual"] = (
            (df_features["Residual"] > residual_threshold)
            & (df_features["Profit"] < df_features["Profit"].quantile(0.25))
        )

        # IsolationForest anomalies
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.config.random_state,
        )
        iso_predictions = iso_forest.fit_predict(X)
        df_features["Is_Anomaly_ISO"] = iso_predictions == -1

        # Final anomaly flag (union)
        df_features["Is_Anomaly"] = df_features["Is_Anomaly_Residual"] | df_features[
            "Is_Anomaly_ISO"
        ]

        # Normalized anomaly score
        df_features["Anomaly_Score"] = (
            df_features["Residual"] - residuals.mean()
        ) / residuals.std()

        # Join back with original df
        anomaly_indices = df_features[df_features["Is_Anomaly"]].index
        anomalies_full = self.df.loc[anomaly_indices].copy()
        anomalies_full["Residual"] = df_features.loc[anomaly_indices, "Residual"]
        anomalies_full["Anomaly_Score"] = df_features.loc[
            anomaly_indices, "Anomaly_Score"
        ]

        # Add human-readable flags
        anomalies_full["Flags"] = anomalies_full.apply(
            lambda row: self._generate_anomaly_flags(row), axis=1
        )

        summary = {
            "total_transactions": len(self.df),
            "anomaly_count": len(anomalies_full),
            "anomaly_percentage": len(anomalies_full) / len(self.df) * 100.0,
            "total_sales_at_risk": float(anomalies_full["Sales"].sum()),
            "total_leakage_in_anomalies": float(
                anomalies_full["Total_Leakage"].sum()
            ),
            "avg_anomaly_score": float(anomalies_full["Anomaly_Score"].mean()),
            "residual_threshold": float(residual_threshold),
        }

        return anomalies_full, summary

    def _generate_anomaly_flags(self, row: pd.Series) -> str:
        """
        Generate descriptive flags for an anomaly row.

        Returns:
            Comma-separated string of flags.
        """
        flags: List[str] = []

        if row.get("Profit", 0.0) < 0:
            flags.append("NEGATIVE_PROFIT")

        if "Discount" in row and row.get("Discount", 0.0) > self.config.high_discount_threshold:
            flags.append("HIGH_DISCOUNT")

        if row.get("Profit_Margin", 0.0) < self.config.low_margin_threshold * 100:
            flags.append("LOW_MARGIN")

        if row.get("Margin_Leakage", 0.0) > row.get("Expected_Profit", 0.0) * 0.5:
            flags.append("HIGH_MARGIN_LEAKAGE")

        return ", ".join(flags) if flags else "OUTLIER"

    def score_leakage_risk(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Score each transaction for leakage risk using classification.

        Steps:
        - Construct Leakage_Flag: 1 if Total_Leakage above 75th percentile or Profit_Margin < 0.
        - Train GradientBoostingClassifier to predict Leakage_Flag.
        - Output leakage probability for each transaction.

        Returns:
            (risk_scores_df, metadata_dict)
        """
        if not self.prepared:
            self.prepare_data()

        leakage_threshold = self.df["Total_Leakage"].quantile(0.75)
        self.df["Leakage_Flag"] = (
            (self.df["Total_Leakage"] > leakage_threshold)
            | (self.df["Profit_Margin"] < 0)
        ).astype(int)

        feature_cols: List[str] = ["Sales", "Profit_Margin"]

        if "Discount" in self.df.columns:
            feature_cols.append("Discount")

        df_encoded = self.df.copy()
        for col in ["Category", "Sub Category", "Region", "Customer Segment"]:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f"{col}_Encoded"] = le.fit_transform(
                    df_encoded[col].astype(str)
                )
                feature_cols.append(f"{col}_Encoded")

        df_model = df_encoded[feature_cols + ["Leakage_Flag"]].copy()
        df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

        X = df_model[feature_cols]
        y = df_model["Leakage_Flag"]

        if len(X) < 20:
            raise ValueError("Insufficient data for risk scoring")

        clf = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            max_depth=4,
        )
        clf.fit(X, y)

        leakage_probs = clf.predict_proba(X)[:, 1]

        result_df = self.df.loc[df_model.index].copy()
        result_df["Leakage_Probability"] = leakage_probs

        feature_importance = (
            pd.DataFrame(
                {
                    "Feature": feature_cols,
                    "Importance": clf.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        metadata = {
            "feature_importance": feature_importance.to_dict("records"),
            "high_risk_count": int((leakage_probs > 0.7).sum()),
            "medium_risk_count": int(
                ((leakage_probs > 0.4) & (leakage_probs <= 0.7)).sum()
            ),
            "low_risk_count": int((leakage_probs <= 0.4).sum()),
            "avg_probability": float(leakage_probs.mean()),
        }

        return result_df, metadata


# ============================================================================
# LLM Insight Generator
# ============================================================================


class LLMInsightGenerator:
    """
    Handles LLM-based narrative insight generation.

    This class wraps LLM client interactions and provides structured
    methods for generating business insights from numeric data.

    Important:
        - This class NEVER asks the LLM to do numeric calculations.
        - It only passes computed numbers and asks for interpretation.
    """

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize with LLM client.

        Args:
            llm_client: LLM client with a conversational_response method.
        """
        self.llm = llm_client

    def _call_llm(self, prompt: str, timeout: int = 30) -> str:
        """
        Safely call LLM with error handling.

        Args:
            prompt: Prompt text.
            timeout: Timeout in seconds (not enforced here, reserved for future use).

        Returns:
            LLM response text or error message.
        """
        try:
            response = self.llm.conversational_response(
                [
                    {"sender": "user", "text": prompt},
                ]
            )
            return response.get("text", "No response generated")
        except Exception as e:
            return f"⚠️ Error generating insights: {str(e)}"

    def generate_executive_summary(self, metrics: Dict[str, float]) -> str:
        """
        Generate executive summary insights from metrics.

        Args:
            metrics: Dictionary of computed metrics.

        Returns:
            Executive summary text.
        """
        prompt = f"""
As a business analyst, provide a concise executive summary of this revenue leakage analysis.

**Key Metrics:**
- Total Sales: ${metrics.get('total_sales', 0):,.2f}
- Total Profit: ${metrics.get('total_profit', 0):,.2f}
- Average Profit Margin: {metrics.get('avg_profit_margin', 0):.2f}%
- Total Revenue Leakage: ${metrics.get('total_leakage', 0):,.2f}
- Discount Leakage: ${metrics.get('total_discount_leakage', 0):,.2f}
- Margin Leakage: ${metrics.get('total_margin_leakage', 0):,.2f}
- Negative Profit Transactions: {metrics.get('negative_profit_count', 0)}

Please respond with:
1. Top 3 critical findings (bullet points).
2. Impact assessment (1–2 sentences on business impact).
3. Priority areas (what needs immediate attention).

Use clear, business-focused language.
"""
        return self._call_llm(prompt)

    def generate_discount_insights(
        self,
        discount_stats: Dict[str, float],
        category_stats: Optional[pd.Series] = None,
    ) -> str:
        """
        Generate insights on discount patterns.

        Args:
            discount_stats: Discount distribution statistics.
            category_stats: Optional Series of category-level discount leakage.

        Returns:
            Discount insights text.
        """
        category_info = ""
        if category_stats is not None and not category_stats.empty:
            top_3 = category_stats.head(3)
            category_info = "\nTop 3 categories by discount leakage:\n"
            for category, leakage in top_3.items():
                category_info += f"- {category}: ${leakage:,.2f}\n"

        prompt = f"""
Analyze these discount patterns and provide strategic insights.

**Discount Statistics:**
- Average Discount Rate: {discount_stats.get('mean', 0):.2%}
- Median Discount: {discount_stats.get('50%', 0):.2%}
- Max Discount: {discount_stats.get('max', 0):.2%}
- High Discount (>20%) Transactions: {discount_stats.get('high_discount_count', 0)}
- Revenue Lost to High Discounts: ${discount_stats.get('high_discount_loss', 0):,.2f}

{category_info}

Please respond with:
1. Whether current discount levels appear sustainable or excessive.
2. Which categories show problematic discount patterns.
3. Specific, actionable recommendations to optimize discounting.

Use concise bullets and business language.
"""
        return self._call_llm(prompt)

    def generate_forecast_insights(
        self,
        forecast_summary: Dict[str, float],
    ) -> str:
        """
        Generate insights on leakage forecast.

        Args:
            forecast_summary: Forecast metrics and trends.

        Returns:
            Forecast insights text.
        """
        trend = (
            "increasing"
            if forecast_summary.get("avg_monthly_forecast", 0)
            > forecast_summary.get("historical_avg", 0)
            else "decreasing"
        )

        prompt = f"""
Analyze this revenue leakage forecast and provide strategic insights.

**Forecast Summary:**
- Historical Average Monthly Leakage: ${forecast_summary.get('historical_avg', 0):,.2f}
- Forecasted Average Monthly Leakage: ${forecast_summary.get('avg_monthly_forecast', 0):,.2f}
- Total Forecasted Leakage: ${forecast_summary.get('total_forecasted_leakage', 0):,.2f}
- Trend Direction: {trend}
- Model MAE: ${forecast_summary.get('mae', 0):,.2f}
- Model RMSE: ${forecast_summary.get('rmse', 0):,.2f}

Please respond with:
1. Trend analysis: what the forecast suggests about future leakage.
2. Business impact: potential financial implications.
3. Preventive actions: specific steps to mitigate forecasted leakage.

Be concise and action-oriented.
"""
        return self._call_llm(prompt)

    def generate_recommendations(
        self,
        summary: Dict[str, Any],
        top_categories: Optional[List[str]] = None,
        top_regions: Optional[List[str]] = None,
    ) -> str:
        """
        Generate comprehensive recommendations to reduce revenue leakage.

        Args:
            summary: Overall analysis summary metrics.
            top_categories: List of worst-performing categories.
            top_regions: List of worst-performing regions.

        Returns:
            Comprehensive recommendations text.
        """
        category_info = ""
        if top_categories:
            category_info = (
                f"\nWorst performing categories (by profit): {', '.join(top_categories)}"
            )

        region_info = ""
        if top_regions:
            region_info = (
                f"\nWorst performing regions (by profit): {', '.join(top_regions)}"
            )

        prompt = f"""
As a business strategy consultant, provide detailed, actionable recommendations to reduce revenue leakage.

**Current Situation:**
- Total Sales: ${summary.get('total_sales', 0):,.2f}
- Total Profit: ${summary.get('total_profit', 0):,.2f}
- Total Revenue Leakage: ${summary.get('total_leakage', 0):,.2f}
- Average Profit Margin: {summary.get('avg_profit_margin', 0):.2f}%
- Negative Profit Transactions: {summary.get('negative_profit_count', 0)}
{category_info}
{region_info}

Please structure your answer as:

1. Immediate Actions (0–30 days)
   - Quick wins to stop leakage and emergency measures.

2. Short-term Strategy (1–3 months)
   - Pricing optimization, discount policy improvements, product mix adjustments.

3. Long-term Strategy (3–12 months)
   - Strategic business model changes and portfolio restructuring.

4. Key Metrics to Track
   - Specific KPIs and suggested target ranges.

5. Expected Impact
   - Revenue recovery potential and indicative timelines.

Use clear headings and bullets. Be as concrete and practical as possible.
"""
        return self._call_llm(prompt)


# ============================================================================
# Streamlit UI Layer
# ============================================================================


def run_revenue_leakage_app(df: pd.DataFrame, llm_client: Any) -> None:
    """
    Main Streamlit application for revenue leakage analysis.

    Args:
        df: Input DataFrame with transaction data.
        llm_client: LLM client instance used for narrative insights.
    """
    st.header("🔍 AI-Powered Revenue Leakage Analysis")
    st.markdown(
        """
    Comprehensive revenue leakage detection across discounting, profitability,
    product performance, regional trends, and predictive forecasting.
    """
    )

    # Configuration sidebar
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        target_margin = (
            st.slider(
                "Target Profit Margin (%)",
                5,
                30,
                15,
                help="Expected profit margin target.",
            )
            / 100
        )

        high_discount_thresh = (
            st.slider(
                "High Discount Threshold (%)",
                10,
                40,
                20,
                help="Threshold for flagging high discounts.",
            )
            / 100
        )

        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            3,
            12,
            6,
            help="Number of months to forecast.",
        )

        anomaly_sensitivity = (
            st.slider(
                "Anomaly Detection Sensitivity (%)",
                1,
                20,
                5,
                help="Expected percentage of outliers (higher = more anomalies).",
            )
            / 100
        )

    # Create configuration
    config = RevenueLeakageConfig(
        target_margin=target_margin,
        high_discount_threshold=high_discount_thresh,
        forecast_horizon=forecast_horizon,
    )

    # Initialize analyzer and LLM wrapper
    try:
        analyzer = RevenueLeakageAnalyzer(df, config)
        analyzer.prepare_data()
        llm_generator = LLMInsightGenerator(llm_client)
    except ValueError as e:
        st.error(f"❌ Data validation error: {str(e)}")
        return

    # Tabs
    tabs = st.tabs(
        [
            "📊 Executive Summary",
            "💰 Discount Analysis",
            "📉 Profit Erosion",
            "📦 Product Performance",
            "🌍 Regional Analysis",
            "🔮 Forecasting",
            "📈 Anomaly Detection",
            "📋 Recommendations",
        ]
    )

    with tabs[0]:
        _render_executive_summary(analyzer, llm_generator)

    with tabs[1]:
        _render_discount_analysis(analyzer, llm_generator)

    with tabs[2]:
        _render_profit_erosion(analyzer)

    with tabs[3]:
        _render_product_performance(analyzer)

    with tabs[4]:
        _render_regional_analysis(analyzer)

    with tabs[5]:
        _render_forecasting(analyzer, llm_generator, forecast_horizon)

    with tabs[6]:
        _render_anomaly_detection(analyzer, anomaly_sensitivity)

    with tabs[7]:
        _render_recommendations(analyzer, llm_generator)


def _render_executive_summary(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator,
) -> None:
    """Render executive summary tab."""
    st.subheader("📊 Executive Summary - Revenue Leakage Overview")

    metrics = analyzer.compute_leakage_metrics()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sales", f"${metrics['total_sales']:,.2f}")
        st.metric("Total Profit", f"${metrics['total_profit']:,.2f}")

    with col2:
        st.metric("Avg Profit Margin", f"{metrics['avg_profit_margin']:.2f}%")
        avg_discount = (
            analyzer.df["Discount"].mean() * 100
            if "Discount" in analyzer.df.columns
            else 0.0
        )
        st.metric("Avg Discount", f"{avg_discount:.2f}%")

    with col3:
        st.metric(
            "Discount Leakage",
            f"${metrics['total_discount_leakage']:,.2f}",
        )
        st.metric(
            "Margin Leakage",
            f"${metrics['total_margin_leakage']:,.2f}",
        )

    with col4:
        st.metric(
            "Total Revenue Leakage",
            f"${metrics['total_leakage']:,.2f}",
            delta=f"-{metrics['avg_leakage_rate']:.2f}%",
            delta_color="inverse",
        )
        potential_savings = metrics["total_leakage"] * 0.7
        st.metric("Potential Savings (70%)", f"${potential_savings:,.2f}")

    # Breakdown chart
    st.subheader("Revenue Leakage Breakdown")

    leakage_data = pd.DataFrame(
        {
            "Category": [
                "Discount Leakage",
                "Margin Leakage",
                "Potential Savings",
            ],
            "Amount": [
                metrics["total_discount_leakage"],
                metrics["total_margin_leakage"],
                potential_savings,
            ],
        }
    )

    fig = px.bar(
        leakage_data,
        x="Category",
        y="Amount",
        title="Revenue Leakage Components",
        color="Category",
        color_discrete_map={
            "Discount Leakage": "#ff6b6b",
            "Margin Leakage": "#ee5a6f",
            "Potential Savings": "#4ecdc4",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI insights
    st.subheader("🤖 AI-Generated Executive Insights")
    with st.spinner("Generating insights..."):
        insights = llm_gen.generate_executive_summary(metrics)
    st.write(insights)


def _render_discount_analysis(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator,
) -> None:
    """Render discount analysis tab."""
    st.subheader("💰 Discount-Driven Revenue Leakage Analysis")

    if "Discount" not in analyzer.df.columns:
        st.warning("⚠️ No discount data available in the dataset.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Discount Distribution")
        fig = px.histogram(
            analyzer.df,
            x="Discount",
            nbins=50,
            title="Distribution of Discount Rates",
            labels={"Discount": "Discount Rate", "count": "Frequency"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### High Discount Impact")
        high_discount = analyzer.df[
            analyzer.df["Discount"] > analyzer.config.high_discount_threshold
        ]
        high_discount_loss = high_discount["Discount_Leakage"].sum()

        st.metric(
            f"Transactions with >{analyzer.config.high_discount_threshold*100:.0f}% Discount",
            len(high_discount),
        )
        st.metric("Revenue Lost (High Discounts)", f"${high_discount_loss:,.2f}")

        if "Category" in analyzer.df.columns:
            st.markdown("##### Discount by Category")
            discount_by_cat = (
                analyzer.df.groupby("Category")
                .agg(
                    {
                        "Discount": "mean",
                        "Discount_Leakage": "sum",
                    }
                )
                .sort_values("Discount_Leakage", ascending=False)
            )

            st.dataframe(
                discount_by_cat.style.format(
                    {
                        "Discount": "{:.2%}",
                        "Discount_Leakage": "${:,.2f}",
                    }
                ),
                height=200,
            )

    # Discount vs profit margin
    st.markdown("#### Discount Impact on Profitability")

    sample_size = min(1000, len(analyzer.df))
    sample_df = analyzer.df.sample(sample_size, random_state=42)

    fig = px.scatter(
        sample_df,
        x="Discount",
        y="Profit_Margin",
        color="Category" if "Category" in sample_df.columns else None,
        title="Discount Rate vs Profit Margin",
        labels={"Discount": "Discount Rate", "Profit_Margin": "Profit Margin (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI discount insights
    st.markdown("#### 🤖 AI-Generated Discount Insights")

    discount_stats = {
        "mean": analyzer.df["Discount"].mean(),
        "50%": analyzer.df["Discount"].median(),
        "max": analyzer.df["Discount"].max(),
        "high_discount_count": len(high_discount),
        "high_discount_loss": high_discount_loss,
    }

    category_stats: Optional[pd.Series] = None
    if "Category" in analyzer.df.columns:
        category_stats = (
            analyzer.df.groupby("Category")["Discount_Leakage"]
            .sum()
            .sort_values(ascending=False)
        )

    with st.spinner("Analyzing discount patterns..."):
        insights = llm_gen.generate_discount_insights(discount_stats, category_stats)
    st.write(insights)


def _render_profit_erosion(analyzer: RevenueLeakageAnalyzer) -> None:
    """Render profit erosion analysis tab."""
    st.subheader("📉 Profit Margin Erosion Analysis")

    # Time-series trend
    if analyzer.date_column:
        try:
            ts_data = analyzer.build_time_series(level="overall")
            ts_data["Date"] = ts_data["YearMonth"].dt.to_timestamp()

            ts_data["Profit_Margin"] = np.where(
                ts_data["Sales"] > 0,
                (ts_data["Profit"] / ts_data["Sales"]) * 100,
                0.0,
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts_data["Date"],
                    y=ts_data["Profit_Margin"],
                    mode="lines+markers",
                    name="Avg Profit Margin",
                    line=dict(color="#4ecdc4", width=3),
                )
            )
            fig.update_layout(
                title="Profit Margin Trend Over Time",
                xaxis_title="Date",
                yaxis_title="Profit Margin (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Simple linear trend
            if len(ts_data) > 2:
                from scipy import stats

                ts_data["Period_Num"] = range(len(ts_data))
                slope, _, _, _, _ = stats.linregress(
                    ts_data["Period_Num"], ts_data["Profit_Margin"]
                )

                trend_direction = "increasing ⬆️" if slope > 0 else "decreasing ⬇️"
                st.info(
                    f"📊 Profit margin trend is **{trend_direction}** "
                    f"at approximately {abs(slope):.2f}% per month."
                )
        except Exception as e:
            st.warning(f"Could not generate time series: {str(e)}")

    # Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Profit Margin Distribution")
        fig = px.histogram(
            analyzer.df,
            x="Profit_Margin",
            nbins=50,
            title="Distribution of Profit Margins",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Low/Negative Margin Analysis")
        low_margin = analyzer.df[
            analyzer.df["Profit_Margin"]
            < analyzer.config.low_margin_threshold * 100
        ]
        negative_margin = analyzer.df[analyzer.df["Profit_Margin"] < 0]

        st.metric(
            f"Transactions with <{analyzer.config.low_margin_threshold*100:.0f}% Margin",
            len(low_margin),
        )
        st.metric("Negative Margin Transactions", len(negative_margin))
        st.metric(
            "Total Loss (Negative Margin)",
            f"${negative_margin['Profit'].sum():,.2f}"
            if len(negative_margin) > 0
            else "$0.00",
        )

    # Category analysis
    if "Category" in analyzer.df.columns:
        st.markdown("#### Category-wise Profit Performance")

        cat_profit = (
            analyzer.df.groupby("Category")
            .agg(
                {
                    "Sales": "sum",
                    "Profit": "sum",
                    "Profit_Margin": "mean",
                }
            )
            .sort_values("Profit_Margin")
        )

        fig = px.bar(
            cat_profit.reset_index(),
            x="Category",
            y="Profit_Margin",
            title="Average Profit Margin by Category",
            color="Profit_Margin",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            cat_profit.style.format(
                {
                    "Sales": "${:,.2f}",
                    "Profit": "${:,.2f}",
                    "Profit_Margin": "{:.2f}%",
                }
            )
        )


def _render_product_performance(analyzer: RevenueLeakageAnalyzer) -> None:
    """Render product/category performance analysis tab."""
    st.subheader("📦 Product & Category Performance Analysis")

    if "Category" not in analyzer.df.columns:
        st.warning("⚠️ No category data available.")
        return

    cat_performance = (
        analyzer.df.groupby("Category")
        .agg(
            {
                "Sales": "sum",
                "Profit": "sum",
                "Discount": "mean"
                if "Discount" in analyzer.df.columns
                else (lambda x: 0.0),
            }
        )
        .sort_values("Profit")
    )

    cat_performance["Profit_Margin"] = np.where(
        cat_performance["Sales"] > 0,
        (cat_performance["Profit"] / cat_performance["Sales"]) * 100,
        0.0,
    )

    # Scatter: sales vs profit
    fig = px.scatter(
        cat_performance.reset_index(),
        x="Sales",
        y="Profit",
        size="Discount" if "Discount" in analyzer.df.columns else None,
        color="Profit_Margin",
        hover_data=["Category"],
        title="Category Performance: Sales vs Profit",
        color_continuous_scale="RdYlGn",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌟 Top Performing Categories")
        top_cats = cat_performance.nlargest(5, "Profit")
        st.dataframe(
            top_cats.style.format(
                {
                    "Sales": "${:,.2f}",
                    "Profit": "${:,.2f}",
                    "Discount": "{:.2%}",
                    "Profit_Margin": "{:.2f}%",
                }
            ).background_gradient(cmap="Greens", subset=["Profit"])
        )

    with col2:
        st.markdown("#### ⚠️ Bottom Performing Categories")
        bottom_cats = cat_performance.nsmallest(5, "Profit")
        st.dataframe(
            bottom_cats.style.format(
                {
                    "Sales": "${:,.2f}",
                    "Profit": "${:,.2f}",
                    "Discount": "{:.2%}",
                    "Profit_Margin": "{:.2f}%",
                }
            ).background_gradient(cmap="Reds", subset=["Profit"])
        )

    if "Sub Category" in analyzer.df.columns:
        st.markdown("#### Sub-Category Deep Dive")

        subcat_performance = (
            analyzer.df.groupby("Sub Category")
            .agg(
                {
                    "Sales": "sum",
                    "Profit": "sum",
                    "Discount": "mean"
                    if "Discount" in analyzer.df.columns
                    else (lambda x: 0.0),
                }
            )
            .sort_values("Profit")
        )

        subcat_performance["Profit_Margin"] = np.where(
            subcat_performance["Sales"] > 0,
            (subcat_performance["Profit"] / subcat_performance["Sales"]) * 100,
            0.0,
        )

        filter_option = st.radio(
            "Show:", ["All", "Profitable Only", "Loss-Making Only"]
        )

        if filter_option == "Profitable Only":
            subcat_display = subcat_performance[subcat_performance["Profit"] > 0]
        elif filter_option == "Loss-Making Only":
            subcat_display = subcat_performance[subcat_performance["Profit"] < 0]
        else:
            subcat_display = subcat_performance

        st.dataframe(
            subcat_display.style.format(
                {
                    "Sales": "${:,.2f}",
                    "Profit": "${:,.2f}",
                    "Discount": "{:.2%}",
                    "Profit_Margin": "{:.2f}%",
                }
            ).background_gradient(cmap="RdYlGn", subset=["Profit_Margin"])
        )


def _render_regional_analysis(analyzer: RevenueLeakageAnalyzer) -> None:
    """Render regional performance analysis tab."""
    st.subheader("🌍 Regional Performance Analysis")

    region_col: Optional[str] = None
    if "Region" in analyzer.df.columns:
        region_col = "Region"
    elif "State" in analyzer.df.columns:
        region_col = "State"

    if not region_col:
        st.warning("⚠️ No regional data available.")
        return

    regional_metrics = analyzer.df.groupby(region_col).agg(
        {
            "Sales": "sum",
            "Profit": "sum",
            "Discount": "mean"
            if "Discount" in analyzer.df.columns
            else (lambda x: 0.0),
        }
    )

    regional_metrics["Profit_Margin"] = np.where(
        regional_metrics["Sales"] > 0,
        (regional_metrics["Profit"] / regional_metrics["Sales"]) * 100,
        0.0,
    )
    regional_metrics = regional_metrics.sort_values("Profit", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            regional_metrics.reset_index(),
            x=region_col,
            y="Sales",
            title=f"Sales by {region_col}",
            color="Profit_Margin",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            regional_metrics.reset_index(),
            values="Profit",
            names=region_col,
            title=f"Profit Distribution by {region_col}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### {region_col} Performance Table")
    st.dataframe(
        regional_metrics.style.format(
            {
                "Sales": "${:,.2f}",
                "Profit": "${:,.2f}",
                "Discount": "{:.2%}",
                "Profit_Margin": "{:.2f}%",
            }
        ).background_gradient(cmap="RdYlGn", subset=["Profit_Margin"])
    )

    avg_margin = regional_metrics["Profit_Margin"].mean()
    problem_regions = regional_metrics[regional_metrics["Profit_Margin"] < avg_margin]

    if len(problem_regions) > 0:
        st.warning(
            f"⚠️ {len(problem_regions)} regions performing below "
            f"average profit margin ({avg_margin:.2f}%)."
        )
        st.dataframe(
            problem_regions.style.format(
                {
                    "Sales": "${:,.2f}",
                    "Profit": "${:,.2f}",
                    "Discount": "{:.2%}",
                    "Profit_Margin": "{:.2f}%",
                }
            )
        )


def _render_forecasting(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator,
    forecast_periods: int,
) -> None:
    """Render forecasting analysis tab."""
    st.subheader("🔮 Revenue Leakage Forecasting")

    if not analyzer.date_column:
        st.warning("⚠️ No date column found for time series forecasting.")
        return

    try:
        with st.spinner("Training forecasting model..."):
            forecast_df, metrics = analyzer.forecast_leakage(forecast_periods)

        fig = go.Figure()

        historical = forecast_df.dropna(subset=["Historical_Leakage"])
        fig.add_trace(
            go.Scatter(
                x=historical["Date"],
                y=historical["Historical_Leakage"],
                mode="lines+markers",
                name="Historical Leakage",
                line=dict(color="#ff6b6b", width=2),
            )
        )

        forecast = forecast_df.dropna(subset=["Forecasted_Leakage"])
        fig.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["Forecasted_Leakage"],
                mode="lines+markers",
                name="Forecasted Leakage",
                line=dict(color="#ffd93d", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Revenue Leakage Forecast",
            xaxis_title="Date",
            yaxis_title="Total Leakage ($)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Forecast Summary")
            st.metric(
                "Total Forecasted Leakage",
                f"${metrics['total_forecasted_leakage']:,.2f}",
            )
            st.metric(
                "Avg Monthly Forecast",
                f"${metrics['avg_monthly_forecast']:,.2f}",
            )
            st.metric(
                "Historical Avg",
                f"${metrics['historical_avg']:,.2f}",
            )

        with col2:
            st.markdown("#### Model Performance")
            if "mae" in metrics:
                st.metric("MAE", f"${metrics['mae']:,.2f}")
            if "rmse" in metrics:
                st.metric("RMSE", f"${metrics['rmse']:,.2f}")
            if "mape" in metrics:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")

        st.markdown("#### Detailed Forecast")
        forecast_only = forecast_df.dropna(subset=["Forecasted_Leakage"])
        st.dataframe(
            forecast_only[["Date", "Forecasted_Leakage"]].style.format(
                {
                    "Date": lambda x: x.strftime("%Y-%m"),
                    "Forecasted_Leakage": "${:,.2f}",
                }
            )
        )

        st.markdown("#### 🤖 AI-Generated Forecast Insights")
        with st.spinner("Generating forecast insights..."):
            insights = llm_gen.generate_forecast_insights(metrics)
        st.write(insights)

    except ValueError as e:
        st.error(f"❌ Forecasting error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def _render_anomaly_detection(
    analyzer: RevenueLeakageAnalyzer,
    contamination: float,
) -> None:
    """Render anomaly detection tab."""
    st.subheader("📈 Anomaly Detection for Revenue Leakage")

    try:
        with st.spinner("Detecting anomalies..."):
            anomalies_df, summary = analyzer.detect_anomalies(contamination)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Anomalous Transactions", summary["anomaly_count"])
            st.metric("Anomaly Rate", f"{summary['anomaly_percentage']:.2f}%")

        with col2:
            st.metric(
                "Sales at Risk",
                f"${summary['total_sales_at_risk']:,.2f}",
            )
            st.metric(
                "Leakage in Anomalies",
                f"${summary['total_leakage_in_anomalies']:,.2f}",
            )

        with col3:
            st.metric(
                "Avg Anomaly Score",
                f"{summary['avg_anomaly_score']:.2f}",
            )

        if "Sales" in anomalies_df.columns and "Profit" in anomalies_df.columns:
            sample_size = min(2000, len(analyzer.df))
            plot_df = analyzer.df.sample(sample_size, random_state=42).copy()
            plot_df["Type"] = "Normal"

            anomaly_indices = set(anomalies_df.index)
            plot_df.loc[plot_df.index.isin(anomaly_indices), "Type"] = "Anomaly"

            fig = px.scatter(
                plot_df,
                x="Sales",
                y="Profit",
                color="Type",
                title="Anomaly Detection: Sales vs Profit",
                color_discrete_map={"Normal": "#4ecdc4", "Anomaly": "#ff6b6b"},
                hover_data=["Discount"] if "Discount" in plot_df.columns else None,
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(anomalies_df) > 0:
            st.markdown("#### Anomalous Transactions Details")

            display_cols = [
                "Sales",
                "Profit",
                "Profit_Margin",
                "Total_Leakage",
                "Anomaly_Score",
                "Flags",
            ]

            if "Category" in anomalies_df.columns:
                display_cols.insert(0, "Category")
            if "Discount" in anomalies_df.columns:
                display_cols.insert(-2, "Discount")

            display_cols = [c for c in display_cols if c in anomalies_df.columns]

            display_df = anomalies_df[display_cols].head(50)

            format_dict: Dict[str, str] = {}
            for col in display_cols:
                if col in ["Sales", "Profit", "Total_Leakage"]:
                    format_dict[col] = "${:,.2f}"
                elif col == "Discount":
                    format_dict[col] = "{:.2%}"
                elif col in ["Profit_Margin", "Anomaly_Score"]:
                    format_dict[col] = "{:.2f}"

            st.dataframe(display_df.style.format(format_dict))

            csv = anomalies_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Anomalous Transactions (CSV)",
                data=csv,
                file_name="revenue_leakage_anomalies.csv",
                mime="text/csv",
            )
        else:
            st.info("✅ No significant anomalies detected.")

    except ValueError as e:
        st.error(f"❌ Anomaly detection error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def _render_recommendations(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator,
) -> None:
    """Render recommendations tab."""
    st.subheader("📋 AI-Powered Recommendations to Reduce Revenue Leakage")

    metrics = analyzer.compute_leakage_metrics()

    # Worst performers
    top_categories: Optional[List[str]] = None
    if "Category" in analyzer.df.columns:
        cat_profit = analyzer.df.groupby("Category")["Profit"].sum().sort_values()
        top_categories = cat_profit.head(3).index.tolist()

    top_regions: Optional[List[str]] = None
    region_col = (
        "Region"
        if "Region" in analyzer.df.columns
        else ("State" if "State" in analyzer.df.columns else None)
    )
    if region_col:
        region_profit = analyzer.df.groupby(region_col)["Profit"].sum().sort_values()
        top_regions = region_profit.head(3).index.tolist()

    with st.spinner("Generating comprehensive recommendations..."):
        recommendations = llm_gen.generate_recommendations(
            metrics,
            top_categories,
            top_regions,
        )

    st.write(recommendations)

    st.markdown("---")
    st.markdown("#### 📥 Export Analysis Report")

    summary_data = {
        "Metric": [
            "Total Sales",
            "Total Profit",
            "Average Profit Margin",
            "Total Discount Leakage",
            "Total Margin Leakage",
            "Total Revenue Leakage",
            "Average Leakage Rate",
            "Negative Profit Transactions",
        ],
        "Value": [
            f"${metrics['total_sales']:,.2f}",
            f"${metrics['total_profit']:,.2f}",
            f"{metrics['avg_profit_margin']:.2f}%",
            f"${metrics['total_discount_leakage']:,.2f}",
            f"${metrics['total_margin_leakage']:,.2f}",
            f"${metrics['total_leakage']:,.2f}",
            f"{metrics['avg_leakage_rate']:.2f}%",
            str(metrics["negative_profit_count"]),
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    csv = summary_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Summary Report (CSV)",
        data=csv,
        file_name="revenue_leakage_summary.csv",
        mime="text/csv",
    )


# ============================================================================
# Legacy Compatibility
# ============================================================================


class RevenueLeakageDetector:
    """
    Legacy compatibility wrapper.

    Maintains backward compatibility with old API while using new architecture.
    """

    def __init__(self, data: pd.DataFrame, llm: Any) -> None:
        """Initialize with data and LLM client."""
        self.data = data
        self.llm = llm

    def detect_leakages(self) -> None:
        """Run the revenue leakage application."""
        run_revenue_leakage_app(self.data, self.llm)
