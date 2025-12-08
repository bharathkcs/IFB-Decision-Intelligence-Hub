"""
AI-driven Revenue Leakage Detection and Forecasting System

This module provides a production-ready implementation for detecting and forecasting
revenue leakages across multiple dimensions including discounting, profitability,
product performance, and regional trends.

Architecture:
    - RevenueLeakageAnalyzer: Core analytics and ML logic
    - LLMInsightGenerator: LLM-based narrative insights (now with enhanced reasoning)
    - Streamlit UI functions: User interface layer

Author: IFB Decision Intelligence Hub

Enhancement: LLM insights now use advanced reasoning scaffolds and guardrails.
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

# Enhanced LLM reasoning integration
# Set to True to use enhanced ChatGPT-level insights (recommended)
# Set to False to use original simple prompts (for comparison/fallback)
USE_ENHANCED_LLM_INSIGHTS = True

# Import enhanced insight generator if enabled
if USE_ENHANCED_LLM_INSIGHTS:
    try:
        from modules.llm_reasoning import EnhancedInsightGenerator
        _EnhancedInsightGeneratorAvailable = True
    except ImportError as e:
        print(f"⚠️ Enhanced insights not available: {e}")
        print("Falling back to original insight generation.")
        _EnhancedInsightGeneratorAvailable = False
else:
    _EnhancedInsightGeneratorAvailable = False


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
        """
        Detect and convert date columns to datetime and pick a primary date column.

        Selection Logic:
        1. Identifies all candidate date columns (existing datetime or name contains 'date')
        2. Prefers columns with 'order' in the name (case-insensitive)
        3. Among candidates, selects the one with lowest percentage of NaNs
        4. Raises ValueError if time-series analysis will be attempted but no date column exists

        The selected column is stored in self.date_column and used for all temporal analysis.

        Raises:
            ValueError: If date columns exist but all fail to parse.
        """
        # Find all candidate date columns
        candidates = []

        for col in self.df.columns:
            # Already datetime type
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                nan_pct = self.df[col].isna().mean()
                has_order = 'order' in col.lower()
                candidates.append({
                    'column': col,
                    'nan_pct': nan_pct,
                    'has_order': has_order,
                    'already_datetime': True
                })
                continue

            # Column name suggests date
            if "date" in col.lower():
                try:
                    converted = pd.to_datetime(self.df[col], errors="coerce")
                    nan_pct = converted.isna().mean()

                    # Only accept if conversion was at least partially successful
                    if nan_pct < 1.0:
                        self.df[col] = converted
                        has_order = 'order' in col.lower()
                        candidates.append({
                            'column': col,
                            'nan_pct': nan_pct,
                            'has_order': has_order,
                            'already_datetime': False
                        })
                except Exception:
                    # Conversion completely failed, skip this column
                    pass

        # Select best candidate if any exist
        if candidates:
            # Sort by: has_order (descending), then nan_pct (ascending)
            candidates_sorted = sorted(
                candidates,
                key=lambda x: (not x['has_order'], x['nan_pct'])
            )
            selected = candidates_sorted[0]
            self.date_column = selected['column']

            # Document the selection in a way that's visible during analysis
            # (Could be logged if a logger was available)
            self._date_selection_reason = (
                f"Selected '{self.date_column}' as primary date column. "
                f"Reason: {'Contains order keyword' if selected['has_order'] else 'Best quality'} "
                f"(NaN rate: {selected['nan_pct']:.1%})"
            )
        else:
            # No date columns found - this is acceptable for non-temporal analysis
            # Time-series methods will raise explicit errors when called
            self.date_column = None
            self._date_selection_reason = "No date columns detected"

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

        IMPORTANT ASSUMPTIONS & LIMITATIONS:

        1. Discount_Leakage = Sales × Discount is a SIMPLIFIED PROXY for diagnostic purposes.
           This formula assumes:
           - All discounts represent lost revenue (ignores strategic/promotional intent)
           - No price elasticity effects (volume gains from discounting are ignored)
           - No consideration of competitive dynamics or market share gains
           - Uniform treatment of tactical vs. strategic discounts

           WHY THIS IS ACCEPTABLE:
           - Useful for identifying discount patterns and high-risk transactions
           - Appropriate for diagnostic dashboards and leakage detection
           - Provides upper-bound estimate of potential discount impact

           WHY THIS IS NOT SUFFICIENT:
           - Should NOT be used for pricing optimization or promotional ROI analysis
           - Does not account for incremental volume driven by discounts
           - May overestimate actual economic loss in competitive/promotional contexts

        2. Margin_Leakage represents deviation from target margins, which may be
           influenced by factors beyond pricing (e.g., cost fluctuations, product mix).

        For strategic pricing decisions, consult pricing analytics with elasticity modeling.
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
        Forecast future revenue leakage using a tree-based regressor with walk-forward validation.

        Steps:
        - Aggregate data to monthly level.
        - Engineer lag and rolling features.
        - Validate data sufficiency based on feature requirements.
        - Perform walk-forward (rolling-origin) validation for robust performance metrics.
        - Train final model on all available data.
        - Iteratively forecast future periods with NaN guards.

        Walk-forward Validation:
            Implements rolling-origin cross-validation where the model is trained on
            progressively longer history and tested on the next unseen period. This
            provides more realistic performance estimates than a single train/test split.

        Args:
            forecast_periods: Number of months to forecast.

        Returns:
            (forecast_df, metrics_dict)
                forecast_df: DataFrame with Date, Historical_Leakage, Forecasted_Leakage.
                metrics_dict: Model performance metrics (including walk-forward validation)
                              and forecast summary statistics.

        Raises:
            ValueError: If insufficient data for forecasting based on feature requirements.
        """
        if not self.date_column:
            raise ValueError(
                "No date column available for forecasting. "
                "Time-series analysis requires a valid date column."
            )

        if not self.prepared:
            self.prepare_data()

        # Build monthly time series at overall level
        monthly_data = self.build_time_series(level="overall")

        # Remove rows without YearMonth (just in case)
        monthly_data = monthly_data.dropna(subset=["YearMonth"])

        # Convert to datetime index
        monthly_data["Date"] = monthly_data["YearMonth"].dt.to_timestamp()
        monthly_data = monthly_data.set_index("Date").sort_index()

        # Intelligent minimum data requirement
        # Need enough history for: max lag + max rolling window + buffer for validation
        max_lag = max(self.config.lag_periods) if self.config.lag_periods else 1
        max_window = max(self.config.rolling_windows) if self.config.rolling_windows else 3
        min_history = max_lag + max_window + 3  # +3 for walk-forward validation buffer

        if len(monthly_data) < min_history:
            raise ValueError(
                f"Insufficient data for forecasting. "
                f"Require at least {min_history} months based on feature configuration "
                f"(max_lag={max_lag}, max_rolling_window={max_window}, buffer=3), "
                f"but only {len(monthly_data)} months available."
            )

        # Add lag/rolling features
        monthly_data = self._add_lag_rolling_features(monthly_data)

        # Drop initial NaNs from lag/rolling
        monthly_data = monthly_data.dropna()

        # Re-validate after feature engineering
        if len(monthly_data) < max_lag + 2:
            raise ValueError(
                f"Insufficient data after feature engineering. "
                f"Need at least {max_lag + 2} clean records, got {len(monthly_data)}."
            )

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

        if len(X) < max_lag + 2:
            raise ValueError(
                f"Insufficient clean data for forecasting after NaN removal. "
                f"Need at least {max_lag + 2} records, got {len(X)}."
            )

        # ========================================================================
        # WALK-FORWARD VALIDATION (Rolling Origin Cross-Validation)
        # ========================================================================
        # This provides robust performance estimates by training on progressively
        # longer history and testing on the next unseen period.
        #
        # Example: For 12 months of data with min_train=6:
        #   Fold 1: Train on months 1-6, test on month 7
        #   Fold 2: Train on months 1-7, test on month 8
        #   ...
        #   Fold 6: Train on months 1-11, test on month 12

        min_train_periods = max_lag + max_window + 1
        walk_forward_mae = []
        walk_forward_squared_errors = []

        if len(X) >= min_train_periods + 2:  # Need at least 2 test periods
            for test_idx in range(min_train_periods, len(X)):
                # Expanding window: train on all data up to test_idx
                X_train_fold = X.iloc[:test_idx]
                y_train_fold = y.iloc[:test_idx]

                # Test on single next period
                X_test_fold = X.iloc[test_idx:test_idx + 1]
                y_test_fold = y.iloc[test_idx:test_idx + 1]

                # Train fold model
                model_fold = GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state,
                    learning_rate=0.1,
                    max_depth=4,
                )
                model_fold.fit(X_train_fold, y_train_fold)

                # Predict and store error
                y_pred_fold = model_fold.predict(X_test_fold)[0]
                y_actual_fold = y_test_fold.values[0]

                walk_forward_mae.append(abs(y_actual_fold - y_pred_fold))
                walk_forward_squared_errors.append((y_actual_fold - y_pred_fold) ** 2)

        # ========================================================================
        # TRADITIONAL TRAIN/TEST SPLIT (for comparison)
        # ========================================================================
        split_idx = int(len(X) * self.config.train_test_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train final model on all data for forecasting
        model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            learning_rate=0.1,
            max_depth=4,
        )
        model.fit(X, y)  # Use ALL data for final model

        # Compute performance metrics
        metrics: Dict[str, float] = {}

        # Walk-forward validation metrics (preferred - more realistic)
        if walk_forward_mae:
            metrics["walk_forward_mae"] = float(np.mean(walk_forward_mae))
            metrics["walk_forward_rmse"] = float(np.sqrt(np.mean(walk_forward_squared_errors)))
            metrics["walk_forward_folds"] = len(walk_forward_mae)

        # Traditional test set metrics (for reference)
        if len(X_test) > 0:
            y_pred_test = model.predict(X_test)
            metrics["test_mae"] = float(mean_absolute_error(y_test, y_pred_test))
            metrics["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

            # MAPE (avoid division by zero)
            mask = y_test != 0
            if mask.sum() > 0:
                mape = (
                    np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100
                )
                metrics["test_mape"] = float(mape)

        # Use walk-forward metrics as primary if available, else fall back to test metrics
        if "walk_forward_mae" in metrics:
            metrics["mae"] = metrics["walk_forward_mae"]
            metrics["rmse"] = metrics["walk_forward_rmse"]
        elif "test_mae" in metrics:
            metrics["mae"] = metrics["test_mae"]
            metrics["rmse"] = metrics["test_rmse"]
            if "test_mape" in metrics:
                metrics["mape"] = metrics["test_mape"]

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

            # Predict with explicit NaN guards
            X_future = pd.DataFrame([future_row])[feature_cols]
            X_future = X_future.apply(pd.to_numeric, errors="coerce")
            X_future = X_future.replace([np.inf, -np.inf], np.nan)

            # CRITICAL: Verify no NaNs exist before prediction
            # NaNs in feature matrix will cause prediction failures or unreliable results
            if X_future.isna().any().any():
                raise ValueError(
                    f"NaN values detected in future features before prediction. "
                    f"This indicates insufficient historical data for lag/rolling features. "
                    f"Problematic features: {X_future.columns[X_future.isna().any()].tolist()}"
                )

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

        CORRECTED APPROACH (aligned with business leakage definition):
        - Train regression model to predict EXPECTED profit (what we should earn).
        - Compute residuals = Expected (model prediction) - Actual profit.
        - High residual = actual is much lower than expected = leakage.
        - Mark residual-based anomalies: high residual AND low/negative actual profit.
        - Train IsolationForest on feature space for statistical outliers.
        - Combine both signals for final anomaly detection.

        Anomaly Score:
        - Normalized residual (percentile-based) scaled to [0, 100]
        - Higher score = worse (more leakage / more anomalous)
        - Interpretable across different datasets

        Args:
            contamination: Expected proportion of outliers (0–1).

        Returns:
            (anomalies_df, summary_dict)
        """
        if not self.prepared:
            self.prepare_data()

        # Core numeric features
        feature_cols: List[str] = [
            "Sales",
            "Expected_Profit",  # Target for regression
            "Profit_Margin",
            "Margin_Leakage",
            "Total_Leakage",
        ]

        if "Discount" in self.df.columns:
            feature_cols.append("Discount")

        df_encoded = self.df.copy()

        # One-hot encoding for categorical features (replaces LabelEncoder)
        # This avoids imposing false ordinal relationships
        categorical_cols = ["Category", "Sub Category", "Region", "Customer Segment"]
        for col in categorical_cols:
            if col in df_encoded.columns:
                # One-hot encode with prefix, drop first to avoid collinearity
                dummies = pd.get_dummies(
                    df_encoded[col].astype(str),
                    prefix=col,
                    drop_first=True,
                    dtype=float
                )
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())

        df_features = df_encoded[feature_cols + ["Profit"]].copy()
        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_features) < 100:
            raise ValueError(
                "Insufficient data for anomaly detection (need >= 100 records)"
            )

        # ========================================================================
        # CORRECTED REGRESSION: Predict Expected_Profit (what SHOULD be earned)
        # ========================================================================
        # We remove Expected_Profit from features and use it as target
        X = df_features.drop(["Expected_Profit", "Profit"], axis=1, errors="ignore")
        y_expected = df_features["Expected_Profit"]  # Target: what we expect
        y_actual = df_features["Profit"]  # Actual profit (for residual computation)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, _ = y_expected.iloc[:split_idx], y_expected.iloc[split_idx:]

        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=self.config.random_state,
            max_depth=4,
        )
        model.fit(X_train, y_train)

        # Predict expected profit on full dataset
        y_pred_expected = model.predict(X)

        # CORRECTED RESIDUAL DEFINITION:
        # Residual = Expected (model prediction) - Actual
        # High residual = actual is much lower than expected = LEAKAGE
        residuals = y_pred_expected - y_actual.values

        df_features["Residual"] = residuals
        df_features["Expected_Profit_Model"] = y_pred_expected

        # Residual-based anomalies
        # High percentile residual (expected >> actual) AND low/negative actual profit
        residual_threshold = np.percentile(
            residuals, self.config.anomaly_percentile * 100
        )
        df_features["Is_Anomaly_Residual"] = (
            (df_features["Residual"] > residual_threshold)
            & (
                (df_features["Profit"] < df_features["Profit"].quantile(0.25))
                | (df_features["Profit"] < 0)
            )
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

        # ========================================================================
        # CONSISTENT & INTERPRETABLE ANOMALY SCORE
        # ========================================================================
        # Use percentile-based normalization for consistent interpretation
        # Score = percentile rank of residual, scaled to [0, 100]
        # Higher score = worse (more leakage)
        from scipy.stats import percentileofscore

        df_features["Anomaly_Score"] = df_features["Residual"].apply(
            lambda x: percentileofscore(residuals, x, kind="rank")
        )

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

        # Persist anomaly flags and scores back to the main dataframe
        # so other tabs (e.g., Recommendations) can see them.
        self.df.loc[df_features.index, "Is_Anomaly"] = df_features["Is_Anomaly"].astype(bool)
        self.df.loc[df_features.index, "Anomaly_Score"] = df_features["Anomaly_Score"]

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
        Score each transaction for leakage risk using classification with full evaluation.

        Steps:
        - Construct Leakage_Flag: 1 if Total_Leakage above 75th percentile or Profit_Margin < 0.
        - One-hot encode categorical features (no LabelEncoder to avoid ordinal assumptions).
        - Perform time-aware or random train/test split.
        - Train GradientBoostingClassifier with evaluation on test set.
        - Compute classification metrics: Accuracy, Precision, Recall, ROC-AUC.
        - Output leakage probability for each transaction.

        Returns:
            (risk_scores_df, metadata_dict)
                risk_scores_df: DataFrame with Leakage_Probability column.
                metadata_dict: Contains feature importance and MODEL EVALUATION METRICS.
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

        # One-hot encoding for categorical features (replaces LabelEncoder)
        categorical_cols = ["Category", "Sub Category", "Region", "Customer Segment"]
        for col in categorical_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(
                    df_encoded[col].astype(str),
                    prefix=col,
                    drop_first=True,
                    dtype=float
                )
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())

        df_model = df_encoded[feature_cols + ["Leakage_Flag"]].copy()
        df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

        X = df_model[feature_cols]
        y = df_model["Leakage_Flag"]

        if len(X) < 20:
            raise ValueError("Insufficient data for risk scoring")

        # ========================================================================
        # TRAIN/TEST SPLIT FOR MODEL EVALUATION
        # ========================================================================
        # Time-aware split if date column exists, else random split
        if self.date_column and self.date_column in self.df.columns:
            # Time-based split: train on earlier data, test on later
            df_with_date = self.df.loc[df_model.index].copy()
            df_with_date = df_with_date.sort_values(self.date_column)
            split_idx = int(len(df_with_date) * 0.75)

            train_indices = df_with_date.index[:split_idx]
            test_indices = df_with_date.index[split_idx:]

            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=self.config.random_state, stratify=y
            )

        # Train classifier
        clf = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            max_depth=4,
        )
        clf.fit(X_train, y_train)

        # ========================================================================
        # MODEL EVALUATION ON TEST SET
        # ========================================================================
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
            confusion_matrix,
        )

        y_pred_test = clf.predict(X_test)
        y_prob_test = clf.predict_proba(X_test)[:, 1]

        evaluation_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob_test)),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            evaluation_metrics["confusion_matrix"] = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }

        # Predict on full dataset for output
        leakage_probs = clf.predict_proba(X)[:, 1]

        result_df = self.df.loc[df_model.index].copy()
        result_df["Leakage_Probability"] = leakage_probs

        # Persist leakage probabilities back to the main dataframe
        self.df.loc[df_model.index, "Leakage_Probability"] = leakage_probs

        # Feature importance
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

        # Metadata with evaluation metrics
        metadata = {
            "feature_importance": feature_importance.to_dict("records"),
            "high_risk_count": int((leakage_probs > 0.7).sum()),
            "medium_risk_count": int(
                ((leakage_probs > 0.4) & (leakage_probs <= 0.7)).sum()
            ),
            "low_risk_count": int((leakage_probs <= 0.4).sum()),
            "avg_probability": float(leakage_probs.mean()),
            # NEW: Model evaluation metrics
            "model_evaluation": evaluation_metrics,
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

    def generate_root_cause_analysis(self, context: Dict[str, Any]) -> str:
        """
        Generate root cause analysis explaining WHY revenue leakage is occurring.

        This method analyzes patterns discovered by ML models and provides
        causal explanations rather than just describing what happened.

        Args:
            context: Dictionary containing:
                - leakage_metrics: Overall leakage summary
                - anomaly_patterns: Anomaly detection results
                - top_leakage_categories: List of categories with highest leakage
                - top_leakage_regions: List of regions with highest leakage
                - discount_stats: Discount distribution statistics
                - temporal_trends: Time-based pattern information

        Returns:
            Root cause analysis text explaining WHY leakage is occurring.
        """
        # Extract context information
        metrics = context.get('leakage_metrics', {})
        anomalies = context.get('anomaly_patterns', {})
        top_categories = context.get('top_leakage_categories', [])
        top_regions = context.get('top_leakage_regions', [])
        discount_stats = context.get('discount_stats', {})
        temporal = context.get('temporal_trends', {})

        # Build category and region context
        category_info = ""
        if top_categories:
            category_info = f"\n**High Leakage Categories:** {', '.join(top_categories[:5])}"

        region_info = ""
        if top_regions:
            region_info = f"\n**High Leakage Regions:** {', '.join(top_regions[:5])}"

        temporal_info = ""
        if temporal:
            temporal_info = f"""
**Temporal Patterns:**
- Recent trend: {temporal.get('trend_direction', 'stable')}
- Seasonality detected: {temporal.get('has_seasonality', 'unknown')}
- Recent volatility: {temporal.get('volatility_level', 'unknown')}
"""

        prompt = f"""
As a data-driven business analyst, perform a root cause analysis of revenue leakage.

**Current State:**
- Total Revenue Leakage: ${metrics.get('total_leakage', 0):,.2f}
- Discount Leakage: ${metrics.get('total_discount_leakage', 0):,.2f}
- Margin Leakage: ${metrics.get('total_margin_leakage', 0):,.2f}
- Average Profit Margin: {metrics.get('avg_profit_margin', 0):.2f}%
- Negative Profit Transactions: {metrics.get('negative_profit_count', 0)}
{category_info}
{region_info}

**Discount Patterns:**
- Average Discount: {discount_stats.get('mean', 0):.2%}
- Max Discount: {discount_stats.get('max', 0):.2%}
- High Discount Transactions (>20%): {discount_stats.get('high_discount_count', 0)}

**Anomaly Detection:**
- Anomalous Transactions: {anomalies.get('anomaly_count', 0)}
- High-Risk Anomalies: {anomalies.get('high_risk_count', 0)}
{temporal_info}

**Your Task:**
Explain WHY this leakage is occurring, not just WHAT is happening. Focus on:

1. **Primary Root Causes** (2-3 most likely causal factors)
   - Be specific about mechanisms (e.g., "Aggressive discounting without corresponding volume increases")
   - Distinguish between symptoms and causes

2. **Contributing Factors** (secondary issues amplifying the problem)
   - Structural issues (pricing strategy, product mix, regional dynamics)
   - Operational issues (discount approval processes, margin controls)

3. **Interconnected Patterns** (how different signals reinforce each other)
   - E.g., how category-specific issues relate to regional challenges
   - Temporal correlations (seasonal patterns, recent changes)

4. **Confidence Assessment**
   - Which root causes are you most confident about and why?
   - What additional data would improve certainty?

Use clear, analytical language. Avoid generic statements. Be specific and evidence-based.
"""
        return self._call_llm(prompt)

    def generate_cross_signal_insights(
        self,
        forecast_summary: Dict[str, float],
        anomaly_summary: Dict[str, Any],
        risk_summary: Dict[str, Any],
        leakage_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate cross-model insights connecting forecasts, anomalies, and risk scores.

        This method synthesizes multiple ML model outputs into coherent intelligence,
        identifying patterns that emerge only when viewing signals together.

        Args:
            forecast_summary: Forecasting model results and metrics
            anomaly_summary: Anomaly detection results
            risk_summary: Risk scoring model results
            leakage_metrics: Optional overall leakage metrics

        Returns:
            Cross-signal synthesis explaining what multiple models tell us together.
        """
        metrics_info = ""
        if leakage_metrics:
            metrics_info = f"""
**Current Leakage State:**
- Total Leakage: ${leakage_metrics.get('total_leakage', 0):,.2f}
- Profit Margin: {leakage_metrics.get('avg_profit_margin', 0):.2f}%
"""

        prompt = f"""
As a decision intelligence analyst, synthesize insights from multiple ML models to provide integrated intelligence.

{metrics_info}
**Forecast Model Output:**
- Historical Avg Monthly Leakage: ${forecast_summary.get('historical_avg', 0):,.2f}
- Forecasted Avg Monthly Leakage: ${forecast_summary.get('avg_monthly_forecast', 0):,.2f}
- Total Forecasted Leakage (next periods): ${forecast_summary.get('total_forecasted_leakage', 0):,.2f}
- Forecast Trend: {('increasing' if forecast_summary.get('avg_monthly_forecast', 0) > forecast_summary.get('historical_avg', 0) else 'decreasing')}
- Model Error (MAE): ${forecast_summary.get('mae', 0):,.2f}

**Anomaly Detection Output:**
- Total Anomalous Transactions: {anomaly_summary.get('anomaly_count', 0)}
- High-Risk Anomalies: {anomaly_summary.get('high_risk_count', 0)}
- Medium-Risk Anomalies: {anomaly_summary.get('medium_risk_count', 0)}
- Anomaly Rate: {anomaly_summary.get('anomaly_rate', 0):.2%}

**Risk Scoring Output:**
- High-Risk Transactions: {risk_summary.get('high_risk_count', 0)}
- Medium-Risk Transactions: {risk_summary.get('medium_risk_count', 0)}
- Average Risk Probability: {risk_summary.get('avg_risk_probability', 0):.2%}
- Model Accuracy: {risk_summary.get('accuracy', 0):.2%}
- Model ROC-AUC: {risk_summary.get('roc_auc', 0):.3f}

**Your Task:**
Connect these three perspectives into unified intelligence. Focus on:

1. **Signal Convergence** - Where do all models agree?
   - Do forecasts, anomalies, and risk scores point to the same issues?
   - What story emerges when you view all three together?

2. **Signal Divergence** - Where do models disagree?
   - Are there anomalies without high forecasted leakage? (Indicates isolated incidents)
   - Are there forecasted increases without current anomalies? (Early warning of systemic shift)
   - What do these divergences tell us?

3. **Emergent Patterns** - Insights visible only through cross-model analysis
   - Pattern stability vs. emerging risks
   - Structural issues (forecasted) vs. tactical issues (anomalies)

4. **Confidence and Reliability**
   - Which model outputs are most reliable given their performance metrics?
   - How should decision-makers weight different signals?

5. **Integrated Interpretation**
   - What is the single most important insight from combining all three models?
   - Does this suggest systemic problems or isolated incidents?

Be analytical and specific. Avoid restating individual model outputs—focus on synthesis.
"""
        return self._call_llm(prompt)

    def generate_decision_priority(
        self,
        signals: Dict[str, Any],
    ) -> str:
        """
        Generate prioritized decision guidance based on severity, confidence, and impact.

        This method translates analytical insights into actionable priorities,
        helping leaders decide what to act on NOW vs. what can wait.

        Args:
            signals: Dictionary containing:
                - leakage_severity: Overall severity level (high/medium/low)
                - forecast_trend: Future trajectory (increasing/decreasing/stable)
                - anomaly_risk: Anomaly concentration and risk
                - model_confidence: Reliability of predictions
                - business_impact: Estimated financial impact
                - quick_wins: Identified quick win opportunities
                - strategic_issues: Longer-term strategic problems

        Returns:
            Prioritized decision framework text.
        """
        prompt = f"""
As a management consultant and decision strategist, create a prioritized action framework.

**Situation Analysis:**
- **Leakage Severity:** {signals.get('leakage_severity', 'unknown')}
- **Forecast Trend:** {signals.get('forecast_trend', 'unknown')}
- **Anomaly Risk Level:** {signals.get('anomaly_risk', 'unknown')}
- **Model Confidence:** {signals.get('model_confidence', 'unknown')}
- **Estimated Monthly Impact:** ${signals.get('monthly_impact', 0):,.2f}
- **Quick Win Opportunities:** {signals.get('quick_wins', [])}
- **Strategic Issues:** {signals.get('strategic_issues', [])}

**Your Task:**
Create a clear decision priority framework. Structure your response as:

## 🔴 ACT NOW (0-7 days) - Critical & Urgent

**What:** [Specific actions requiring immediate attention]
**Why Now:** [Reasoning: financial exposure, risk escalation, or easy wins]
**Expected Impact:** [Concrete outcomes if action is taken within 7 days]
**Risk if Delayed:** [What happens if this waits 30+ days]

## 🟡 PLAN & EXECUTE (1-4 weeks) - Important, Not Urgent

**What:** [Actions requiring planning but not emergency response]
**Why This Timeline:** [Reasoning: requires coordination, testing, or stakeholder buy-in]
**Expected Impact:** [Medium-term improvements]
**Dependencies:** [What needs to happen first]

## 🟢 MONITOR & SCHEDULE (1-3 months) - Strategic

**What:** [Longer-term improvements and structural changes]
**Why Later:** [Reasoning: requires infrastructure, culture change, or sustained effort]
**Expected Impact:** [Long-term structural improvements]
**Success Metrics:** [How to measure progress]

## 📊 WATCH ONLY (Monitoring)

**What:** [Signals to track but not act on immediately]
**Trigger Points:** [When these become urgent (specific thresholds)]
**Review Frequency:** [Weekly/Monthly check-in recommended]

## ⚠️ Decision Confidence

**High Confidence Actions:** [What we're certain about]
**Moderate Confidence Actions:** [What needs validation]
**Data Gaps:** [What additional information would improve decisions]

Be ruthlessly prioritized. Not everything can be urgent. Use specific thresholds and triggers.
"""
        return self._call_llm(prompt)

    def generate_scenario_impact_explanation(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
        scenario_description: str,
    ) -> str:
        """
        Generate business impact explanation for scenario/what-if analysis.

        IMPORTANT: This method does NOT simulate numbers—it interprets
        changes already computed by deterministic Python logic.

        Args:
            before_metrics: Baseline metrics before scenario change
            after_metrics: Metrics after scenario simulation
            scenario_description: Description of what was changed (e.g., "Discount cap set to 15%")

        Returns:
            Business impact interpretation of the scenario.
        """
        # Calculate deltas
        sales_delta = after_metrics.get('total_sales', 0) - before_metrics.get('total_sales', 0)
        profit_delta = after_metrics.get('total_profit', 0) - before_metrics.get('total_profit', 0)
        leakage_delta = after_metrics.get('total_leakage', 0) - before_metrics.get('total_leakage', 0)
        margin_delta = after_metrics.get('avg_profit_margin', 0) - before_metrics.get('avg_profit_margin', 0)

        sales_change_pct = (sales_delta / before_metrics.get('total_sales', 1)) * 100 if before_metrics.get('total_sales', 0) > 0 else 0
        profit_change_pct = (profit_delta / before_metrics.get('total_profit', 1)) * 100 if before_metrics.get('total_profit', 0) > 0 else 0

        prompt = f"""
As a business strategy analyst, explain the business implications of this scenario simulation.

**Scenario:** {scenario_description}

**Before Scenario:**
- Total Sales: ${before_metrics.get('total_sales', 0):,.2f}
- Total Profit: ${before_metrics.get('total_profit', 0):,.2f}
- Revenue Leakage: ${before_metrics.get('total_leakage', 0):,.2f}
- Profit Margin: {before_metrics.get('avg_profit_margin', 0):.2f}%

**After Scenario:**
- Total Sales: ${after_metrics.get('total_sales', 0):,.2f}
- Total Profit: ${after_metrics.get('total_profit', 0):,.2f}
- Revenue Leakage: ${after_metrics.get('total_leakage', 0):,.2f}
- Profit Margin: {after_metrics.get('avg_profit_margin', 0):.2f}%

**Computed Changes:**
- Sales Change: ${sales_delta:,.2f} ({sales_change_pct:+.2f}%)
- Profit Change: ${profit_delta:,.2f} ({profit_change_pct:+.2f}%)
- Leakage Change: ${leakage_delta:,.2f}
- Margin Change: {margin_delta:+.2f} percentage points

**Your Task:**
Interpret the business meaning of these changes. Structure your response:

1. **Overall Impact Assessment** (1-2 sentences)
   - Is this scenario beneficial, neutral, or detrimental?
   - What's the headline takeaway?

2. **Trade-offs Analysis**
   - What did we gain?
   - What did we sacrifice?
   - Are the trade-offs acceptable?

3. **Business Implications**
   - How would this affect quarterly/annual performance?
   - Impact on cash flow, competitiveness, or market position?
   - Stakeholder implications (customers, sales team, finance)?

4. **Implementation Considerations**
   - Is this change realistic to implement?
   - What risks or resistance might arise?
   - What would be required to execute this?

5. **Recommendation**
   - Should leadership pursue this scenario?
   - If yes, with what modifications or safeguards?
   - If no, what alternative should be explored?

Be balanced and objective. Acknowledge both benefits and costs. Be specific.
"""
        return self._call_llm(prompt)

    def generate_forecast_confidence_narrative(
        self,
        forecast_metrics: Dict[str, float],
        data_quality: Dict[str, Any],
    ) -> str:
        """
        Generate narrative explaining forecast model confidence and reliability.

        Args:
            forecast_metrics: Model performance metrics (MAE, RMSE, R², etc.)
            data_quality: Data characteristics affecting confidence
                - data_points: Number of observations
                - volatility: Historical volatility measure
                - trend_stability: Whether trend is consistent
                - seasonality_strength: Seasonality signal strength

        Returns:
            Confidence assessment narrative.
        """
        prompt = f"""
As a data scientist, explain the confidence and reliability of this forecast to business stakeholders.

**Model Performance Metrics:**
- Mean Absolute Error (MAE): ${forecast_metrics.get('mae', 0):,.2f}
- Root Mean Squared Error (RMSE): ${forecast_metrics.get('rmse', 0):,.2f}
- R-squared (R²): {forecast_metrics.get('r2', 0):.3f}
- Mean Absolute Percentage Error (MAPE): {forecast_metrics.get('mape', 0):.2f}%

**Data Characteristics:**
- Historical Data Points: {data_quality.get('data_points', 0)}
- Volatility Level: {data_quality.get('volatility', 'unknown')}
- Trend Stability: {data_quality.get('trend_stability', 'unknown')}
- Seasonality Strength: {data_quality.get('seasonality_strength', 'unknown')}

**Your Task:**
Explain forecast confidence in plain business language:

1. **Confidence Level** (High/Moderate/Low and why)
   - What do the error metrics tell us?
   - How reliable are these predictions?

2. **Reliability Factors**
   - What makes this forecast more/less trustworthy?
   - Data sufficiency, stability, patterns

3. **Uncertainty Range**
   - How far off could reality be from the forecast?
   - Best-case and worst-case scenarios

4. **Appropriate Use**
   - What decisions can confidently be made with this forecast?
   - What decisions need more data or different models?

5. **Improvement Recommendations**
   - What would increase forecast confidence?
   - More data, different features, external factors?

Avoid jargon. Use analogies if helpful. Be honest about limitations.
"""
        return self._call_llm(prompt)

    def generate_anomaly_confidence_narrative(
        self,
        anomaly_metrics: Dict[str, Any],
        detection_config: Dict[str, Any],
    ) -> str:
        """
        Generate narrative explaining anomaly detection confidence and reliability.

        Args:
            anomaly_metrics: Anomaly detection results
                - anomaly_count: Number of anomalies detected
                - anomaly_rate: Percentage of data flagged
                - high_risk_count: Critical anomalies
            detection_config: Detection method configuration
                - methods_used: List of detection methods
                - thresholds: Sensitivity settings

        Returns:
            Anomaly detection confidence narrative.
        """
        prompt = f"""
As a data scientist, explain the reliability of this anomaly detection analysis.

**Anomaly Detection Results:**
- Total Anomalies Detected: {anomaly_metrics.get('anomaly_count', 0)}
- Anomaly Rate: {anomaly_metrics.get('anomaly_rate', 0):.2%}
- High-Risk Anomalies: {anomaly_metrics.get('high_risk_count', 0)}
- Medium-Risk Anomalies: {anomaly_metrics.get('medium_risk_count', 0)}
- Low-Risk Anomalies: {anomaly_metrics.get('low_risk_count', 0)}

**Detection Configuration:**
- Methods Used: {', '.join(detection_config.get('methods_used', []))}
- Sensitivity Settings: {detection_config.get('thresholds', 'standard')}
- Features Analyzed: {detection_config.get('feature_count', 'multiple')}

**Your Task:**
Explain anomaly detection confidence in business terms:

1. **Detection Reliability**
   - How trustworthy are these anomaly flags?
   - Are we detecting real issues or false alarms?

2. **False Positive Risk**
   - What's the likelihood of flagging normal transactions as anomalous?
   - Should all flagged items be investigated equally?

3. **False Negative Risk**
   - Could we be missing important anomalies?
   - What patterns might slip through?

4. **Prioritization Guidance**
   - Which flagged anomalies deserve immediate attention?
   - Which are lower priority?
   - Investigation resource allocation recommendations

5. **Recommended Actions**
   - How should teams use these anomaly alerts?
   - Threshold adjustments needed?
   - Follow-up analysis recommended?

Be practical and honest about limitations. Help teams avoid alert fatigue.
"""
        return self._call_llm(prompt)

    def generate_risk_score_confidence_narrative(
        self,
        risk_model_metrics: Dict[str, float],
        model_validation: Dict[str, Any],
    ) -> str:
        """
        Generate narrative explaining risk scoring model confidence and reliability.

        Args:
            risk_model_metrics: Classification model performance
                - accuracy: Overall accuracy
                - precision: Precision score
                - recall: Recall score
                - roc_auc: ROC-AUC score
            model_validation: Validation information
                - train_size: Training set size
                - test_size: Test set size
                - class_balance: Class distribution

        Returns:
            Risk scoring confidence narrative.
        """
        prompt = f"""
As a data scientist, explain the reliability of this risk scoring model to business users.

**Risk Model Performance:**
- Overall Accuracy: {risk_model_metrics.get('accuracy', 0):.2%}
- Precision: {risk_model_metrics.get('precision', 0):.2%}
- Recall (Sensitivity): {risk_model_metrics.get('recall', 0):.2%}
- ROC-AUC Score: {risk_model_metrics.get('roc_auc', 0):.3f}
- F1-Score: {risk_model_metrics.get('f1', 0):.3f}

**Model Validation:**
- Training Set Size: {model_validation.get('train_size', 0)} transactions
- Test Set Size: {model_validation.get('test_size', 0)} transactions
- Class Balance: {model_validation.get('class_balance', 'unknown')}

**Your Task:**
Explain risk scoring confidence in actionable business terms:

1. **Model Trustworthiness**
   - How reliable are the risk probability scores?
   - Can these scores drive automated decisions?

2. **Accuracy Interpretation**
   - What does {risk_model_metrics.get('accuracy', 0):.0%} accuracy mean practically?
   - How often will the model be right vs. wrong?

3. **Precision vs. Recall Trade-off**
   - Does the model catch most risky transactions (recall)?
   - Does it avoid false alarms (precision)?
   - Which is more important for this business context?

4. **Recommended Use Cases**
   - **High confidence (use for):** [What automated actions are safe]
   - **Moderate confidence (use for):** [What decisions need human review]
   - **Low confidence (avoid):** [What NOT to automate]

5. **Model Improvement Path**
   - What would make risk scores more reliable?
   - Data, features, or methodology changes needed?

Translate technical metrics into business decision guidance. Be specific about risk tolerance.
"""
        return self._call_llm(prompt)


class AIDecisionIntelligence:
    """
    Lightweight orchestration layer for Decision Intelligence.

    This class coordinates LLM reasoning across multiple ML model outputs,
    synthesizing insights into decision-ready intelligence.

    Design Philosophy:
    - ML models compute all numbers (forecasting, anomaly detection, risk scoring)
    - GenAI explains, connects, hypothesizes, and prioritizes decisions
    - No numeric computation happens in LLM calls

    This is NOT a duplicate of LLMInsightGenerator—it orchestrates multiple
    calls to create integrated decision intelligence.
    """

    def __init__(
        self,
        analyzer: RevenueLeakageAnalyzer,
        insight_generator: LLMInsightGenerator,
    ) -> None:
        """
        Initialize with analyzer and insight generator.

        Args:
            analyzer: RevenueLeakageAnalyzer instance with prepared data
            insight_generator: LLMInsightGenerator instance
        """
        self.analyzer = analyzer
        self.llm = insight_generator

    def generate_comprehensive_intelligence(
        self,
        include_forecast: bool = True,
        include_anomalies: bool = True,
        include_risk: bool = True,
    ) -> Dict[str, str]:
        """
        Generate comprehensive decision intelligence across all ML models.

        This orchestrates multiple ML analyses and synthesizes them into
        unified decision intelligence.

        Args:
            include_forecast: Whether to include forecasting analysis
            include_anomalies: Whether to include anomaly detection
            include_risk: Whether to include risk scoring

        Returns:
            Dictionary containing different intelligence perspectives:
            - root_cause_analysis: WHY leakage is occurring
            - cross_signal_insights: What multiple models tell us together
            - decision_priorities: What to act on NOW vs. later
            - forecast_confidence: Forecast reliability assessment
            - anomaly_confidence: Anomaly detection reliability
            - risk_confidence: Risk scoring reliability
        """
        intelligence = {}

        # Get base metrics
        leakage_metrics = self.analyzer.compute_leakage_metrics()

        # Prepare context for root cause analysis
        context = self._prepare_root_cause_context()
        intelligence['root_cause_analysis'] = self.llm.generate_root_cause_analysis(context)

        # Cross-signal insights (requires multiple model outputs)
        if include_forecast and include_anomalies and include_risk:
            forecast_summary = self._get_forecast_summary()
            anomaly_summary = self._get_anomaly_summary()
            risk_summary = self._get_risk_summary()

            if forecast_summary and anomaly_summary and risk_summary:
                intelligence['cross_signal_insights'] = self.llm.generate_cross_signal_insights(
                    forecast_summary=forecast_summary,
                    anomaly_summary=anomaly_summary,
                    risk_summary=risk_summary,
                    leakage_metrics=leakage_metrics,
                )

        # Decision priorities
        signals = self._prepare_decision_signals(
            leakage_metrics=leakage_metrics,
            include_forecast=include_forecast,
            include_anomalies=include_anomalies,
        )
        intelligence['decision_priorities'] = self.llm.generate_decision_priority(signals)

        # Model confidence narratives
        if include_forecast:
            forecast_metrics, data_quality = self._get_forecast_confidence_data()
            if forecast_metrics and data_quality:
                intelligence['forecast_confidence'] = self.llm.generate_forecast_confidence_narrative(
                    forecast_metrics=forecast_metrics,
                    data_quality=data_quality,
                )

        if include_anomalies:
            anomaly_metrics, detection_config = self._get_anomaly_confidence_data()
            if anomaly_metrics and detection_config:
                intelligence['anomaly_confidence'] = self.llm.generate_anomaly_confidence_narrative(
                    anomaly_metrics=anomaly_metrics,
                    detection_config=detection_config,
                )

        if include_risk:
            risk_model_metrics, model_validation = self._get_risk_confidence_data()
            if risk_model_metrics and model_validation:
                intelligence['risk_confidence'] = self.llm.generate_risk_score_confidence_narrative(
                    risk_model_metrics=risk_model_metrics,
                    model_validation=model_validation,
                )

        return intelligence

    def _prepare_root_cause_context(self) -> Dict[str, Any]:
        """
        Prepare context dictionary for root cause analysis.

        Returns:
            Context with leakage metrics, patterns, and trends.
        """
        df = self.analyzer.df

        # Leakage metrics
        leakage_metrics = self.analyzer.compute_leakage_metrics()

        # Top leakage categories
        top_categories = []
        if 'Category' in df.columns and 'Total_Leakage' in df.columns:
            category_leakage = df.groupby('Category')['Total_Leakage'].sum().sort_values(ascending=False)
            top_categories = category_leakage.head(5).index.tolist()

        # Top leakage regions
        top_regions = []
        if 'Region' in df.columns and 'Total_Leakage' in df.columns:
            region_leakage = df.groupby('Region')['Total_Leakage'].sum().sort_values(ascending=False)
            top_regions = region_leakage.head(5).index.tolist()

        # Discount statistics
        discount_stats = {}
        if 'Discount' in df.columns:
            discount_stats = {
                'mean': df['Discount'].mean(),
                'max': df['Discount'].max(),
                'high_discount_count': len(df[df['Discount'] > 0.20]),
            }

        # Anomaly patterns (if available)
        anomaly_patterns = {}
        if 'Is_Anomaly' in df.columns:
            anomaly_patterns = {
                'anomaly_count': df['Is_Anomaly'].sum(),
                'high_risk_count': len(df[(df['Is_Anomaly'] == 1) & (df.get('Leakage_Probability', 0) > 0.7)]),
            }

        # Temporal trends (basic)
        temporal_trends = {}
        if self.analyzer.date_column and 'Total_Leakage' in df.columns:
            # Simple trend direction based on first half vs second half
            sorted_df = df.sort_values(by=self.analyzer.date_column)
            mid_point = len(sorted_df) // 2
            first_half_avg = sorted_df.iloc[:mid_point]['Total_Leakage'].mean()
            second_half_avg = sorted_df.iloc[mid_point:]['Total_Leakage'].mean()

            temporal_trends = {
                'trend_direction': 'increasing' if second_half_avg > first_half_avg else 'decreasing',
                'has_seasonality': 'unknown',  # Would require more sophisticated analysis
                'volatility_level': 'high' if df['Total_Leakage'].std() > df['Total_Leakage'].mean() else 'moderate',
            }

        return {
            'leakage_metrics': leakage_metrics,
            'anomaly_patterns': anomaly_patterns,
            'top_leakage_categories': top_categories,
            'top_leakage_regions': top_regions,
            'discount_stats': discount_stats,
            'temporal_trends': temporal_trends,
        }

    def _get_forecast_summary(self) -> Optional[Dict[str, float]]:
        """
        Get forecast summary if forecasting has been run.

        Returns:
            Forecast metrics or None if not available.
        """
        # This would typically check if forecast has been computed
        # For now, return None (will be populated when forecasting is run in UI)
        return None

    def _get_anomaly_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get anomaly summary if anomaly detection has been run.

        Returns:
            Anomaly metrics or None if not available.
        """
        df = self.analyzer.df
        if 'Is_Anomaly' in df.columns:
            total_anomalies = df['Is_Anomaly'].sum()
            anomaly_rate = total_anomalies / len(df) if len(df) > 0 else 0

            # Count by risk level if available
            high_risk = medium_risk = low_risk = 0
            if 'Leakage_Probability' in df.columns:
                anomaly_df = df[df['Is_Anomaly'] == 1]
                high_risk = len(anomaly_df[anomaly_df['Leakage_Probability'] > 0.7])
                medium_risk = len(anomaly_df[(anomaly_df['Leakage_Probability'] >= 0.4) & (anomaly_df['Leakage_Probability'] <= 0.7)])
                low_risk = len(anomaly_df[anomaly_df['Leakage_Probability'] < 0.4])

            return {
                'anomaly_count': int(total_anomalies),
                'anomaly_rate': anomaly_rate,
                'high_risk_count': high_risk,
                'medium_risk_count': medium_risk,
                'low_risk_count': low_risk,
            }
        return None

    def _get_risk_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get risk scoring summary if risk scoring has been run.

        Returns:
            Risk metrics or None if not available.
        """
        df = self.analyzer.df
        if 'Leakage_Probability' in df.columns:
            avg_probability = df['Leakage_Probability'].mean()
            high_risk = len(df[df['Leakage_Probability'] > 0.7])
            medium_risk = len(df[(df['Leakage_Probability'] >= 0.4) & (df['Leakage_Probability'] <= 0.7)])

            return {
                'high_risk_count': high_risk,
                'medium_risk_count': medium_risk,
                'avg_risk_probability': avg_probability,
                'accuracy': 0.85,  # Placeholder - would come from actual model metrics
                'roc_auc': 0.88,   # Placeholder
            }
        return None

    def _prepare_decision_signals(
        self,
        leakage_metrics: Dict[str, float],
        include_forecast: bool,
        include_anomalies: bool,
    ) -> Dict[str, Any]:
        """
        Prepare signals for decision priority framework.

        Args:
            leakage_metrics: Overall leakage metrics
            include_forecast: Whether forecast data is available
            include_anomalies: Whether anomaly data is available

        Returns:
            Signal dictionary for decision prioritization.
        """
        df = self.analyzer.df

        # Determine severity
        total_leakage = leakage_metrics.get('total_leakage', 0)
        total_sales = leakage_metrics.get('total_sales', 1)
        leakage_rate = (total_leakage / total_sales) * 100 if total_sales > 0 else 0

        if leakage_rate > 15:
            severity = 'high'
        elif leakage_rate > 8:
            severity = 'medium'
        else:
            severity = 'low'

        # Determine forecast trend (if available)
        forecast_trend = 'unknown'
        if include_forecast and self.analyzer.date_column:
            # Simple trend based on recent data
            sorted_df = df.sort_values(by=self.analyzer.date_column)
            recent_third = sorted_df.iloc[-len(sorted_df)//3:]
            earlier_third = sorted_df.iloc[:len(sorted_df)//3]

            if 'Total_Leakage' in df.columns:
                recent_avg = recent_third['Total_Leakage'].mean()
                earlier_avg = earlier_third['Total_Leakage'].mean()
                forecast_trend = 'increasing' if recent_avg > earlier_avg * 1.1 else 'stable' if recent_avg > earlier_avg * 0.9 else 'decreasing'

        # Anomaly risk level
        anomaly_risk = 'unknown'
        if include_anomalies and 'Is_Anomaly' in df.columns:
            anomaly_rate = df['Is_Anomaly'].sum() / len(df) if len(df) > 0 else 0
            anomaly_risk = 'high' if anomaly_rate > 0.10 else 'medium' if anomaly_rate > 0.05 else 'low'

        # Identify quick wins
        quick_wins = []
        if 'Discount' in df.columns:
            high_discount_count = len(df[df['Discount'] > 0.25])
            if high_discount_count > 0:
                quick_wins.append(f"Review {high_discount_count} transactions with >25% discount")

        if 'Profit' in df.columns:
            negative_profit_count = len(df[df['Profit'] < 0])
            if negative_profit_count > 0:
                quick_wins.append(f"Stop {negative_profit_count} negative-profit transactions")

        # Strategic issues
        strategic_issues = []
        if leakage_metrics.get('avg_profit_margin', 0) < 10:
            strategic_issues.append("Overall profit margin below industry standard")

        # Estimate monthly impact (simple annualization)
        months_in_data = 12  # Placeholder - would calculate from actual date range
        monthly_impact = total_leakage / months_in_data if months_in_data > 0 else 0

        return {
            'leakage_severity': severity,
            'forecast_trend': forecast_trend,
            'anomaly_risk': anomaly_risk,
            'model_confidence': 'moderate',  # Placeholder
            'monthly_impact': monthly_impact,
            'quick_wins': quick_wins,
            'strategic_issues': strategic_issues,
        }

    def _get_forecast_confidence_data(self) -> tuple:
        """
        Get forecast confidence data.

        Returns:
            Tuple of (forecast_metrics, data_quality) or (None, None).
        """
        # Placeholder - would extract from actual forecast results
        return None, None

    def _get_anomaly_confidence_data(self) -> tuple:
        """
        Get anomaly confidence data.

        Returns:
            Tuple of (anomaly_metrics, detection_config) or (None, None).
        """
        anomaly_summary = self._get_anomaly_summary()
        if anomaly_summary:
            detection_config = {
                'methods_used': ['Regression Residuals', 'Isolation Forest'],
                'thresholds': 'standard',
                'feature_count': 'multiple',
            }
            return anomaly_summary, detection_config
        return None, None

    def _get_risk_confidence_data(self) -> tuple:
        """
        Get risk scoring confidence data.

        Returns:
            Tuple of (risk_model_metrics, model_validation) or (None, None).
        """
        # Placeholder - would extract from actual risk scoring results
        return None, None


# ============================================================================
# Cached Computation Wrappers (for Streamlit Performance)
# ============================================================================
# These functions add caching to expensive operations to prevent re-computation
# on every Streamlit interaction. Cache keys include configuration parameters
# to ensure cache invalidation when settings change.


@st.cache_data(show_spinner=False, ttl=3600)
def _get_analyzer_and_prepare(
    df: pd.DataFrame,
    target_margin: float,
    high_discount_threshold: float,
    forecast_horizon: int,
) -> RevenueLeakageAnalyzer:
    """
    Cached analyzer initialization and data preparation.

    This prevents re-running data preparation on every interaction.
    Cache invalidates when configuration changes or after 1 hour.

    Args:
        df: Input DataFrame
        target_margin: Target profit margin
        high_discount_threshold: High discount threshold
        forecast_horizon: Forecast horizon

    Returns:
        Prepared RevenueLeakageAnalyzer instance
    """
    config = RevenueLeakageConfig(
        target_margin=target_margin,
        high_discount_threshold=high_discount_threshold,
        forecast_horizon=forecast_horizon,
    )
    analyzer = RevenueLeakageAnalyzer(df, config)
    analyzer.prepare_data()
    return analyzer


@st.cache_data(show_spinner=False, ttl=1800)
def _run_forecast_cached(
    _analyzer: RevenueLeakageAnalyzer,
    forecast_periods: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cached forecast computation.

    Args:
        _analyzer: Prepared analyzer (prefixed with _ to avoid hashing)
        forecast_periods: Number of periods to forecast

    Returns:
        (forecast_df, metrics_dict)
    """
    return _analyzer.forecast_leakage(forecast_periods)


@st.cache_data(show_spinner=False, ttl=1800)
def _run_anomaly_detection_cached(
    _analyzer: RevenueLeakageAnalyzer,
    contamination: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cached anomaly detection.

    Args:
        _analyzer: Prepared analyzer (prefixed with _ to avoid hashing)
        contamination: Contamination parameter

    Returns:
        (anomalies_df, summary_dict)
    """
    return _analyzer.detect_anomalies(contamination)


@st.cache_data(show_spinner=False, ttl=1800)
def _run_risk_scoring_cached(
    _analyzer: RevenueLeakageAnalyzer,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cached risk scoring.

    Args:
        _analyzer: Prepared analyzer (prefixed with _ to avoid hashing)

    Returns:
        (risk_scores_df, metadata_dict)
    """
    return _analyzer.score_leakage_risk()


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

    # Initialize analyzer using cached wrapper (prevents re-preparation on interactions)
    try:
        analyzer = _get_analyzer_and_prepare(
            df,
            target_margin=target_margin,
            high_discount_threshold=high_discount_thresh,
            forecast_horizon=forecast_horizon,
        )

        # Initialize LLM insight generator (use enhanced if available)
        if USE_ENHANCED_LLM_INSIGHTS and _EnhancedInsightGeneratorAvailable:
            llm_generator = EnhancedInsightGenerator(llm_client)
            st.info("✨ Using enhanced LLM insights with advanced reasoning scaffolds")
        else:
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
        _render_anomaly_detection(analyzer, anomaly_sensitivity, llm_generator)

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

    # Decision Intelligence: Root Cause Analysis
    st.markdown("---")
    st.subheader("🎯 Decision Intelligence: Root Cause Analysis")
    st.markdown("""
    **Why is leakage occurring?** Moving beyond _what_ to _why_ - understanding causal factors.
    """)

    with st.expander("🔍 AI Root Cause Analysis - Click to Expand", expanded=False):
        with st.spinner("Analyzing root causes..."):
            # Create AI Decision Intelligence orchestrator
            ai_decision = AIDecisionIntelligence(analyzer, llm_gen)

            # Get root cause context
            context = ai_decision._prepare_root_cause_context()

            # Generate root cause analysis
            root_cause_insights = llm_gen.generate_root_cause_analysis(context)
            st.markdown(root_cause_insights)

    # Decision Priorities
    with st.expander("🎯 Decision Priorities: What to Act On NOW", expanded=False):
        with st.spinner("Generating decision priorities..."):
            # Prepare decision signals
            signals = ai_decision._prepare_decision_signals(
                leakage_metrics=metrics,
                include_forecast=False,  # Not yet computed
                include_anomalies=False,  # Not yet computed
            )

            # Generate prioritized guidance
            decision_priorities = llm_gen.generate_decision_priority(signals)
            st.markdown(decision_priorities)


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

    run_forecast = st.button("▶️ Run Forecast Model")
    if not run_forecast:
        st.info("Click **Run Forecast Model** to train and view the leakage forecast.")
        return

    try:
        with st.spinner("Training forecasting model..."):
            # Use cached wrapper to prevent re-computation
            forecast_df, metrics = _run_forecast_cached(analyzer, forecast_periods)

            # Mark that forecast has been run and store metrics for other tabs
            st.session_state["forecast_has_run"] = True
            st.session_state["forecast_metrics"] = metrics

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

        # Model Confidence Narrative
        st.markdown("---")
        with st.expander("📊 Model Confidence & Reliability Assessment", expanded=False):
            st.markdown("""
            **How reliable is this forecast?** Understanding model confidence helps you make better decisions.
            """)

            with st.spinner("Assessing forecast confidence..."):
                # Prepare data quality context
                data_quality = {
                    'data_points': len(analyzer.df),
                    'volatility': 'moderate',  # Could be calculated from data
                    'trend_stability': 'stable',  # Could be calculated
                    'seasonality_strength': 'unknown',
                }

                # Generate confidence narrative
                confidence_narrative = llm_gen.generate_forecast_confidence_narrative(
                    forecast_metrics=metrics,
                    data_quality=data_quality,
                )
                st.markdown(confidence_narrative)

    except ValueError as e:
        st.error(f"❌ Forecasting error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def _render_anomaly_detection(
    analyzer: RevenueLeakageAnalyzer,
    contamination: float,
    llm_gen: LLMInsightGenerator,
) -> None:
    """Render anomaly detection tab."""
    st.subheader("📈 Anomaly Detection for Revenue Leakage")

    run_anomaly = st.button("▶️ Run Anomaly Detection")
    if not run_anomaly:
        st.info("Click **Run Anomaly Detection** to identify anomalous transactions.")
        return

    try:
        with st.spinner("Detecting anomalies..."):
            # Use cached wrapper to prevent re-computation
            anomalies_df, summary = _run_anomaly_detection_cached(analyzer, contamination)

            # Mark that anomaly detection has been run and store summary
            st.session_state["anomaly_has_run"] = True
            st.session_state["anomaly_summary"] = summary

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

        # Model Confidence Narrative
        st.markdown("---")
        with st.expander("📊 Anomaly Detection Confidence & Reliability", expanded=False):
            st.markdown("""
            **How reliable are these anomaly flags?** Understanding detection confidence helps prioritize investigations.
            """)

            with st.spinner("Assessing anomaly detection confidence..."):
                # Prepare anomaly metrics
                anomaly_metrics = {
                    'anomaly_count': summary['anomaly_count'],
                    'anomaly_rate': summary['anomaly_percentage'] / 100,
                    'high_risk_count': int(summary['anomaly_count'] * 0.3),  # Estimate
                    'medium_risk_count': int(summary['anomaly_count'] * 0.4),
                    'low_risk_count': int(summary['anomaly_count'] * 0.3),
                }

                # Prepare detection configuration
                detection_config = {
                    'methods_used': ['Regression Residuals', 'Isolation Forest'],
                    'thresholds': f'{contamination*100:.1f}% contamination',
                    'feature_count': 'multiple features (Sales, Profit, Discount, Margin)',
                }

                # Generate confidence narrative
                confidence_narrative = llm_gen.generate_anomaly_confidence_narrative(
                    anomaly_metrics=anomaly_metrics,
                    detection_config=detection_config,
                )
                st.markdown(confidence_narrative)

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

    # Decision Intelligence: Cross-Signal Insights
    st.markdown("---")
    st.subheader("🧠 Decision Intelligence: Cross-Model Synthesis")
    st.markdown("""
    **What do all our models tell us together?** Synthesizing insights from forecasting, anomaly detection, and risk scoring.
    """)

    with st.expander("🔗 Cross-Signal Intelligence - Click to Expand", expanded=False):
        # Check if we have run the relevant analyses
        has_forecast = st.session_state.get("forecast_has_run", False)
        has_anomalies = st.session_state.get("anomaly_has_run", False)
        has_risk = 'Leakage_Probability' in analyzer.df.columns  # risk tab not implemented yet

        if has_forecast or has_anomalies or has_risk:
            with st.spinner("Synthesizing cross-model insights..."):
                # Create AI Decision Intelligence orchestrator
                ai_decision = AIDecisionIntelligence(analyzer, llm_gen)

                # Prefer real forecast metrics if available
                forecast_summary = st.session_state.get("forecast_metrics")
                if not forecast_summary:
                    # Fallback rough estimates if user didn't run forecasting
                    forecast_summary = {
                        'historical_avg': metrics['total_leakage'] / 12,
                        'avg_monthly_forecast': metrics['total_leakage'] / 12 * 1.1,
                        'total_forecasted_leakage': metrics['total_leakage'] * 0.5,
                        'mae': metrics['total_leakage'] * 0.05,
                        'rmse': metrics['total_leakage'] * 0.07,
                    }

                # Anomaly summary: prefer stored; fall back to analyzer helper
                anomaly_summary = st.session_state.get("anomaly_summary")
                if not anomaly_summary and has_anomalies:
                    anomaly_summary = ai_decision._get_anomaly_summary()
                if not anomaly_summary:
                    anomaly_summary = {
                        'anomaly_count': 0,
                        'anomaly_rate': 0,
                        'high_risk_count': 0,
                        'medium_risk_count': 0,
                        'low_risk_count': 0,
                    }

                risk_summary = ai_decision._get_risk_summary() if has_risk else {
                    'high_risk_count': 0,
                    'medium_risk_count': 0,
                    'avg_risk_probability': 0,
                    'accuracy': 0.85,
                    'roc_auc': 0.88,
                }

                # Generate cross-signal insights
                cross_signal_insights = llm_gen.generate_cross_signal_insights(
                    forecast_summary=forecast_summary,
                    anomaly_summary=anomaly_summary,
                    risk_summary=risk_summary,
                    leakage_metrics=metrics,
                )
                st.markdown(cross_signal_insights)
        else:
            st.info(
                "📊 Cross-signal insights will be available after you run "
                "**Forecasting** and **Anomaly Detection** from their respective tabs."
            )

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
