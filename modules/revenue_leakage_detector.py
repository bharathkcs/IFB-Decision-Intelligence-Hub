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
<<<<<<< HEAD
from typing import Dict, List, Optional, Tuple, Any, Callable
=======
from typing import Dict, List, Optional, Tuple, Any
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    IsolationForest
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

    # Required columns
    required_columns: List[str] = field(default_factory=lambda: ['Sales', 'Profit'])
    optional_columns: List[str] = field(default_factory=lambda: [
        'Discount', 'Category', 'Sub Category', 'Region',
        'State', 'Customer Segment', 'Order Date', 'Order_Date'
    ])


# ============================================================================
# Core Analytics Engine
# ============================================================================

class RevenueLeakageAnalyzer:
    """
    Core analytics engine for revenue leakage detection and forecasting.
<<<<<<< HEAD

    This class contains all business logic for:
    - Data preparation and validation
    - Feature engineering
    - Leakage metrics computation
    - Time-series forecasting
    - Anomaly detection
    - Risk scoring

    No Streamlit UI code is included in this class.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[RevenueLeakageConfig] = None):
        """
        Initialize the analyzer with data and configuration.

        Args:
            df: Input DataFrame containing sales/transaction data
            config: Configuration object (uses defaults if None)

        Raises:
            ValueError: If required columns are missing
        """
        self.df = df.copy()
        self.config = config or RevenueLeakageConfig()
        self.date_column: Optional[str] = None
        self.prepared = False

        # Validate schema
        self._validate_schema()

    def _validate_schema(self) -> None:
        """
        Validate that required columns exist in the dataset.

        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in self.config.required_columns
                       if col not in self.df.columns]

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
        - Adds lag and rolling features

        Returns:
            Prepared DataFrame with all engineered features
        """
        if self.prepared:
            return self.df

        # Convert numeric fields
        numeric_fields = ['Sales', 'Profit', 'Discount', 'Quantity']
        for field in numeric_fields:
            if field in self.df.columns:
                self.df[field] = pd.to_numeric(self.df[field], errors='coerce')

        # Detect and convert date columns
        self._detect_and_convert_dates()

        # Normalize discount logic
        if 'Discount' in self.df.columns:
            self._normalize_discounts()

        # Add leakage metrics
        self._compute_leakage_metrics()

        # Add time-based features if date column exists
        if self.date_column:
            self._add_time_features()

        self.prepared = True
        return self.df

    def _detect_and_convert_dates(self) -> None:
        """Detect and convert date columns to datetime."""
        for col in self.df.columns:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                if self.date_column is None:
                    self.date_column = col
                continue

            # Check if column name suggests it's a date
=======
    """

    def __init__(self, df: pd.DataFrame, config: Optional[RevenueLeakageConfig] = None):
        self.df = df.copy()
        self.config = config or RevenueLeakageConfig()
        self.date_column: Optional[str] = None
        self.prepared = False

        self._validate_schema()

    def _validate_schema(self) -> None:
        missing_cols = [col for col in self.config.required_columns
                        if col not in self.df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(self.df.columns)}"
            )

    def prepare_data(self) -> pd.DataFrame:
        if self.prepared:
            return self.df

        # Convert numeric fields
        numeric_fields = ['Sales', 'Profit', 'Discount', 'Quantity']
        for field in numeric_fields:
            if field in self.df.columns:
                self.df[field] = pd.to_numeric(self.df[field], errors='coerce')

        # Detect and convert date columns
        self._detect_and_convert_dates()

        # Normalize discount logic
        if 'Discount' in self.df.columns:
            self._normalize_discounts()

        # Add leakage metrics
        self._compute_leakage_metrics()

        # Add time-based features if date column exists
        if self.date_column:
            self._add_time_features()

        self.prepared = True
        return self.df

    def _detect_and_convert_dates(self) -> None:
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                if self.date_column is None:
                    self.date_column = col
                continue

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            if 'date' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    if self.date_column is None:
                        self.date_column = col
                except Exception:
                    pass

    def _normalize_discounts(self) -> None:
<<<<<<< HEAD
        """
        Normalize discount values to fractions and handle edge cases.

        Detects if discounts are in percentage format (>1) and converts to fractions.
        Clips values to reasonable range [0, 0.9].
        """
        # Check if values look like percentages
        if self.df['Discount'].abs().max() > 1.0:
            self.df['Discount'] = self.df['Discount'] / 100.0

        # Clip to reasonable range
        self.df['Discount'] = self.df['Discount'].clip(0, 0.9)

        # Handle negative or extreme values
        self.df.loc[self.df['Discount'] < 0, 'Discount'] = 0
        self.df.loc[self.df['Discount'] > 0.9, 'Discount'] = 0.9

    def _compute_leakage_metrics(self) -> None:
        """
        Compute comprehensive leakage metrics.

        Creates:
        - Profit_Margin: Actual profit margin percentage
        - Expected_Profit: Expected profit based on target margin
        - Margin_Leakage: Shortfall from expected profit
        - Discount_Leakage: Revenue lost to discounts
        - Total_Leakage: Combined leakage
        - Leakage_Rate: Leakage as percentage of sales
        """
        # Profit margin (with safe division)
        self.df['Profit_Margin'] = np.where(
            self.df['Sales'] > 0,
            (self.df['Profit'] / self.df['Sales']) * 100,
            0
        )

        # Expected profit based on target margin
        if 'Category' in self.df.columns and self.config.category_margins:
            # Use category-specific margins if available
            self.df['Target_Margin'] = self.df['Category'].map(
                self.config.category_margins
            ).fillna(self.config.target_margin)
        else:
            self.df['Target_Margin'] = self.config.target_margin

        self.df['Expected_Profit'] = self.df['Sales'] * self.df['Target_Margin']

        # Margin leakage (only positive - shortfall from expected)
        self.df['Margin_Leakage'] = np.maximum(
            self.df['Expected_Profit'] - self.df['Profit'],
            0
        )

        # Discount leakage
        if 'Discount' in self.df.columns:
            self.df['Discount_Leakage'] = self.df['Sales'] * self.df['Discount']
        else:
            self.df['Discount_Leakage'] = 0

        # Total leakage
        self.df['Total_Leakage'] = (
            self.df['Margin_Leakage'] + self.df['Discount_Leakage']
        )

        # Leakage rate (with safe division)
        self.df['Leakage_Rate'] = np.where(
            self.df['Sales'] > 0,
            (self.df['Total_Leakage'] / self.df['Sales']) * 100,
            0
        )

    def _add_time_features(self) -> None:
        """Add time-based features for temporal analysis."""
        if not self.date_column or self.date_column not in self.df.columns:
            return

        date_col = self.df[self.date_column]

        # Basic time features
        self.df['Year'] = date_col.dt.year
        self.df['Month'] = date_col.dt.month
        self.df['Quarter'] = date_col.dt.quarter
        self.df['YearMonth'] = date_col.dt.to_period('M')
        self.df['DayOfWeek'] = date_col.dt.dayofweek

    def compute_leakage_metrics(self) -> Dict[str, float]:
        """
        Compute overall leakage metrics summary.

        Returns:
            Dictionary containing key leakage metrics
        """
        if not self.prepared:
            self.prepare_data()

        return {
            'total_sales': float(self.df['Sales'].sum()),
            'total_profit': float(self.df['Profit'].sum()),
            'avg_profit_margin': float(
                (self.df['Profit'].sum() / self.df['Sales'].sum() * 100)
                if self.df['Sales'].sum() > 0 else 0
            ),
            'total_discount_leakage': float(self.df['Discount_Leakage'].sum()),
            'total_margin_leakage': float(self.df['Margin_Leakage'].sum()),
            'total_leakage': float(self.df['Total_Leakage'].sum()),
            'avg_leakage_rate': float(self.df['Leakage_Rate'].mean()),
            'negative_profit_count': int((self.df['Profit'] < 0).sum()),
            'negative_profit_amount': float(
                self.df.loc[self.df['Profit'] < 0, 'Profit'].sum()
            )
        }

    def build_time_series(
        self,
        level: str = "overall",
        groupby_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build time series aggregation at specified level.

        Args:
            level: Aggregation level ("overall", "region", "category")
            groupby_col: Optional specific column to group by

        Returns:
            Time series DataFrame with period index

        Raises:
            ValueError: If date column not available
        """
        if not self.date_column:
            raise ValueError("No date column available for time series analysis")

        if not self.prepared:
            self.prepare_data()

        # Determine grouping
        if groupby_col:
            group_cols = ['YearMonth', groupby_col]
        elif level == "region" and 'Region' in self.df.columns:
            group_cols = ['YearMonth', 'Region']
        elif level == "category" and 'Category' in self.df.columns:
            group_cols = ['YearMonth', 'Category']
        else:
            group_cols = ['YearMonth']

        # Aggregate
        ts_data = self.df.groupby(group_cols).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Total_Leakage': 'sum',
            'Discount_Leakage': 'sum',
            'Margin_Leakage': 'sum'
        }).reset_index()

        # Compute leakage rate
        ts_data['Leakage_Rate'] = np.where(
            ts_data['Sales'] > 0,
            (ts_data['Total_Leakage'] / ts_data['Sales']) * 100,
            0
        )

        return ts_data

    def forecast_leakage(
        self,
        forecast_periods: int = 6
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Forecast future revenue leakage using tree-based regression.

        This method:
        - Aggregates data to monthly level
        - Engineers lag and rolling features
        - Trains gradient boosting model with walk-forward validation
        - Forecasts future periods
        - Computes performance metrics

        Args:
            forecast_periods: Number of months to forecast

        Returns:
            Tuple of (forecast_df, metrics_dict)
            - forecast_df: DataFrame with historical and forecasted values
            - metrics_dict: Model performance metrics (MAE, RMSE, etc.)

        Raises:
            ValueError: If insufficient historical data
        """
        if not self.date_column:
            raise ValueError("No date column available for forecasting")

        if not self.prepared:
            self.prepare_data()

        # Build monthly time series
        monthly_data = self.build_time_series(level="overall")

        # Convert to datetime index
=======
        if self.df['Discount'].abs().max() > 1.0:
            self.df['Discount'] = self.df['Discount'] / 100.0

        self.df['Discount'] = self.df['Discount'].clip(0, 0.9)
        self.df.loc[self.df['Discount'] < 0, 'Discount'] = 0
        self.df.loc[self.df['Discount'] > 0.9, 'Discount'] = 0.9

    def _compute_leakage_metrics(self) -> None:
        self.df['Profit_Margin'] = np.where(
            self.df['Sales'] > 0,
            (self.df['Profit'] / self.df['Sales']) * 100,
            0
        )

        if 'Category' in self.df.columns and self.config.category_margins:
            self.df['Target_Margin'] = self.df['Category'].map(
                self.config.category_margins
            ).fillna(self.config.target_margin)
        else:
            self.df['Target_Margin'] = self.config.target_margin

        self.df['Expected_Profit'] = self.df['Sales'] * self.df['Target_Margin']

        self.df['Margin_Leakage'] = np.maximum(
            self.df['Expected_Profit'] - self.df['Profit'],
            0
        )

        if 'Discount' in self.df.columns:
            self.df['Discount_Leakage'] = self.df['Sales'] * self.df['Discount']
        else:
            self.df['Discount_Leakage'] = 0

        self.df['Total_Leakage'] = (
            self.df['Margin_Leakage'] + self.df['Discount_Leakage']
        )

        self.df['Leakage_Rate'] = np.where(
            self.df['Sales'] > 0,
            (self.df['Total_Leakage'] / self.df['Sales']) * 100,
            0
        )

    def _add_time_features(self) -> None:
        if not self.date_column or self.date_column not in self.df.columns:
            return

        date_col = self.df[self.date_column]

        self.df['Year'] = date_col.dt.year
        self.df['Month'] = date_col.dt.month
        self.df['Quarter'] = date_col.dt.quarter
        self.df['YearMonth'] = date_col.dt.to_period('M')
        self.df['DayOfWeek'] = date_col.dt.dayofweek

    def compute_leakage_metrics(self) -> Dict[str, float]:
        if not self.prepared:
            self.prepare_data()

        return {
            'total_sales': float(self.df['Sales'].sum()),
            'total_profit': float(self.df['Profit'].sum()),
            'avg_profit_margin': float(
                (self.df['Profit'].sum() / self.df['Sales'].sum() * 100)
                if self.df['Sales'].sum() > 0 else 0
            ),
            'total_discount_leakage': float(self.df['Discount_Leakage'].sum()),
            'total_margin_leakage': float(self.df['Margin_Leakage'].sum()),
            'total_leakage': float(self.df['Total_Leakage'].sum()),
            'avg_leakage_rate': float(self.df['Leakage_Rate'].mean()),
            'negative_profit_count': int((self.df['Profit'] < 0).sum()),
            'negative_profit_amount': float(
                self.df.loc[self.df['Profit'] < 0, 'Profit'].sum()
            )
        }

    def build_time_series(
        self,
        level: str = "overall",
        groupby_col: Optional[str] = None
    ) -> pd.DataFrame:
        if not self.date_column:
            raise ValueError("No date column available for time series analysis")

        if not self.prepared:
            self.prepare_data()

        if groupby_col:
            group_cols = ['YearMonth', groupby_col]
        elif level == "region" and 'Region' in self.df.columns:
            group_cols = ['YearMonth', 'Region']
        elif level == "category" and 'Category' in self.df.columns:
            group_cols = ['YearMonth', 'Category']
        else:
            group_cols = ['YearMonth']

        ts_data = self.df.groupby(group_cols).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Total_Leakage': 'sum',
            'Discount_Leakage': 'sum',
            'Margin_Leakage': 'sum'
        }).reset_index()

        ts_data['Leakage_Rate'] = np.where(
            ts_data['Sales'] > 0,
            (ts_data['Total_Leakage'] / ts_data['Sales']) * 100,
            0
        )

        return ts_data

    def forecast_leakage(
        self,
        forecast_periods: int = 6
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if not self.date_column:
            raise ValueError("No date column available for forecasting")

        if not self.prepared:
            self.prepare_data()

        monthly_data = self.build_time_series(level="overall")

        # Remove rows with missing YearMonth
        monthly_data = monthly_data.dropna(subset=['YearMonth'])

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        monthly_data['Date'] = monthly_data['YearMonth'].dt.to_timestamp()
        monthly_data = monthly_data.set_index('Date').sort_index()

        if len(monthly_data) < 6:
            raise ValueError(
                f"Insufficient data for forecasting. Need at least 6 months, "
                f"got {len(monthly_data)}"
            )

<<<<<<< HEAD
        # Engineer features
        monthly_data = self._add_lag_rolling_features(monthly_data)

        # Drop rows with NaN (from lag/rolling features)
=======
        monthly_data = self._add_lag_rolling_features(monthly_data)

        # Drop rows with NaN from lag/rolling features
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        monthly_data = monthly_data.dropna()

        if len(monthly_data) < 4:
            raise ValueError("Insufficient data after feature engineering")

<<<<<<< HEAD
        # Prepare features and target
        feature_cols = ['Month', 'Quarter', 'Year']

        # Add engineered features
=======
        # Add calendar features
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        monthly_data['Month'] = monthly_data.index.month
        monthly_data['Quarter'] = monthly_data.index.quarter
        monthly_data['Year'] = monthly_data.index.year

<<<<<<< HEAD
        # Add lag and rolling features
=======
        feature_cols = ['Month', 'Quarter', 'Year']

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        for lag in self.config.lag_periods:
            col = f'Leakage_Lag{lag}'
            if col in monthly_data.columns:
                feature_cols.append(col)

        for window in self.config.rolling_windows:
            col = f'Rolling_{window}M_Leakage'
            if col in monthly_data.columns:
                feature_cols.append(col)

<<<<<<< HEAD
        X = monthly_data[feature_cols]
        y = monthly_data['Total_Leakage']

        # Time-based split
=======
        # Build X, y and clean NaNs/infs
        X = monthly_data[feature_cols].copy()
        y = monthly_data['Total_Leakage'].copy()

        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)

        valid_mask = (~X.isna().any(axis=1)) & (~y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        monthly_data = monthly_data.loc[valid_mask]

        if len(X) < 4:
            raise ValueError("Insufficient clean data for forecasting after NaN removal")

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        split_idx = int(len(X) * self.config.train_test_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

<<<<<<< HEAD
        # Train model
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            learning_rate=0.1,
            max_depth=4
        )
        model.fit(X_train, y_train)

<<<<<<< HEAD
        # Evaluate on test set
        metrics = {}
=======
        metrics: Dict[str, float] = {}
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

<<<<<<< HEAD
            # Compute MAPE (avoiding division by zero)
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(
                    np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])
                ) * 100
                metrics['mape'] = float(mape)

<<<<<<< HEAD
        # Generate future forecasts
=======
        # Future forecasting
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        last_date = monthly_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='MS'
        )

<<<<<<< HEAD
        # Iteratively forecast future periods
        future_predictions = []
        forecast_df = monthly_data.copy()

        for i, future_date in enumerate(future_dates):
            # Build features for this future period
=======
        future_predictions: List[float] = []
        forecast_df = monthly_data.copy()

        for future_date in future_dates:
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            future_row = {
                'Month': future_date.month,
                'Quarter': future_date.quarter,
                'Year': future_date.year
            }

<<<<<<< HEAD
            # Add lag features from recent history
            for lag in self.config.lag_periods:
                lag_idx = len(forecast_df) - lag
                if lag_idx >= 0:
                    if 'Forecast_Leakage' in forecast_df.columns:
                        future_row[f'Leakage_Lag{lag}'] = (
                            forecast_df.iloc[lag_idx].get('Forecast_Leakage') or
                            forecast_df.iloc[lag_idx]['Total_Leakage']
                        )
                    else:
                        future_row[f'Leakage_Lag{lag}'] = (
                            forecast_df.iloc[lag_idx]['Total_Leakage']
                        )
                else:
                    future_row[f'Leakage_Lag{lag}'] = 0

            # Add rolling features
            for window in self.config.rolling_windows:
                window_data = forecast_df.tail(window)
                if 'Forecast_Leakage' in window_data.columns:
                    values = []
                    for _, row in window_data.iterrows():
                        values.append(
                            row.get('Forecast_Leakage') or row['Total_Leakage']
                        )
                    future_row[f'Rolling_{window}M_Leakage'] = np.mean(values)
                else:
                    future_row[f'Rolling_{window}M_Leakage'] = (
                        window_data['Total_Leakage'].mean()
                    )

            # Predict
            X_future = pd.DataFrame([future_row])[feature_cols]
            prediction = model.predict(X_future)[0]
            future_predictions.append(prediction)

            # Add to forecast_df for next iteration
            new_row = pd.DataFrame({
                'Total_Leakage': [0],
=======
            # Lag features from recent history / forecasts
            for lag in self.config.lag_periods:
                lag_idx = len(forecast_df) - lag
                if lag_idx >= 0:
                    prev_row = forecast_df.iloc[lag_idx]

                    if ('Forecast_Leakage' in prev_row.index and
                            not pd.isna(prev_row['Forecast_Leakage'])):
                        value = prev_row['Forecast_Leakage']
                    elif not pd.isna(prev_row.get('Total_Leakage', 0)):
                        value = prev_row['Total_Leakage']
                    else:
                        value = 0
                else:
                    value = 0
                future_row[f'Leakage_Lag{lag}'] = value

            # Rolling features mixing historical + forecasted
            for window in self.config.rolling_windows:
                window_data = forecast_df.tail(window)
                values: List[float] = []
                for _, row in window_data.iterrows():
                    if ('Forecast_Leakage' in row.index and
                            not pd.isna(row['Forecast_Leakage'])):
                        values.append(row['Forecast_Leakage'])
                    elif not pd.isna(row.get('Total_Leakage', 0)):
                        values.append(row['Total_Leakage'])
                    else:
                        values.append(0)
                future_row[f'Rolling_{window}M_Leakage'] = float(np.mean(values)) if values else 0.0

            X_future = pd.DataFrame([future_row])[feature_cols]
            X_future = X_future.apply(pd.to_numeric, errors='coerce')
            X_future = X_future.replace([np.inf, -np.inf], np.nan).fillna(0)

            prediction = float(model.predict(X_future)[0])
            future_predictions.append(prediction)

            new_row = pd.DataFrame({
                'Total_Leakage': [0.0],
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
                'Forecast_Leakage': [prediction]
            }, index=[future_date])
            forecast_df = pd.concat([forecast_df, new_row])

<<<<<<< HEAD
        # Build output dataframe
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        result_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Leakage': future_predictions
        })

<<<<<<< HEAD
        # Add historical data
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        historical_df = pd.DataFrame({
            'Date': monthly_data.index,
            'Historical_Leakage': monthly_data['Total_Leakage'].values
        })

<<<<<<< HEAD
        # Combine
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        full_forecast = pd.concat([
            historical_df.set_index('Date'),
            result_df.set_index('Date')
        ], axis=1).reset_index()
        full_forecast.columns = ['Date', 'Historical_Leakage', 'Forecasted_Leakage']

<<<<<<< HEAD
        # Add summary metrics
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        metrics['total_forecasted_leakage'] = float(sum(future_predictions))
        metrics['avg_monthly_forecast'] = float(np.mean(future_predictions))
        metrics['historical_avg'] = float(monthly_data['Total_Leakage'].mean())

        return full_forecast, metrics

    def _add_lag_rolling_features(self, ts_df: pd.DataFrame) -> pd.DataFrame:
<<<<<<< HEAD
        """
        Add lag and rolling window features to time series.

        Args:
            ts_df: Time series DataFrame

        Returns:
            DataFrame with additional lag and rolling features
        """
        df = ts_df.copy()

        # Lag features
        for lag in self.config.lag_periods:
            df[f'Leakage_Lag{lag}'] = df['Total_Leakage'].shift(lag)

        # Rolling features
=======
        df = ts_df.copy()

        for lag in self.config.lag_periods:
            df[f'Leakage_Lag{lag}'] = df['Total_Leakage'].shift(lag)

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        for window in self.config.rolling_windows:
            df[f'Rolling_{window}M_Leakage'] = (
                df['Total_Leakage'].rolling(window=window, min_periods=1).mean()
            )

        return df

    def detect_anomalies(
        self,
        contamination: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
<<<<<<< HEAD
        """
        Detect anomalous transactions using regression residuals and IsolationForest.

        This method:
        - Trains a regression model to predict expected profit
        - Computes residuals (expected - actual)
        - Identifies anomalies where residual is high AND profit is low
        - Optionally combines with IsolationForest

        Args:
            contamination: Expected proportion of outliers (0-1)

        Returns:
            Tuple of (anomalies_df, summary_dict)
            - anomalies_df: DataFrame of detected anomalies
            - summary_dict: Summary statistics
        """
        if not self.prepared:
            self.prepare_data()

        # Prepare features
        feature_cols = ['Sales', 'Profit', 'Profit_Margin',
                       'Margin_Leakage', 'Total_Leakage']
=======
        if not self.prepared:
            self.prepare_data()

        feature_cols = ['Sales', 'Profit', 'Profit_Margin',
                        'Margin_Leakage', 'Total_Leakage']
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)

        if 'Discount' in self.df.columns:
            feature_cols.append('Discount')

<<<<<<< HEAD
        # Encode categorical features
        categorical_cols = []
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        df_encoded = self.df.copy()

        for col in ['Category', 'Sub Category', 'Region', 'Customer Segment']:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_Encoded'] = le.fit_transform(
                    df_encoded[col].astype(str)
                )
                feature_cols.append(f'{col}_Encoded')
<<<<<<< HEAD
                categorical_cols.append(col)

        # Drop NaN
        df_features = df_encoded[feature_cols].dropna()
=======

        df_features = df_encoded[feature_cols].copy()
        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)

        if len(df_features) < 100:
            raise ValueError("Insufficient data for anomaly detection (need >= 100 records)")

<<<<<<< HEAD
        # Train regression model to predict expected profit margin
        X = df_features.drop(['Profit'], axis=1, errors='ignore')
        y = df_features['Profit']

        # Use 80% for training
=======
        X = df_features.drop(['Profit'], axis=1, errors='ignore')
        y = df_features['Profit']

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=self.config.random_state,
            max_depth=4
        )
        model.fit(X_train, y_train)

<<<<<<< HEAD
        # Predict on full dataset
        y_pred_all = model.predict(X)

        # Compute residuals
=======
        y_pred_all = model.predict(X)

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        residuals = y_pred_all - y.values
        df_features['Residual'] = residuals
        df_features['Expected_Profit_Model'] = y_pred_all

<<<<<<< HEAD
        # Define anomaly threshold based on residuals
        residual_threshold = np.percentile(residuals, self.config.anomaly_percentile * 100)

        # Anomalies: high residual AND low/negative actual profit
=======
        residual_threshold = np.percentile(residuals, self.config.anomaly_percentile * 100)

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        df_features['Is_Anomaly_Residual'] = (
            (df_features['Residual'] > residual_threshold) &
            (df_features['Profit'] < df_features['Profit'].quantile(0.25))
        )

<<<<<<< HEAD
        # Apply IsolationForest
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.config.random_state
        )
        iso_predictions = iso_forest.fit_predict(X)
        df_features['Is_Anomaly_ISO'] = (iso_predictions == -1)

<<<<<<< HEAD
        # Combine both methods
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        df_features['Is_Anomaly'] = (
            df_features['Is_Anomaly_Residual'] | df_features['Is_Anomaly_ISO']
        )

<<<<<<< HEAD
        # Anomaly score (normalized residual)
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        df_features['Anomaly_Score'] = (
            (df_features['Residual'] - residuals.mean()) / residuals.std()
        )

<<<<<<< HEAD
        # Get original data for anomalies
        anomaly_indices = df_features[df_features['Is_Anomaly']].index
        anomalies_full = self.df.loc[anomaly_indices].copy()

        # Add anomaly metrics
        anomalies_full['Residual'] = df_features.loc[anomaly_indices, 'Residual']
        anomalies_full['Anomaly_Score'] = df_features.loc[anomaly_indices, 'Anomaly_Score']

        # Add flags
=======
        anomaly_indices = df_features[df_features['Is_Anomaly']].index
        anomalies_full = self.df.loc[anomaly_indices].copy()

        anomalies_full['Residual'] = df_features.loc[anomaly_indices, 'Residual']
        anomalies_full['Anomaly_Score'] = df_features.loc[anomaly_indices, 'Anomaly_Score']

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        anomalies_full['Flags'] = anomalies_full.apply(
            lambda row: self._generate_anomaly_flags(row), axis=1
        )

<<<<<<< HEAD
        # Summary
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        summary = {
            'total_transactions': len(self.df),
            'anomaly_count': len(anomalies_full),
            'anomaly_percentage': len(anomalies_full) / len(self.df) * 100,
            'total_sales_at_risk': float(anomalies_full['Sales'].sum()),
            'total_leakage_in_anomalies': float(anomalies_full['Total_Leakage'].sum()),
            'avg_anomaly_score': float(anomalies_full['Anomaly_Score'].mean()),
            'residual_threshold': float(residual_threshold)
        }

        return anomalies_full, summary

    def _generate_anomaly_flags(self, row: pd.Series) -> str:
<<<<<<< HEAD
        """Generate descriptive flags for anomaly."""
        flags = []
=======
        flags: List[str] = []
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)

        if row.get('Profit', 0) < 0:
            flags.append('NEGATIVE_PROFIT')

        if 'Discount' in row and row.get('Discount', 0) > self.config.high_discount_threshold:
            flags.append('HIGH_DISCOUNT')

        if row.get('Profit_Margin', 0) < self.config.low_margin_threshold * 100:
            flags.append('LOW_MARGIN')

        if row.get('Margin_Leakage', 0) > row.get('Expected_Profit', 0) * 0.5:
            flags.append('HIGH_MARGIN_LEAKAGE')

        return ', '.join(flags) if flags else 'OUTLIER'

    def score_leakage_risk(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
<<<<<<< HEAD
        """
        Score each transaction for leakage risk using classification.

        Trains a binary classifier to predict probability of significant leakage.

        Returns:
            Tuple of (risk_scores_df, metadata_dict)
            - risk_scores_df: DataFrame with leakage probabilities
            - metadata_dict: Feature importance and model info
        """
        if not self.prepared:
            self.prepare_data()

        # Create binary target: high leakage flag
=======
        if not self.prepared:
            self.prepare_data()

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        leakage_threshold = self.df['Total_Leakage'].quantile(0.75)
        self.df['Leakage_Flag'] = (
            (self.df['Total_Leakage'] > leakage_threshold) |
            (self.df['Profit_Margin'] < 0)
        ).astype(int)

<<<<<<< HEAD
        # Prepare features
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        feature_cols = ['Sales', 'Profit_Margin']

        if 'Discount' in self.df.columns:
            feature_cols.append('Discount')

<<<<<<< HEAD
        # Encode categoricals
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        df_encoded = self.df.copy()
        for col in ['Category', 'Sub Category', 'Region', 'Customer Segment']:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_Encoded'] = le.fit_transform(
                    df_encoded[col].astype(str)
                )
                feature_cols.append(f'{col}_Encoded')

<<<<<<< HEAD
        # Prepare dataset
        df_model = df_encoded[feature_cols + ['Leakage_Flag']].dropna()
=======
        df_model = df_encoded[feature_cols + ['Leakage_Flag']].copy()
        df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)

        X = df_model[feature_cols]
        y = df_model['Leakage_Flag']

<<<<<<< HEAD
        # Train classifier
=======
        if len(X) < 20:
            raise ValueError("Insufficient data for risk scoring")

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        clf = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            max_depth=4
        )
        clf.fit(X, y)

<<<<<<< HEAD
        # Predict probabilities
        leakage_probs = clf.predict_proba(X)[:, 1]

        # Build result dataframe
        result_df = self.df.loc[df_model.index].copy()
        result_df['Leakage_Probability'] = leakage_probs

        # Feature importance
=======
        leakage_probs = clf.predict_proba(X)[:, 1]

        result_df = self.df.loc[df_model.index].copy()
        result_df['Leakage_Probability'] = leakage_probs

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)

<<<<<<< HEAD
        # Metadata
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        metadata = {
            'feature_importance': feature_importance.to_dict('records'),
            'high_risk_count': int((leakage_probs > 0.7).sum()),
            'medium_risk_count': int(((leakage_probs > 0.4) & (leakage_probs <= 0.7)).sum()),
            'low_risk_count': int((leakage_probs <= 0.4).sum()),
            'avg_probability': float(leakage_probs.mean())
        }

        return result_df, metadata


# ============================================================================
# LLM Insight Generator
# ============================================================================

class LLMInsightGenerator:
    """
    Handles LLM-based narrative insight generation.
<<<<<<< HEAD

    This class wraps LLM client interactions and provides structured
    methods for generating business insights from numeric data.

    Important: This class NEVER asks the LLM to do numeric calculations.
    It only passes computed numbers and asks for interpretation.
    """

    def __init__(self, llm_client: Any):
        """
        Initialize with LLM client.

        Args:
            llm_client: LLM client with conversational_response method
        """
        self.llm = llm_client

    def _call_llm(self, prompt: str, timeout: int = 30) -> str:
        """
        Safely call LLM with error handling.

        Args:
            prompt: Prompt text
            timeout: Timeout in seconds

        Returns:
            LLM response text or error message
        """
=======
    """

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    def _call_llm(self, prompt: str, timeout: int = 30) -> str:
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        try:
            response = self.llm.conversational_response([
                {'sender': 'user', 'text': prompt}
            ])
            return response.get('text', 'No response generated')
        except Exception as e:
            return f"⚠️ Error generating insights: {str(e)}"

    def generate_executive_summary(self, metrics: Dict[str, float]) -> str:
<<<<<<< HEAD
        """
        Generate executive summary insights from metrics.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Executive summary text
        """
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
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

**Please provide:**
1. **Top 3 Critical Findings** (bullet points)
2. **Impact Assessment** (1-2 sentences on business impact)
3. **Priority Areas** (what needs immediate attention)

Keep it concise and business-focused.
"""
        return self._call_llm(prompt)

    def generate_discount_insights(
        self,
        discount_stats: Dict[str, float],
        category_stats: Optional[pd.DataFrame] = None
    ) -> str:
<<<<<<< HEAD
        """
        Generate insights on discount patterns.

        Args:
            discount_stats: Discount distribution statistics
            category_stats: Optional category-level discount stats

        Returns:
            Discount insights text
        """
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        category_info = ""
        if category_stats is not None and not category_stats.empty:
            top_3 = category_stats.head(3)
            category_info = "\n**Top 3 Categories by Discount Leakage:**\n"
<<<<<<< HEAD
            for idx, row in top_3.iterrows():
                category_info += f"- {idx}: ${row.get('Discount_Leakage', 0):,.2f}\n"
=======
            for category, leakage in top_3.items():
                category_info += f"- {category}: ${leakage:,.2f}\n"
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)

        prompt = f"""
Analyze these discount patterns and provide strategic insights:

**Discount Statistics:**
- Average Discount Rate: {discount_stats.get('mean', 0):.2%}
- Median Discount: {discount_stats.get('50%', 0):.2%}
- Max Discount: {discount_stats.get('max', 0):.2%}
- High Discount (>20%) Transactions: {discount_stats.get('high_discount_count', 0)}
- Revenue Lost to High Discounts: ${discount_stats.get('high_discount_loss', 0):,.2f}

{category_info}

**Please provide:**
1. Are current discount levels sustainable or excessive?
2. Which categories show problematic discount patterns?
3. Specific recommendations for discount optimization

Be specific and actionable.
"""
        return self._call_llm(prompt)

    def generate_forecast_insights(self, forecast_summary: Dict[str, float]) -> str:
<<<<<<< HEAD
        """
        Generate insights on leakage forecast.

        Args:
            forecast_summary: Forecast metrics and trends

        Returns:
            Forecast insights text
        """
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        trend = "increasing" if (
            forecast_summary.get('avg_monthly_forecast', 0) >
            forecast_summary.get('historical_avg', 0)
        ) else "decreasing"

        prompt = f"""
Analyze this revenue leakage forecast and provide strategic insights:

**Forecast Summary:**
- Historical Average Monthly Leakage: ${forecast_summary.get('historical_avg', 0):,.2f}
- Forecasted Average Monthly Leakage: ${forecast_summary.get('avg_monthly_forecast', 0):,.2f}
- Total Forecasted Leakage: ${forecast_summary.get('total_forecasted_leakage', 0):,.2f}
- Trend Direction: {trend}
- Model MAE: ${forecast_summary.get('mae', 0):,.2f}
- Model RMSE: ${forecast_summary.get('rmse', 0):,.2f}

**Please provide:**
1. **Trend Analysis**: What does the forecast tell us about future leakage?
2. **Business Impact**: Potential financial implications
3. **Preventive Actions**: Specific steps to mitigate forecasted leakage

Keep it concise and actionable.
"""
        return self._call_llm(prompt)

    def generate_recommendations(
        self,
        summary: Dict[str, Any],
        top_categories: Optional[List[str]] = None,
        top_regions: Optional[List[str]] = None
    ) -> str:
<<<<<<< HEAD
        """
        Generate comprehensive recommendations.

        Args:
            summary: Overall analysis summary
            top_categories: List of worst-performing categories
            top_regions: List of worst-performing regions

        Returns:
            Comprehensive recommendations text
        """
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        category_info = ""
        if top_categories:
            category_info = f"\n**Worst Performing Categories:** {', '.join(top_categories)}"

        region_info = ""
        if top_regions:
            region_info = f"\n**Worst Performing Regions:** {', '.join(top_regions)}"

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

**Please provide structured recommendations:**

### 1. Immediate Actions (0-30 days)
- Quick wins to stop leakage
- Emergency measures

### 2. Short-term Strategy (1-3 months)
- Pricing optimization
- Discount policy improvements
- Product mix adjustments

### 3. Long-term Strategy (3-12 months)
- Strategic business model changes
- Portfolio restructuring

### 4. Key Metrics to Track
- Specific KPIs
- Target values

### 5. Expected Impact
- Revenue recovery potential
- Timeline for results

Be specific, quantify where possible, and prioritize by impact.
"""
        return self._call_llm(prompt)


# ============================================================================
# Streamlit UI Layer
# ============================================================================

def run_revenue_leakage_app(df: pd.DataFrame, llm_client: Any) -> None:
<<<<<<< HEAD
    """
    Main Streamlit application for revenue leakage analysis.

    Args:
        df: Input DataFrame
        llm_client: LLM client instance
    """
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.header("🔍 AI-Powered Revenue Leakage Analysis")
    st.markdown("""
    Comprehensive revenue leakage detection across discounting, profitability,
    product performance, regional trends, and predictive forecasting.
    """)

<<<<<<< HEAD
    # Configuration sidebar
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        target_margin = st.slider(
            "Target Profit Margin (%)",
            5, 30, 15,
            help="Expected profit margin target"
        ) / 100

        high_discount_thresh = st.slider(
            "High Discount Threshold (%)",
            10, 40, 20,
            help="Threshold for flagging high discounts"
        ) / 100

        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            3, 12, 6,
            help="Number of months to forecast"
        )

        anomaly_sensitivity = st.slider(
            "Anomaly Detection Sensitivity (%)",
            1, 20, 5,
            help="Expected percentage of outliers"
        ) / 100

<<<<<<< HEAD
    # Create configuration
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    config = RevenueLeakageConfig(
        target_margin=target_margin,
        high_discount_threshold=high_discount_thresh,
        forecast_horizon=forecast_horizon
    )

<<<<<<< HEAD
    # Initialize analyzer
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    try:
        analyzer = RevenueLeakageAnalyzer(df, config)
        analyzer.prepare_data()
        llm_generator = LLMInsightGenerator(llm_client)
    except ValueError as e:
        st.error(f"❌ Data validation error: {str(e)}")
        return

<<<<<<< HEAD
    # Create tabs
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    tabs = st.tabs([
        "📊 Executive Summary",
        "💰 Discount Analysis",
        "📉 Profit Erosion",
        "📦 Product Performance",
        "🌍 Regional Analysis",
        "🔮 Forecasting",
        "📈 Anomaly Detection",
        "📋 Recommendations"
    ])

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
    llm_gen: LLMInsightGenerator
) -> None:
 
    """Render executive summary tab."""
    st.subheader("📊 Executive Summary - Revenue Leakage Overview")

    # Compute metrics
    metrics = analyzer.compute_leakage_metrics()

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sales", f"${metrics['total_sales']:,.2f}")
        st.metric("Total Profit", f"${metrics['total_profit']:,.2f}")

    with col2:
        st.metric("Avg Profit Margin", f"{metrics['avg_profit_margin']:.2f}%")
        avg_discount = (
            analyzer.df['Discount'].mean() * 100
            if 'Discount' in analyzer.df.columns else 0
        )
        st.metric("Avg Discount", f"{avg_discount:.2f}%")

    with col3:
        st.metric(
            "Discount Leakage",
            f"${metrics['total_discount_leakage']:,.2f}"
        )
        st.metric(
            "Margin Leakage",
            f"${metrics['total_margin_leakage']:,.2f}"
        )

    with col4:
        st.metric(
            "Total Revenue Leakage",
            f"${metrics['total_leakage']:,.2f}",
            delta=f"-{metrics['avg_leakage_rate']:.2f}%",
            delta_color="inverse"
        )
        potential_savings = metrics['total_leakage'] * 0.7
        st.metric("Potential Savings (70%)", f"${potential_savings:,.2f}")

<<<<<<< HEAD
    # Visualization
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("Revenue Leakage Breakdown")

    leakage_data = pd.DataFrame({
        'Category': ['Discount Leakage', 'Margin Leakage', 'Potential Savings'],
        'Amount': [
            metrics['total_discount_leakage'],
            metrics['total_margin_leakage'],
            potential_savings
        ]
    })

    fig = px.bar(
        leakage_data,
        x='Category',
        y='Amount',
        title='Revenue Leakage Components',
        color='Category',
        color_discrete_map={
            'Discount Leakage': '#ff6b6b',
            'Margin Leakage': '#ee5a6f',
            'Potential Savings': '#4ecdc4'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
    # AI insights
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("🤖 AI-Generated Executive Insights")
    with st.spinner("Generating insights..."):
        insights = llm_gen.generate_executive_summary(metrics)
    st.write(insights)


def _render_discount_analysis(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator
) -> None:
<<<<<<< HEAD
    """Render discount analysis tab."""
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("💰 Discount-Driven Revenue Leakage Analysis")

    if 'Discount' not in analyzer.df.columns:
        st.warning("⚠️ No discount data available in the dataset")
        return

<<<<<<< HEAD
    # Discount distribution
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Discount Distribution")
        fig = px.histogram(
            analyzer.df,
            x='Discount',
            nbins=50,
            title='Distribution of Discount Rates',
            labels={'Discount': 'Discount Rate', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### High Discount Impact")
        high_discount = analyzer.df[
            analyzer.df['Discount'] > analyzer.config.high_discount_threshold
        ]
        high_discount_loss = high_discount['Discount_Leakage'].sum()

        st.metric(
            f"Transactions with >{analyzer.config.high_discount_threshold*100:.0f}% Discount",
            len(high_discount)
        )
        st.metric("Revenue Lost (High Discounts)", f"${high_discount_loss:,.2f}")

<<<<<<< HEAD
        # Category breakdown
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        if 'Category' in analyzer.df.columns:
            st.markdown("##### Discount by Category")
            discount_by_cat = analyzer.df.groupby('Category').agg({
                'Discount': 'mean',
                'Discount_Leakage': 'sum'
            }).sort_values('Discount_Leakage', ascending=False)

            st.dataframe(
                discount_by_cat.style.format({
                    'Discount': '{:.2%}',
                    'Discount_Leakage': '${:,.2f}'
                }),
                height=200
            )

<<<<<<< HEAD
    # Discount vs Profit
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.markdown("#### Discount Impact on Profitability")

    sample_size = min(1000, len(analyzer.df))
    sample_df = analyzer.df.sample(sample_size, random_state=42)

    fig = px.scatter(
        sample_df,
        x='Discount',
        y='Profit_Margin',
        color='Category' if 'Category' in sample_df.columns else None,
        title='Discount Rate vs Profit Margin',
        labels={'Discount': 'Discount Rate', 'Profit_Margin': 'Profit Margin (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
    # AI insights
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.markdown("#### 🤖 AI-Generated Discount Insights")

    discount_stats = {
        'mean': analyzer.df['Discount'].mean(),
        '50%': analyzer.df['Discount'].median(),
        'max': analyzer.df['Discount'].max(),
        'high_discount_count': len(high_discount),
        'high_discount_loss': high_discount_loss
    }

    category_stats = None
    if 'Category' in analyzer.df.columns:
        category_stats = analyzer.df.groupby('Category')['Discount_Leakage'].sum().sort_values(ascending=False)

    with st.spinner("Analyzing discount patterns..."):
        insights = llm_gen.generate_discount_insights(discount_stats, category_stats)
    st.write(insights)


def _render_profit_erosion(analyzer: RevenueLeakageAnalyzer) -> None:
<<<<<<< HEAD
    """Render profit erosion analysis tab."""
    st.subheader("📉 Profit Margin Erosion Analysis")

    # Time series if available
=======
    st.subheader("📉 Profit Margin Erosion Analysis")

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    if analyzer.date_column:
        try:
            ts_data = analyzer.build_time_series(level="overall")
            ts_data['Date'] = ts_data['YearMonth'].dt.to_timestamp()

<<<<<<< HEAD
            # Compute profit margin for time series
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            ts_data['Profit_Margin'] = np.where(
                ts_data['Sales'] > 0,
                (ts_data['Profit'] / ts_data['Sales']) * 100,
                0
            )

<<<<<<< HEAD
            # Plot trend
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_data['Date'],
                y=ts_data['Profit_Margin'],
                mode='lines+markers',
                name='Avg Profit Margin',
                line=dict(color='#4ecdc4', width=3)
            ))
            fig.update_layout(
                title='Profit Margin Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Profit Margin (%)'
            )
            st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
            # Trend analysis
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            if len(ts_data) > 2:
                from scipy import stats
                ts_data['Period_Num'] = range(len(ts_data))
                slope, _, _, _, _ = stats.linregress(
                    ts_data['Period_Num'],
                    ts_data['Profit_Margin']
                )

                trend_direction = "increasing ⬆️" if slope > 0 else "decreasing ⬇️"
                st.info(
                    f"📊 Profit margin trend: **{trend_direction}** "
                    f"at {abs(slope):.2f}% per month"
                )
        except Exception as e:
            st.warning(f"Could not generate time series: {str(e)}")

<<<<<<< HEAD
    # Distribution
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Profit Margin Distribution")
        fig = px.histogram(
            analyzer.df,
            x='Profit_Margin',
            nbins=50,
            title='Distribution of Profit Margins'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Low/Negative Margin Analysis")
        low_margin = analyzer.df[
            analyzer.df['Profit_Margin'] < analyzer.config.low_margin_threshold * 100
        ]
        negative_margin = analyzer.df[analyzer.df['Profit_Margin'] < 0]

        st.metric(
            f"Transactions with <{analyzer.config.low_margin_threshold*100:.0f}% Margin",
            len(low_margin)
        )
        st.metric("Negative Margin Transactions", len(negative_margin))
        st.metric(
            "Total Loss (Negative Margin)",
            f"${negative_margin['Profit'].sum():,.2f}" if len(negative_margin) > 0 else "$0.00"
        )

<<<<<<< HEAD
    # Category analysis
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    if 'Category' in analyzer.df.columns:
        st.markdown("#### Category-wise Profit Performance")

        cat_profit = analyzer.df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Profit_Margin': 'mean'
        }).sort_values('Profit_Margin')

        fig = px.bar(
            cat_profit.reset_index(),
            x='Category',
            y='Profit_Margin',
            title='Average Profit Margin by Category',
            color='Profit_Margin',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            cat_profit.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Profit_Margin': '{:.2f}%'
            })
        )


def _render_product_performance(analyzer: RevenueLeakageAnalyzer) -> None:
<<<<<<< HEAD
    """Render product performance analysis tab."""
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("📦 Product & Category Performance Analysis")

    if 'Category' not in analyzer.df.columns:
        st.warning("⚠️ No category data available")
        return

<<<<<<< HEAD
    # Category performance
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    cat_performance = analyzer.df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Discount': 'mean' if 'Discount' in analyzer.df.columns else lambda x: 0
    }).sort_values('Profit')

    cat_performance['Profit_Margin'] = np.where(
        cat_performance['Sales'] > 0,
        (cat_performance['Profit'] / cat_performance['Sales']) * 100,
        0
    )

<<<<<<< HEAD
    # Scatter plot
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    fig = px.scatter(
        cat_performance.reset_index(),
        x='Sales',
        y='Profit',
        size='Discount' if 'Discount' in analyzer.df.columns else None,
        color='Profit_Margin',
        hover_data=['Category'],
        title='Category Performance: Sales vs Profit',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
    # Top and bottom performers
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌟 Top Performing Categories")
        top_cats = cat_performance.nlargest(5, 'Profit')
        st.dataframe(
            top_cats.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.2%}',
                'Profit_Margin': '{:.2f}%'
            }).background_gradient(cmap='Greens', subset=['Profit'])
        )

    with col2:
        st.markdown("#### ⚠️ Bottom Performing Categories")
        bottom_cats = cat_performance.nsmallest(5, 'Profit')
        st.dataframe(
            bottom_cats.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.2%}',
                'Profit_Margin': '{:.2f}%'
            }).background_gradient(cmap='Reds', subset=['Profit'])
        )

<<<<<<< HEAD
    # Sub-category analysis
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    if 'Sub Category' in analyzer.df.columns:
        st.markdown("#### Sub-Category Deep Dive")

        subcat_performance = analyzer.df.groupby('Sub Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Discount': 'mean' if 'Discount' in analyzer.df.columns else lambda x: 0
        }).sort_values('Profit')

        subcat_performance['Profit_Margin'] = np.where(
            subcat_performance['Sales'] > 0,
            (subcat_performance['Profit'] / subcat_performance['Sales']) * 100,
            0
        )

<<<<<<< HEAD
        # Filter
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        filter_option = st.radio("Show:", ["All", "Profitable Only", "Loss-Making Only"])

        if filter_option == "Profitable Only":
            subcat_display = subcat_performance[subcat_performance['Profit'] > 0]
        elif filter_option == "Loss-Making Only":
            subcat_display = subcat_performance[subcat_performance['Profit'] < 0]
        else:
            subcat_display = subcat_performance

        st.dataframe(
            subcat_display.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.2%}',
                'Profit_Margin': '{:.2f}%'
            }).background_gradient(cmap='RdYlGn', subset=['Profit_Margin'])
        )


def _render_regional_analysis(analyzer: RevenueLeakageAnalyzer) -> None:
<<<<<<< HEAD
    """Render regional performance analysis tab."""
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("🌍 Regional Performance Analysis")

    region_col = None
    if 'Region' in analyzer.df.columns:
        region_col = 'Region'
    elif 'State' in analyzer.df.columns:
        region_col = 'State'

    if not region_col:
        st.warning("⚠️ No regional data available")
        return

<<<<<<< HEAD
    # Regional metrics
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    regional_metrics = analyzer.df.groupby(region_col).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Discount': 'mean' if 'Discount' in analyzer.df.columns else lambda x: 0
    })

    regional_metrics['Profit_Margin'] = np.where(
        regional_metrics['Sales'] > 0,
        (regional_metrics['Profit'] / regional_metrics['Sales']) * 100,
        0
    )
    regional_metrics = regional_metrics.sort_values('Profit', ascending=False)

<<<<<<< HEAD
    # Visualizations
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            regional_metrics.reset_index(),
            x=region_col,
            y='Sales',
            title=f'Sales by {region_col}',
            color='Profit_Margin',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            regional_metrics.reset_index(),
            values='Profit',
            names=region_col,
            title=f'Profit Distribution by {region_col}'
        )
        st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
    # Table
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.markdown(f"#### {region_col} Performance Table")
    st.dataframe(
        regional_metrics.style.format({
            'Sales': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Discount': '{:.2%}',
            'Profit_Margin': '{:.2f}%'
        }).background_gradient(cmap='RdYlGn', subset=['Profit_Margin'])
    )

<<<<<<< HEAD
    # Problem regions
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    avg_margin = regional_metrics['Profit_Margin'].mean()
    problem_regions = regional_metrics[regional_metrics['Profit_Margin'] < avg_margin]

    if len(problem_regions) > 0:
        st.warning(
            f"⚠️ {len(problem_regions)} regions performing below "
            f"average profit margin ({avg_margin:.2f}%)"
        )
        st.dataframe(
            problem_regions.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.2%}',
                'Profit_Margin': '{:.2f}%'
            })
        )


def _render_forecasting(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator,
    forecast_periods: int
) -> None:
<<<<<<< HEAD
    """Render forecasting analysis tab."""
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("🔮 Revenue Leakage Forecasting")

    if not analyzer.date_column:
        st.warning("⚠️ No date column found for time series forecasting")
        return

    try:
<<<<<<< HEAD
        # Generate forecast
        with st.spinner("Training forecasting model..."):
            forecast_df, metrics = analyzer.forecast_leakage(forecast_periods)

        # Visualize
        fig = go.Figure()

        # Historical
=======
        with st.spinner("Training forecasting model..."):
            forecast_df, metrics = analyzer.forecast_leakage(forecast_periods)

        fig = go.Figure()

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        historical = forecast_df.dropna(subset=['Historical_Leakage'])
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['Historical_Leakage'],
            mode='lines+markers',
            name='Historical Leakage',
            line=dict(color='#ff6b6b', width=2)
        ))

<<<<<<< HEAD
        # Forecast
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        forecast = forecast_df.dropna(subset=['Forecasted_Leakage'])
        fig.add_trace(go.Scatter(
            x=forecast['Date'],
            y=forecast['Forecasted_Leakage'],
            mode='lines+markers',
            name='Forecasted Leakage',
            line=dict(color='#ffd93d', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Revenue Leakage Forecast',
            xaxis_title='Date',
            yaxis_title='Total Leakage ($)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
        # Metrics
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Forecast Summary")
            st.metric(
                "Total Forecasted Leakage",
                f"${metrics['total_forecasted_leakage']:,.2f}"
            )
            st.metric(
                "Avg Monthly Forecast",
                f"${metrics['avg_monthly_forecast']:,.2f}"
            )
            st.metric(
                "Historical Avg",
                f"${metrics['historical_avg']:,.2f}"
            )

        with col2:
            st.markdown("#### Model Performance")
            if 'mae' in metrics:
                st.metric("MAE", f"${metrics['mae']:,.2f}")
            if 'rmse' in metrics:
                st.metric("RMSE", f"${metrics['rmse']:,.2f}")
            if 'mape' in metrics:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")

<<<<<<< HEAD
        # Forecast table
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        st.markdown("#### Detailed Forecast")
        forecast_only = forecast_df.dropna(subset=['Forecasted_Leakage'])
        st.dataframe(
            forecast_only[['Date', 'Forecasted_Leakage']].style.format({
                'Date': lambda x: x.strftime('%Y-%m'),
                'Forecasted_Leakage': '${:,.2f}'
            })
        )

<<<<<<< HEAD
        # AI insights
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
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
    contamination: float
) -> None:
<<<<<<< HEAD
    """Render anomaly detection tab."""
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    st.subheader("📈 Anomaly Detection for Revenue Leakage")

    try:
        with st.spinner("Detecting anomalies..."):
            anomalies_df, summary = analyzer.detect_anomalies(contamination)

<<<<<<< HEAD
        # Summary metrics
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Anomalous Transactions", summary['anomaly_count'])
            st.metric(
                "Anomaly Rate",
                f"{summary['anomaly_percentage']:.2f}%"
            )

        with col2:
            st.metric(
                "Sales at Risk",
                f"${summary['total_sales_at_risk']:,.2f}"
            )
            st.metric(
                "Leakage in Anomalies",
                f"${summary['total_leakage_in_anomalies']:,.2f}"
            )

        with col3:
            st.metric(
                "Avg Anomaly Score",
                f"{summary['avg_anomaly_score']:.2f}"
            )

<<<<<<< HEAD
        # Visualization
        if 'Sales' in anomalies_df.columns and 'Profit' in anomalies_df.columns:
            # Sample for performance
=======
        if 'Sales' in anomalies_df.columns and 'Profit' in anomalies_df.columns:
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            sample_size = min(2000, len(analyzer.df))
            plot_df = analyzer.df.sample(sample_size, random_state=42).copy()
            plot_df['Type'] = 'Normal'

<<<<<<< HEAD
            # Mark anomalies
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            anomaly_indices = set(anomalies_df.index)
            plot_df.loc[plot_df.index.isin(anomaly_indices), 'Type'] = 'Anomaly'

            fig = px.scatter(
                plot_df,
                x='Sales',
                y='Profit',
                color='Type',
                title='Anomaly Detection: Sales vs Profit',
                color_discrete_map={'Normal': '#4ecdc4', 'Anomaly': '#ff6b6b'},
                hover_data=['Discount'] if 'Discount' in plot_df.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

<<<<<<< HEAD
        # Anomaly details
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        if len(anomalies_df) > 0:
            st.markdown("#### Anomalous Transactions Details")

            display_cols = ['Sales', 'Profit', 'Profit_Margin',
<<<<<<< HEAD
                          'Total_Leakage', 'Anomaly_Score', 'Flags']
=======
                            'Total_Leakage', 'Anomaly_Score', 'Flags']
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)

            if 'Category' in anomalies_df.columns:
                display_cols.insert(0, 'Category')
            if 'Discount' in anomalies_df.columns:
                display_cols.insert(-2, 'Discount')

            display_cols = [col for col in display_cols if col in anomalies_df.columns]

<<<<<<< HEAD
            # Show top 50
            display_df = anomalies_df[display_cols].head(50)

            # Format
            format_dict = {}
=======
            display_df = anomalies_df[display_cols].head(50)

            format_dict: Dict[str, str] = {}
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            for col in display_cols:
                if col in ['Sales', 'Profit', 'Total_Leakage']:
                    format_dict[col] = '${:,.2f}'
                elif col == 'Discount':
                    format_dict[col] = '{:.2%}'
                elif col in ['Profit_Margin', 'Anomaly_Score']:
                    format_dict[col] = '{:.2f}'

            st.dataframe(display_df.style.format(format_dict))

<<<<<<< HEAD
            # Download
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
            csv = anomalies_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Anomalous Transactions (CSV)",
                data=csv,
                file_name="revenue_leakage_anomalies.csv",
                mime="text/csv"
            )
        else:
            st.info("✅ No significant anomalies detected")

    except ValueError as e:
        st.error(f"❌ Anomaly detection error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def _render_recommendations(
    analyzer: RevenueLeakageAnalyzer,
    llm_gen: LLMInsightGenerator
) -> None:
<<<<<<< HEAD
    """Render recommendations tab."""
    st.subheader("📋 AI-Powered Recommendations to Reduce Revenue Leakage")

    # Compile summary
    metrics = analyzer.compute_leakage_metrics()

    # Get worst performers
=======
    st.subheader("📋 AI-Powered Recommendations to Reduce Revenue Leakage")

    metrics = analyzer.compute_leakage_metrics()

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    top_categories = None
    if 'Category' in analyzer.df.columns:
        cat_profit = analyzer.df.groupby('Category')['Profit'].sum().sort_values()
        top_categories = cat_profit.head(3).index.tolist()

    top_regions = None
    region_col = 'Region' if 'Region' in analyzer.df.columns else (
        'State' if 'State' in analyzer.df.columns else None
    )
    if region_col:
        region_profit = analyzer.df.groupby(region_col)['Profit'].sum().sort_values()
        top_regions = region_profit.head(3).index.tolist()

<<<<<<< HEAD
    # Generate recommendations
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    with st.spinner("Generating comprehensive recommendations..."):
        recommendations = llm_gen.generate_recommendations(
            metrics,
            top_categories,
            top_regions
        )

    st.write(recommendations)

<<<<<<< HEAD
    # Export options
    st.markdown("---")
    st.markdown("#### 📥 Export Analysis Report")

    # Summary CSV
=======
    st.markdown("---")
    st.markdown("#### 📥 Export Analysis Report")

>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
    summary_data = {
        'Metric': [
            'Total Sales',
            'Total Profit',
            'Average Profit Margin',
            'Total Discount Leakage',
            'Total Margin Leakage',
            'Total Revenue Leakage',
            'Average Leakage Rate',
            'Negative Profit Transactions'
        ],
        'Value': [
            f"${metrics['total_sales']:,.2f}",
            f"${metrics['total_profit']:,.2f}",
            f"{metrics['avg_profit_margin']:.2f}%",
            f"${metrics['total_discount_leakage']:,.2f}",
            f"${metrics['total_margin_leakage']:,.2f}",
            f"${metrics['total_leakage']:,.2f}",
            f"{metrics['avg_leakage_rate']:.2f}%",
            str(metrics['negative_profit_count'])
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    csv = summary_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Summary Report (CSV)",
        data=csv,
        file_name="revenue_leakage_summary.csv",
        mime="text/csv"
    )


# ============================================================================
# Legacy Compatibility
# ============================================================================

class RevenueLeakageDetector:
    """
    Legacy compatibility wrapper.
<<<<<<< HEAD

    Maintains backward compatibility with old API while using new architecture.
    """

    def __init__(self, data: pd.DataFrame, llm: Any):
        """Initialize with data and LLM client."""
=======
    """

    def __init__(self, data: pd.DataFrame, llm: Any):
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        self.data = data
        self.llm = llm

    def detect_leakages(self) -> None:
<<<<<<< HEAD
        """Run the revenue leakage application."""
=======
>>>>>>> 374c3e0 (Update revenue leakage logic and UI components)
        run_revenue_leakage_app(self.data, self.llm)
