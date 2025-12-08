"""
Context Builder for Data-Grounded Insights

Assembles rich, structured context from metrics and data patterns
to provide LLMs with the information needed for high-quality insights.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class ContextBuilder:
    """
    Builds structured, data-grounded context for LLM prompts.

    Instead of just listing metrics, this class:
    - Provides comparative context (vs. benchmarks, historical)
    - Identifies notable patterns
    - Highlights anomalies and outliers
    - Creates relational context (how metrics connect)
    """

    @staticmethod
    def build_metric_context(
        metrics: Dict[str, Any],
        metric_name: str,
        format_type: str = 'currency'
    ) -> str:
        """
        Build rich context for a single metric.

        Args:
            metrics: Dictionary of all metrics
            metric_name: Specific metric to contextualize
            format_type: How to format (currency, percentage, number)

        Returns:
            Formatted metric with context
        """
        value = metrics.get(metric_name, 0)

        # Format value based on type
        if format_type == 'currency':
            formatted = f"${value:,.2f}"
        elif format_type == 'percentage':
            formatted = f"{value:.2f}%"
        else:
            formatted = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)

        # Try to provide comparative context
        context_parts = [formatted]

        # Check for related metrics that provide context
        if metric_name.endswith('_current'):
            historical_key = metric_name.replace('_current', '_historical')
            if historical_key in metrics:
                hist_value = metrics[historical_key]
                if hist_value != 0:
                    change_pct = ((value - hist_value) / hist_value) * 100
                    direction = "↑" if change_pct > 0 else "↓"
                    context_parts.append(
                        f"({direction} {abs(change_pct):.1f}% vs. historical)"
                    )

        return " ".join(context_parts)

    @staticmethod
    def build_comparative_context(
        metrics: Dict[str, Any],
        comparisons: Optional[List[tuple]] = None
    ) -> str:
        """
        Build comparative context showing relationships between metrics.

        Args:
            metrics: Dictionary of metrics
            comparisons: Optional list of (metric1, metric2, comparison_type) tuples

        Returns:
            Comparative context string
        """
        context = []

        # Auto-detect key comparisons
        auto_comparisons = []

        # Sales vs Profit
        if 'total_sales' in metrics and 'total_profit' in metrics:
            sales = metrics['total_sales']
            profit = metrics['total_profit']
            if sales > 0:
                profit_margin = (profit / sales) * 100
                context.append(
                    f"- Profit represents {profit_margin:.1f}% of sales "
                    f"(${profit:,.0f} profit on ${sales:,.0f} sales)"
                )

        # Leakage vs Sales
        if 'total_leakage' in metrics and 'total_sales' in metrics:
            leakage = metrics['total_leakage']
            sales = metrics['total_sales']
            if sales > 0:
                leakage_pct = (leakage / sales) * 100
                severity = (
                    "critical" if leakage_pct > 10
                    else "concerning" if leakage_pct > 5
                    else "moderate"
                )
                context.append(
                    f"- Leakage represents {leakage_pct:.1f}% of total sales ({severity})"
                )

        # Anomaly rate
        if 'anomaly_count' in metrics and 'total_records' in metrics:
            anomaly_count = metrics['anomaly_count']
            total = metrics['total_records']
            if total > 0:
                anomaly_rate = (anomaly_count / total) * 100
                assessment = (
                    "very high (investigate immediately)" if anomaly_rate > 10
                    else "high (requires attention)" if anomaly_rate > 5
                    else "moderate" if anomaly_rate > 1
                    else "low (within normal range)"
                )
                context.append(
                    f"- Anomaly rate: {anomaly_rate:.2f}% ({anomaly_count} of {total} records) - {assessment}"
                )

        # Forecast trend
        if 'avg_monthly_forecast' in metrics and 'historical_avg' in metrics:
            forecast = metrics['avg_monthly_forecast']
            historical = metrics['historical_avg']
            if historical > 0:
                change = ((forecast - historical) / historical) * 100
                direction = "deteriorating" if change > 0 else "improving"
                context.append(
                    f"- Forecast trend: {direction} by {abs(change):.1f}% "
                    f"(${forecast:,.0f} forecasted vs ${historical:,.0f} historical average)"
                )

        if context:
            return "**Comparative Context:**\n" + "\n".join(context)
        else:
            return ""

    @staticmethod
    def build_distribution_context(
        data: Optional[pd.DataFrame],
        column: str,
        metric_type: str = 'numerical'
    ) -> str:
        """
        Build context about data distribution.

        Args:
            data: DataFrame containing raw data
            column: Column to analyze
            metric_type: Type of data (numerical, categorical)

        Returns:
            Distribution context string
        """
        if data is None or column not in data.columns:
            return ""

        context = []

        if metric_type == 'numerical':
            # Numerical distribution
            values = data[column].dropna()
            if len(values) > 0:
                q25, q50, q75 = values.quantile([0.25, 0.5, 0.75])
                mean = values.mean()
                std = values.std()

                context.append(f"**{column} Distribution:**")
                context.append(f"- Median: {q50:,.2f}, Mean: {mean:,.2f}")
                context.append(f"- Range: {values.min():,.2f} to {values.max():,.2f}")
                context.append(f"- Q1-Q3: {q25:,.2f} to {q75:,.2f}")

                # Check for skewness
                if mean > q50 * 1.2:
                    context.append(f"- Distribution: Right-skewed (few high values pull mean up)")
                elif mean < q50 * 0.8:
                    context.append(f"- Distribution: Left-skewed")

                # Check for outliers
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                outliers = values[values > outlier_threshold]
                if len(outliers) > 0:
                    context.append(
                        f"- Outliers: {len(outliers)} values above {outlier_threshold:,.2f} "
                        f"({len(outliers)/len(values)*100:.1f}% of data)"
                    )

        elif metric_type == 'categorical':
            # Categorical distribution
            value_counts = data[column].value_counts()
            total = len(data[column].dropna())

            context.append(f"**{column} Distribution:**")
            context.append(f"- {len(value_counts)} unique values")

            # Show top categories
            top_n = 5
            for i, (category, count) in enumerate(value_counts.head(top_n).items(), 1):
                pct = (count / total) * 100
                context.append(f"  {i}. {category}: {count} ({pct:.1f}%)")

            # Check concentration
            top1_pct = (value_counts.iloc[0] / total) * 100 if len(value_counts) > 0 else 0
            if top1_pct > 50:
                context.append(f"- Highly concentrated: Top category represents {top1_pct:.1f}% of data")
            elif top1_pct < 10:
                context.append(f"- Well distributed: No dominant category")

        return "\n".join(context) if context else ""

    @staticmethod
    def build_temporal_context(
        metrics: Dict[str, Any],
        temporal_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Build temporal/time-based context.

        Args:
            metrics: Dictionary of metrics
            temporal_data: Optional time series data

        Returns:
            Temporal context string
        """
        context = []

        # Check for trend information in metrics
        if 'trend_direction' in metrics:
            direction = metrics['trend_direction']
            context.append(f"- Overall trend: {direction}")

        if 'volatility_level' in metrics:
            volatility = metrics['volatility_level']
            context.append(f"- Volatility: {volatility}")

        if 'has_seasonality' in metrics:
            seasonality = metrics['has_seasonality']
            if seasonality:
                context.append("- Seasonal patterns detected")

        # Recent vs historical
        if 'recent_avg' in metrics and 'historical_avg' in metrics:
            recent = metrics['recent_avg']
            historical = metrics['historical_avg']
            if historical != 0:
                change = ((recent - historical) / historical) * 100
                direction = "increased" if change > 0 else "decreased"
                context.append(
                    f"- Recent trend: {direction} by {abs(change):.1f}% "
                    f"(${recent:,.0f} recent vs ${historical:,.0f} historical)"
                )

        if context:
            return "**Temporal Patterns:**\n" + "\n".join(context)
        else:
            return ""

    @staticmethod
    def build_model_confidence_context(
        model_metrics: Dict[str, float],
        model_type: str = 'forecasting'
    ) -> str:
        """
        Build context about model reliability and confidence.

        Args:
            model_metrics: Model performance metrics
            model_type: Type of model (forecasting, classification, anomaly)

        Returns:
            Model confidence context string
        """
        context = []

        if model_type == 'forecasting':
            mae = model_metrics.get('mae', 0)
            rmse = model_metrics.get('rmse', 0)
            r2 = model_metrics.get('r2', 0)
            mape = model_metrics.get('mape', 0)

            context.append("**Model Reliability:**")

            # R² interpretation
            if r2 >= 0.8:
                context.append(f"- R² = {r2:.3f}: High explanatory power (explains {r2*100:.1f}% of variance)")
            elif r2 >= 0.5:
                context.append(f"- R² = {r2:.3f}: Moderate explanatory power")
            else:
                context.append(f"- R² = {r2:.3f}: Low explanatory power (predictions have high uncertainty)")

            # MAPE interpretation
            if mape > 0:
                if mape < 10:
                    context.append(f"- MAPE = {mape:.1f}%: Highly accurate forecasts")
                elif mape < 20:
                    context.append(f"- MAPE = {mape:.1f}%: Reasonably accurate forecasts")
                else:
                    context.append(f"- MAPE = {mape:.1f}%: High forecast error (use cautiously)")

            # MAE/RMSE
            if mae > 0:
                context.append(f"- Average error magnitude: ${mae:,.0f}")

        elif model_type == 'classification':
            accuracy = model_metrics.get('accuracy', 0)
            precision = model_metrics.get('precision', 0)
            recall = model_metrics.get('recall', 0)
            roc_auc = model_metrics.get('roc_auc', 0)

            context.append("**Model Reliability:**")

            # Overall performance
            if accuracy >= 0.85 and roc_auc >= 0.85:
                context.append(f"- Strong model performance (Accuracy: {accuracy:.1%}, ROC-AUC: {roc_auc:.3f})")
            elif accuracy >= 0.70:
                context.append(f"- Moderate model performance (Accuracy: {accuracy:.1%}, ROC-AUC: {roc_auc:.3f})")
            else:
                context.append(f"- Limited model performance (Accuracy: {accuracy:.1%}) - interpret cautiously")

            # Precision vs Recall trade-off
            if precision > 0 and recall > 0:
                if precision >= 0.8 and recall >= 0.8:
                    context.append("- Balanced performance: Good at both identifying risks and avoiding false alarms")
                elif precision > recall + 0.2:
                    context.append(f"- High precision ({precision:.1%}) but lower recall ({recall:.1%}): Conservative, may miss some risks")
                elif recall > precision + 0.2:
                    context.append(f"- High recall ({recall:.1%}) but lower precision ({precision:.1%}): Sensitive, may have false alarms")

        elif model_type == 'anomaly':
            anomaly_rate = model_metrics.get('anomaly_rate', 0)
            contamination = model_metrics.get('contamination', 0.1)

            context.append("**Detection Reliability:**")

            if anomaly_rate < contamination * 0.5:
                context.append(f"- Low anomaly rate ({anomaly_rate:.1%}): Either clean data or underdetection")
            elif anomaly_rate > contamination * 2:
                context.append(f"- High anomaly rate ({anomaly_rate:.1%}): Many irregularities or overdetection")
            else:
                context.append(f"- Normal anomaly rate ({anomaly_rate:.1%}): Reasonable detection level")

        return "\n".join(context) if context else ""

    @staticmethod
    def build_comprehensive_context(
        metrics: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        model_metrics: Optional[Dict[str, Any]] = None,
        include_sections: Optional[List[str]] = None
    ) -> str:
        """
        Build comprehensive context combining multiple context types.

        Args:
            metrics: Primary metrics dictionary
            data: Optional raw data for distribution analysis
            model_metrics: Optional model performance metrics
            include_sections: Which sections to include (default: all)

        Returns:
            Comprehensive context string
        """
        all_sections = ['comparative', 'temporal', 'model', 'distribution']
        sections = include_sections if include_sections else all_sections

        context_parts = []

        if 'comparative' in sections:
            comp_context = ContextBuilder.build_comparative_context(metrics)
            if comp_context:
                context_parts.append(comp_context)

        if 'temporal' in sections:
            temp_context = ContextBuilder.build_temporal_context(metrics)
            if temp_context:
                context_parts.append(temp_context)

        if 'model' in sections and model_metrics:
            model_type = model_metrics.get('model_type', 'forecasting')
            model_context = ContextBuilder.build_model_confidence_context(
                model_metrics, model_type
            )
            if model_context:
                context_parts.append(model_context)

        return "\n\n".join(context_parts) if context_parts else ""

    @staticmethod
    def identify_notable_patterns(metrics: Dict[str, Any]) -> List[str]:
        """
        Identify notable patterns that LLM should focus on.

        Args:
            metrics: Metrics dictionary

        Returns:
            List of notable patterns to highlight
        """
        patterns = []

        # Check for severe issues
        if metrics.get('negative_profit_count', 0) > 10:
            count = metrics['negative_profit_count']
            patterns.append(
                f"⚠️ CRITICAL: {count} transactions with negative profit "
                "(immediate attention required)"
            )

        # Check for high leakage rate
        total_sales = metrics.get('total_sales', 0)
        total_leakage = metrics.get('total_leakage', 0)
        if total_sales > 0 and total_leakage > 0:
            leakage_rate = (total_leakage / total_sales) * 100
            if leakage_rate > 15:
                patterns.append(
                    f"⚠️ HIGH: Leakage at {leakage_rate:.1f}% of sales "
                    "(typically should be <5%)"
                )

        # Check for deteriorating trends
        if metrics.get('trend_direction') == 'increasing' and 'avg_monthly_forecast' in metrics:
            patterns.append(
                "⚠️ TREND: Leakage forecasted to increase "
                "(proactive intervention needed)"
            )

        # Check for high anomaly concentration
        anomaly_rate = metrics.get('anomaly_rate', 0)
        if anomaly_rate > 0.10:
            patterns.append(
                f"⚠️ ANOMALY: {anomaly_rate:.1%} of transactions flagged as anomalous "
                "(investigate root causes)"
            )

        # Check for concerning discount patterns
        high_discount_count = metrics.get('high_discount_count', 0)
        total_count = metrics.get('total_count', 1)
        if high_discount_count > 0 and total_count > 0:
            high_disc_rate = (high_discount_count / total_count) * 100
            if high_disc_rate > 20:
                patterns.append(
                    f"💡 PATTERN: {high_disc_rate:.1f}% of transactions have high discounts (>20%) "
                    "(review discount policy)"
                )

        return patterns
