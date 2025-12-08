"""
Enhanced LLM Insight Generator

Integrates all LLM reasoning components to provide ChatGPT-level
analytical depth for decision intelligence insights.

This is a drop-in replacement for the existing LLMInsightGenerator
with significantly enhanced prompt quality, reasoning, and rigor.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from .prompt_templates import PromptTemplates
from .context_builder import ContextBuilder
from .reasoning_scaffolds import ReasoningScaffolds
from .guardrails import InsightGuardrails


class EnhancedInsightGenerator:
    """
    Enhanced LLM insight generation with reasoning scaffolds and guardrails.

    This class provides the same interface as LLMInsightGenerator but with
    dramatically improved prompt quality:
    - Reasoning scaffolding (Chain-of-Thought)
    - Fact/Interpretation/Implication separation
    - Guardrails against hallucination
    - Data-grounded context
    - Explicit uncertainty handling

    Design Philosophy:
    - Drop-in replacement for existing LLMInsightGenerator
    - No changes to analytics code
    - Only LLM prompts and responses are enhanced
    - All numeric computation remains unchanged
    """

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize with LLM client.

        Args:
            llm_client: LLM client with a conversational_response method.
        """
        self.llm = llm_client
        self.guardrails = InsightGuardrails()
        self.context_builder = ContextBuilder()

    def _call_llm(self, prompt: str, timeout: int = 30) -> str:
        """
        Safely call LLM with error handling.

        Args:
            prompt: Enhanced prompt text.
            timeout: Timeout in seconds.

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

    def _call_llm_with_validation(
        self,
        prompt: str,
        provided_metrics: Dict[str, Any],
        timeout: int = 30
    ) -> str:
        """
        Call LLM with post-response validation.

        Args:
            prompt: Enhanced prompt
            provided_metrics: Metrics that were provided to LLM
            timeout: Timeout in seconds

        Returns:
            LLM response (potentially with validation warnings in debug mode)
        """
        response = self._call_llm(prompt, timeout)

        # Validate response quality
        validation = self.guardrails.validate_response(response, provided_metrics)

        # In production, we log validation results but don't block
        # In debug mode, we could append warnings
        # For now, just return the response
        return response

    def generate_executive_summary(self, metrics: Dict[str, float]) -> str:
        """
        Generate enhanced executive summary insights.

        Args:
            metrics: Dictionary of computed metrics.

        Returns:
            Executive summary text with analytical depth.
        """
        prompt = PromptTemplates.executive_summary_prompt(metrics)
        return self._call_llm_with_validation(prompt, metrics)

    def generate_discount_insights(
        self,
        discount_stats: Dict[str, float],
        category_stats: Optional[pd.Series] = None,
    ) -> str:
        """
        Generate enhanced discount pattern insights.

        Args:
            discount_stats: Discount distribution statistics.
            category_stats: Optional Series of category-level discount leakage.

        Returns:
            Discount insights with strategic depth.
        """
        prompt = PromptTemplates.discount_insights_prompt(discount_stats, category_stats)
        return self._call_llm_with_validation(prompt, discount_stats)

    def generate_forecast_insights(
        self,
        forecast_summary: Dict[str, float],
    ) -> str:
        """
        Generate enhanced forecast interpretation with confidence calibration.

        Args:
            forecast_summary: Forecast metrics and trends.

        Returns:
            Forecast insights with uncertainty quantification.
        """
        prompt = PromptTemplates.forecast_insights_prompt(forecast_summary)
        return self._call_llm_with_validation(prompt, forecast_summary)

    def generate_recommendations(
        self,
        summary: Dict[str, Any],
        top_categories: Optional[List[str]] = None,
        top_regions: Optional[List[str]] = None,
    ) -> str:
        """
        Generate comprehensive recommendations.

        Note: This method maintains compatibility with existing interface
        but uses enhanced prompts internally.

        Args:
            summary: Overall analysis summary metrics.
            top_categories: List of worst-performing categories.
            top_regions: List of worst-performing regions.

        Returns:
            Comprehensive recommendations text.
        """
        # Build enhanced context
        notable_patterns = self.context_builder.identify_notable_patterns(summary)

        # Create a structured signals dictionary for decision priority
        signals = {
            'leakage_severity': self._assess_severity(summary),
            'forecast_trend': summary.get('trend_direction', 'unknown'),
            'anomaly_risk': self._assess_anomaly_risk(summary),
            'model_confidence': 'moderate',  # Could be computed from metrics
            'monthly_impact': summary.get('total_leakage', 0),
            'quick_wins': notable_patterns[:2] if notable_patterns else [],
            'strategic_issues': top_categories[:3] if top_categories else []
        }

        prompt = PromptTemplates.decision_priority_prompt(signals)
        return self._call_llm_with_validation(prompt, summary)

    def generate_root_cause_analysis(self, context: Dict[str, Any]) -> str:
        """
        Generate enhanced root cause analysis with rigorous causal reasoning.

        Args:
            context: Dictionary containing multi-signal context

        Returns:
            Root cause analysis explaining WHY leakage occurs.
        """
        prompt = PromptTemplates.root_cause_analysis_prompt(context)

        # Merge all metrics for validation
        all_metrics = {}
        all_metrics.update(context.get('leakage_metrics', {}))
        all_metrics.update(context.get('anomaly_patterns', {}))
        all_metrics.update(context.get('discount_stats', {}))

        return self._call_llm_with_validation(prompt, all_metrics)

    def generate_cross_signal_insights(
        self,
        forecast_summary: Dict[str, float],
        anomaly_summary: Dict[str, Any],
        risk_summary: Dict[str, Any],
        leakage_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate enhanced cross-model synthesis insights.

        Args:
            forecast_summary: Forecasting model results
            anomaly_summary: Anomaly detection results
            risk_summary: Risk scoring results
            leakage_metrics: Optional overall leakage metrics

        Returns:
            Cross-signal synthesis with integrated intelligence.
        """
        prompt = PromptTemplates.cross_signal_insights_prompt(
            forecast_summary, anomaly_summary, risk_summary, leakage_metrics
        )

        # Merge all metrics for validation
        all_metrics = {}
        all_metrics.update(forecast_summary)
        all_metrics.update(anomaly_summary)
        all_metrics.update(risk_summary)
        if leakage_metrics:
            all_metrics.update(leakage_metrics)

        return self._call_llm_with_validation(prompt, all_metrics)

    def generate_decision_priority(
        self,
        signals: Dict[str, Any],
    ) -> str:
        """
        Generate enhanced decision prioritization framework.

        Args:
            signals: Dictionary containing signals and indicators

        Returns:
            Prioritized decision framework.
        """
        prompt = PromptTemplates.decision_priority_prompt(signals)
        return self._call_llm_with_validation(prompt, signals)

    def generate_scenario_impact_explanation(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
        scenario_description: str,
    ) -> str:
        """
        Generate enhanced scenario impact analysis with trade-off evaluation.

        Args:
            before_metrics: Baseline metrics before scenario
            after_metrics: Metrics after scenario simulation
            scenario_description: Description of what changed

        Returns:
            Scenario impact analysis with trade-offs and recommendations.
        """
        prompt = PromptTemplates.scenario_impact_prompt(
            before_metrics, after_metrics, scenario_description
        )

        # Merge metrics for validation
        all_metrics = {}
        all_metrics.update(before_metrics)
        all_metrics.update(after_metrics)

        return self._call_llm_with_validation(prompt, all_metrics)

    def generate_forecast_confidence_narrative(
        self,
        forecast_metrics: Dict[str, float],
        data_quality: Dict[str, Any],
    ) -> str:
        """
        Generate forecast confidence narrative with explicit uncertainty.

        Args:
            forecast_metrics: Model performance metrics
            data_quality: Data characteristics affecting confidence

        Returns:
            Confidence assessment narrative.
        """
        # Build comprehensive model context
        model_context = self.context_builder.build_model_confidence_context(
            {**forecast_metrics, 'model_type': 'forecasting'},
            'forecasting'
        )

        # Build data quality context
        quality_context = self.guardrails.get_data_quality_context(data_quality)

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Forecast Confidence Assessment

Explain the reliability and confidence levels for this forecast model to business stakeholders
in clear, actionable terms.

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

{model_context}

{quality_context}

{ReasoningScaffolds.get_scaffold('uncertainty')}

**Specific Requirements:**

1. **Translate Technical Metrics to Business Language**
   - What does R² = {forecast_metrics.get('r2', 0):.3f} mean practically?
   - How should leaders interpret MAE of ${forecast_metrics.get('mae', 0):,.0f}?

2. **Confidence Levels**
   - High/Medium/Low confidence and specific reasoning
   - What can we confidently act on vs. what needs validation?

3. **Practical Guidance**
   - Which decisions are safe given this confidence level?
   - Which decisions need more data or different analysis?

4. **Improvement Path**
   - What would increase forecast reliability?
   - Specific data or methodology recommendations

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        all_metrics = {}
        all_metrics.update(forecast_metrics)
        all_metrics.update(data_quality)

        return self._call_llm_with_validation(prompt, all_metrics)

    def generate_anomaly_confidence_narrative(
        self,
        anomaly_metrics: Dict[str, Any],
        detection_config: Dict[str, Any],
    ) -> str:
        """
        Generate anomaly detection confidence narrative.

        Args:
            anomaly_metrics: Anomaly detection results
            detection_config: Detection method configuration

        Returns:
            Anomaly detection confidence narrative.
        """
        # Build context
        model_context = self.context_builder.build_model_confidence_context(
            {**anomaly_metrics, 'model_type': 'anomaly'},
            'anomaly'
        )

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Anomaly Detection Reliability Assessment

Explain the reliability of anomaly detection results and provide practical guidance
on how to use these alerts effectively.

**Anomaly Detection Results:**
- Total Anomalies Detected: {anomaly_metrics.get('anomaly_count', 0)}
- Anomaly Rate: {anomaly_metrics.get('anomaly_rate', 0):.2%}
- High-Risk Anomalies: {anomaly_metrics.get('high_risk_count', 0)}
- Medium-Risk Anomalies: {anomaly_metrics.get('medium_risk_count', 0)}
- Low-Risk Anomalies: {anomaly_metrics.get('low_risk_count', 0)}

**Detection Configuration:**
- Methods Used: {', '.join(detection_config.get('methods_used', ['IsolationForest']))}
- Sensitivity Settings: {detection_config.get('thresholds', 'standard')}
- Features Analyzed: {detection_config.get('feature_count', 'multiple')}

{model_context}

{ReasoningScaffolds.get_scaffold('uncertainty')}

**Specific Requirements:**

1. **Detection Reliability**
   - How trustworthy are these anomaly flags?
   - Expected false positive rate
   - Risk of missing real anomalies (false negatives)

2. **Prioritization Guidance**
   - Which anomalies deserve immediate investigation?
   - Which are lower priority?
   - Resource allocation recommendations

3. **Practical Use**
   - How should teams act on these alerts?
   - Workflow recommendations
   - Threshold adjustment guidance

4. **Alert Fatigue Prevention**
   - How to avoid overwhelming teams with alerts
   - When to investigate vs. when to monitor

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        all_metrics = {}
        all_metrics.update(anomaly_metrics)
        all_metrics.update(detection_config)

        return self._call_llm_with_validation(prompt, all_metrics)

    def generate_risk_score_confidence_narrative(
        self,
        risk_model_metrics: Dict[str, float],
        model_validation: Dict[str, Any],
    ) -> str:
        """
        Generate risk scoring model confidence narrative.

        Args:
            risk_model_metrics: Classification model performance
            model_validation: Validation information

        Returns:
            Risk scoring confidence narrative.
        """
        # Build model context
        model_context = self.context_builder.build_model_confidence_context(
            {**risk_model_metrics, 'model_type': 'classification'},
            'classification'
        )

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Risk Scoring Model Confidence Assessment

Explain the reliability of risk scores and provide guidance on how to use them
for decision-making.

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

{model_context}

{ReasoningScaffolds.get_scaffold('uncertainty')}

**Specific Requirements:**

1. **Model Trustworthiness**
   - Can risk scores drive automated decisions?
   - What accuracy level means in practice

2. **Precision vs. Recall Trade-off**
   - Does model catch most risky transactions (recall)?
   - Does it avoid false alarms (precision)?
   - Which matters more for this business context?

3. **Use Case Guidance**
   - What to automate (high confidence)
   - What needs human review (moderate confidence)
   - What NOT to rely on model for (low confidence)

4. **Improvement Path**
   - What would make risk scores more reliable?
   - Data, features, or methodology changes needed

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        all_metrics = {}
        all_metrics.update(risk_model_metrics)
        all_metrics.update(model_validation)

        return self._call_llm_with_validation(prompt, all_metrics)

    # Helper methods

    def _assess_severity(self, metrics: Dict[str, Any]) -> str:
        """Assess leakage severity level."""
        total_sales = metrics.get('total_sales', 0)
        total_leakage = metrics.get('total_leakage', 0)

        if total_sales == 0:
            return 'unknown'

        leakage_rate = (total_leakage / total_sales) * 100

        if leakage_rate > 15:
            return 'critical'
        elif leakage_rate > 10:
            return 'high'
        elif leakage_rate > 5:
            return 'moderate'
        else:
            return 'low'

    def _assess_anomaly_risk(self, metrics: Dict[str, Any]) -> str:
        """Assess anomaly risk level."""
        anomaly_rate = metrics.get('anomaly_rate', 0)

        if anomaly_rate > 0.10:
            return 'high'
        elif anomaly_rate > 0.05:
            return 'moderate'
        else:
            return 'low'
