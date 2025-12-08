"""
Advanced Prompt Templates for LLM Insight Generation

Provides ChatGPT-level prompt templates with:
- Reasoning scaffolding
- Fact/Interpretation/Implication separation
- Guardrails against hallucination
- Data-grounded context
- Uncertainty handling
"""

from typing import Dict, Any, Optional, List
from .reasoning_scaffolds import ReasoningScaffolds
from .guardrails import InsightGuardrails
from .context_builder import ContextBuilder


class PromptTemplates:
    """
    Advanced prompt templates for high-quality analytical insights.

    These templates incorporate:
    - Specific analyst personas (not generic "business analyst")
    - Reasoning scaffolds for structured thinking
    - Explicit guardrails against hallucination
    - Rich, data-grounded context
    - Clear output structure requirements
    """

    @staticmethod
    def executive_summary_prompt(metrics: Dict[str, float]) -> str:
        """
        Enhanced executive summary prompt with analytical rigor.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Enhanced prompt string
        """
        # Build context
        comparative_context = ContextBuilder.build_comparative_context(metrics)
        notable_patterns = ContextBuilder.identify_notable_patterns(metrics)

        patterns_section = ""
        if notable_patterns:
            patterns_section = "\n**Notable Patterns Detected:**\n" + "\n".join(
                f"- {pattern}" for pattern in notable_patterns
            )

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Executive Summary of Revenue Leakage Analysis

Generate a concise but analytically rigorous executive summary of this revenue leakage analysis.

**Key Metrics:**
- Total Sales: ${metrics.get('total_sales', 0):,.2f}
- Total Profit: ${metrics.get('total_profit', 0):,.2f}
- Average Profit Margin: {metrics.get('avg_profit_margin', 0):.2f}%
- Total Revenue Leakage: ${metrics.get('total_leakage', 0):,.2f}
- Discount Leakage: ${metrics.get('total_discount_leakage', 0):,.2f}
- Margin Leakage: ${metrics.get('total_margin_leakage', 0):,.2f}
- Negative Profit Transactions: {metrics.get('negative_profit_count', 0)}

{comparative_context}

{patterns_section}

{ReasoningScaffolds.fact_interpretation_implication_scaffold()}

**Specific Requirements:**
1. Lead with the single most critical finding (what needs immediate attention)
2. Quantify impact precisely (use exact $ and % from metrics above)
3. Distinguish between structural issues (long-term) vs tactical issues (short-term)
4. State confidence level: What are we certain about vs. what needs investigation?
5. Be specific about implications: What decisions should change based on this analysis?

**Format:**
Keep response under 200 words but make every word count. Use bullet points for clarity.
Start with the most urgent finding, not a generic overview.

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt

    @staticmethod
    def discount_insights_prompt(
        discount_stats: Dict[str, float],
        category_stats: Optional[Any] = None,
    ) -> str:
        """
        Enhanced discount pattern analysis prompt.

        Args:
            discount_stats: Discount distribution statistics
            category_stats: Optional category-level discount leakage

        Returns:
            Enhanced prompt string
        """
        category_info = ""
        if category_stats is not None and not category_stats.empty:
            top_3 = category_stats.head(3)
            category_info = "\n**Top 3 Categories by Discount Leakage:**\n"
            for category, leakage in top_3.items():
                category_info += f"- {category}: ${leakage:,.2f}\n"

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Discount Pattern Analysis and Strategic Assessment

Analyze discount patterns to identify whether discounting is strategic (driving value) or
problematic (destroying value without corresponding benefits).

**Discount Statistics:**
- Average Discount Rate: {discount_stats.get('mean', 0):.2%}
- Median Discount: {discount_stats.get('50%', 0):.2%}
- Max Discount: {discount_stats.get('max', 0):.2%}
- 75th Percentile: {discount_stats.get('75%', 0):.2%}
- High Discount (>20%) Transactions: {discount_stats.get('high_discount_count', 0)}
- Revenue Lost to High Discounts: ${discount_stats.get('high_discount_loss', 0):,.2f}

{category_info}

{ReasoningScaffolds.chain_of_thought_scaffold()}

**Specific Analysis Requirements:**

1. **Discount Sustainability Assessment**
   - Is the average {discount_stats.get('mean', 0):.1%} discount rate sustainable?
   - Compare to typical healthy ranges (5-15% for most B2B/retail contexts)
   - What does the distribution (mean vs median) tell us about discount practices?

2. **Problem Identification**
   - Are discounts concentrated (few large discounts) or widespread?
   - Which specific categories show problematic discount patterns? Be specific.
   - Is discount loss proportional to sales volume or concentrated in low-volume high-discount deals?

3. **Strategic Recommendations**
   - Specific discount policy recommendations (not "optimize discounts" but exact thresholds/rules)
   - Which discounts to eliminate vs. which to keep
   - Expected $ impact of recommendations

4. **Confidence & Validation**
   - What assumptions underpin these recommendations?
   - What data would strengthen confidence in these recommendations?

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt

    @staticmethod
    def forecast_insights_prompt(
        forecast_summary: Dict[str, float],
    ) -> str:
        """
        Enhanced forecast interpretation prompt with confidence calibration.

        Args:
            forecast_summary: Forecast metrics and trends

        Returns:
            Enhanced prompt string
        """
        trend = (
            "increasing"
            if forecast_summary.get("avg_monthly_forecast", 0)
            > forecast_summary.get("historical_avg", 0)
            else "decreasing"
        )

        model_metrics = {
            'mae': forecast_summary.get('mae', 0),
            'rmse': forecast_summary.get('rmse', 0),
            'r2': forecast_summary.get('r2', 0),
            'mape': forecast_summary.get('mape', 0),
            'model_type': 'forecasting'
        }

        model_context = ContextBuilder.build_model_confidence_context(
            model_metrics, 'forecasting'
        )

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Forecast Interpretation with Confidence Calibration

Interpret the revenue leakage forecast, clearly separating what the model predicts from
how confident we should be in those predictions.

**Forecast Summary:**
- Historical Average Monthly Leakage: ${forecast_summary.get('historical_avg', 0):,.2f}
- Forecasted Average Monthly Leakage: ${forecast_summary.get('avg_monthly_forecast', 0):,.2f}
- Total Forecasted Leakage (next periods): ${forecast_summary.get('total_forecasted_leakage', 0):,.2f}
- Trend Direction: {trend}
- Model Error (MAE): ${forecast_summary.get('mae', 0):,.2f}
- Model Error (RMSE): ${forecast_summary.get('rmse', 0):,.2f}
- R² Score: {forecast_summary.get('r2', 0):.3f}

{model_context}

{ReasoningScaffolds.combine_scaffolds(['fii', 'uncertainty'])}

**Specific Analysis Requirements:**

1. **Trend Interpretation**
   - What does the {trend} trend mean in practical terms?
   - Quantify the change: by how much and over what timeframe?
   - Is this trend accelerating, stable, or decelerating?

2. **Business Impact Projection**
   - What is the $ impact if trend continues?
   - Best case, expected case, worst case scenarios
   - At what point does this become critical?

3. **Forecast Confidence**
   - Given MAE of ${forecast_summary.get('mae', 0):,.0f}, how reliable is this forecast?
   - What decisions can be made confidently vs. what needs more data?
   - How far into the future are these forecasts valid?

4. **Intervention Opportunities**
   - What specific actions could alter this forecasted trajectory?
   - Which interventions have highest leverage?
   - How would we measure if interventions are working?

**Format:**
Be precise about confidence levels. Don't say "forecast shows X will happen"—say
"forecast suggests X is likely IF current patterns continue, with confidence level Y
based on model performance Z."

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt

    @staticmethod
    def root_cause_analysis_prompt(context: Dict[str, Any]) -> str:
        """
        Enhanced root cause analysis prompt with rigorous causal reasoning.

        Args:
            context: Multi-signal context dictionary

        Returns:
            Enhanced prompt string
        """
        metrics = context.get('leakage_metrics', {})
        anomalies = context.get('anomaly_patterns', {})
        top_categories = context.get('top_leakage_categories', [])
        top_regions = context.get('top_leakage_regions', [])
        discount_stats = context.get('discount_stats', {})
        temporal = context.get('temporal_trends', {})

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

        # Build comprehensive context
        comprehensive_context = ContextBuilder.build_comprehensive_context(
            metrics, model_metrics={'model_type': 'anomaly', **anomalies}
        )

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Root Cause Analysis of Revenue Leakage

Conduct a rigorous root cause analysis that explains WHY revenue leakage is occurring,
not just WHAT is happening. Distinguished causal factors from symptoms.

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

{comprehensive_context}

{ReasoningScaffolds.root_cause_reasoning_scaffold()}

**Critical Constraints:**

1. **Distinguish Symptoms from Causes**
   - "High discounts" is a WHAT (symptom), not a WHY (cause)
   - Ask: WHY are discounts high? (competitive pressure? poor pricing? sales incentives?)

2. **Propose Causal Mechanisms**
   - For each hypothesis, explain the causal chain
   - Why would X lead to Y?
   - What evidence supports this mechanism?

3. **Prioritize Root Causes**
   - Which causes are most impactful?
   - Which are most certain?
   - Which are most addressable?

4. **Identify Validation Needs**
   - What data would confirm/refute each hypothesis?
   - What can be tested with existing data?

**Format:**
Structure response with clear hypothesis → evidence → confidence for each root cause.
Rank causes by impact and certainty.

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt

    @staticmethod
    def cross_signal_insights_prompt(
        forecast_summary: Dict[str, float],
        anomaly_summary: Dict[str, Any],
        risk_summary: Dict[str, Any],
        leakage_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Enhanced cross-model synthesis prompt.

        Args:
            forecast_summary: Forecasting model results
            anomaly_summary: Anomaly detection results
            risk_summary: Risk scoring results
            leakage_metrics: Overall leakage metrics

        Returns:
            Enhanced prompt string
        """
        metrics_info = ""
        if leakage_metrics:
            metrics_info = f"""
**Current Leakage State:**
- Total Leakage: ${leakage_metrics.get('total_leakage', 0):,.2f}
- Profit Margin: {leakage_metrics.get('avg_profit_margin', 0):.2f}%
"""

        # Build model confidence contexts
        forecast_context = ContextBuilder.build_model_confidence_context(
            {**forecast_summary, 'model_type': 'forecasting'}, 'forecasting'
        )

        classification_context = ContextBuilder.build_model_confidence_context(
            {**risk_summary, 'model_type': 'classification'}, 'classification'
        )

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Cross-Model Intelligence Synthesis

Synthesize insights from three analytical models (Forecast, Anomaly Detection, Risk Scoring)
into unified intelligence that is more valuable than any single model.

{metrics_info}

**Forecast Model Output:**
- Historical Avg Monthly Leakage: ${forecast_summary.get('historical_avg', 0):,.2f}
- Forecasted Avg Monthly Leakage: ${forecast_summary.get('avg_monthly_forecast', 0):,.2f}
- Total Forecasted Leakage (next periods): ${forecast_summary.get('total_forecasted_leakage', 0):,.2f}
- Forecast Trend: {('increasing' if forecast_summary.get('avg_monthly_forecast', 0) > forecast_summary.get('historical_avg', 0) else 'decreasing')}
- Model Error (MAE): ${forecast_summary.get('mae', 0):,.2f}

{forecast_context}

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

{classification_context}

{ReasoningScaffolds.cross_signal_synthesis_scaffold()}

**Specific Analysis Requirements:**

1. **Signal Agreement Analysis**
   - Do all three models point to the same conclusion?
   - Example: High forecast + High anomalies + High risk scores → Systemic problem
   - What unified story emerges when signals agree?

2. **Signal Divergence Analysis**
   - Where do models disagree?
   - Example: Low current anomalies but high forecasted leakage → Early warning
   - What do divergences reveal about problem nature?

3. **Model Reliability Weighting**
   - Which model should we trust most given performance metrics?
   - When models disagree, which signal should dominate decision-making?

4. **Integrated Interpretation**
   - What is the single most important finding from all three models combined?
   - Are we seeing systemic deterioration or isolated incidents?

5. **Decision Implications**
   - What actions does this integrated view suggest?
   - What's urgent (act now) vs. important (monitor closely)?

**Format:**
Focus on synthesis, not repetition. Don't just summarize each model—explain what we
learn by viewing all three together that we couldn't learn from any single model.

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt

    @staticmethod
    def decision_priority_prompt(signals: Dict[str, Any]) -> str:
        """
        Enhanced decision prioritization prompt.

        Args:
            signals: Dictionary of signals and indicators

        Returns:
            Enhanced prompt string
        """
        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Decision Prioritization Framework

Translate analytical insights into a clear, prioritized action framework that helps
leaders decide what to act on NOW vs. what to monitor vs. what can wait.

**Situation Analysis:**
- **Leakage Severity:** {signals.get('leakage_severity', 'unknown')}
- **Forecast Trend:** {signals.get('forecast_trend', 'unknown')}
- **Anomaly Risk Level:** {signals.get('anomaly_risk', 'unknown')}
- **Model Confidence:** {signals.get('model_confidence', 'unknown')}
- **Estimated Monthly Impact:** ${signals.get('monthly_impact', 0):,.2f}
- **Quick Win Opportunities:** {signals.get('quick_wins', 'To be identified')}
- **Strategic Issues:** {signals.get('strategic_issues', 'To be identified')}

{ReasoningScaffolds.decision_priority_scaffold()}

**Critical Requirements:**

1. **Ruthless Prioritization**
   - Not everything can be urgent
   - Force-rank based on: Urgency × Impact × Confidence
   - Provide specific thresholds for escalation

2. **Action Specificity**
   - NO: "Improve discount management"
   - YES: "Cap Category X discounts at 15% starting next week, expect $25K monthly reduction"

3. **Confidence Calibration**
   - Separate high-confidence actions (safe to act) from low-confidence (need validation)
   - State what evidence supports each priority level

4. **Risk Assessment**
   - What's the risk of acting vs. not acting?
   - What's reversible vs. irreversible?

5. **Success Metrics**
   - How will we know if the action worked?
   - Specific metrics and timeframes

**Format:**
Use the priority levels (🔴 Critical, 🟡 Important, 🟢 Strategic, 📊 Monitor) with
clear reasoning for each classification. Be specific about actions, owners, and timelines.

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt

    @staticmethod
    def scenario_impact_prompt(
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
        scenario_description: str,
    ) -> str:
        """
        Enhanced scenario analysis prompt with trade-off evaluation.

        Args:
            before_metrics: Baseline metrics
            after_metrics: Post-scenario metrics
            scenario_description: Description of scenario change

        Returns:
            Enhanced prompt string
        """
        # Calculate deltas
        sales_delta = after_metrics.get('total_sales', 0) - before_metrics.get('total_sales', 0)
        profit_delta = after_metrics.get('total_profit', 0) - before_metrics.get('total_profit', 0)
        leakage_delta = after_metrics.get('total_leakage', 0) - before_metrics.get('total_leakage', 0)
        margin_delta = after_metrics.get('avg_profit_margin', 0) - before_metrics.get('avg_profit_margin', 0)

        sales_change_pct = (sales_delta / before_metrics.get('total_sales', 1)) * 100 if before_metrics.get('total_sales', 0) > 0 else 0
        profit_change_pct = (profit_delta / before_metrics.get('total_profit', 1)) * 100 if before_metrics.get('total_profit', 0) > 0 else 0

        prompt = f"""
{InsightGuardrails.get_analyst_persona()}

## YOUR TASK: Scenario Impact Assessment with Trade-off Analysis

Analyze the business implications of this scenario simulation, focusing on trade-offs,
feasibility, and recommendation quality.

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

**Computed Changes (These are facts - do not recalculate):**
- Sales Change: ${sales_delta:,.2f} ({sales_change_pct:+.2f}%)
- Profit Change: ${profit_delta:,.2f} ({profit_change_pct:+.2f}%)
- Leakage Change: ${leakage_delta:,.2f}
- Margin Change: {margin_delta:+.2f} percentage points

{ReasoningScaffolds.scenario_analysis_scaffold()}

**Specific Analysis Requirements:**

1. **Overall Assessment**
   - One-sentence headline: Is this scenario beneficial, neutral, or detrimental?
   - Quantify net benefit/cost

2. **Trade-off Matrix**
   - What did we gain? (Specific $ or %)
   - What did we sacrifice? (Specific $ or %)
   - Are the trade-offs acceptable given business context?

3. **Second-Order Effects**
   - What ripple effects might occur?
   - Customer reaction, competitive response, operational impacts

4. **Feasibility**
   - How difficult to implement? (Technical, organizational, political)
   - What resources needed?
   - What resistance might arise?

5. **Recommendation**
   - Clear YES/NO/MAYBE with specific reasoning
   - If YES, what modifications or safeguards?
   - If NO, what alternative to explore?
   - Confidence level in recommendation

**Format:**
Be balanced - acknowledge both benefits and costs. Don't advocate for change just because
it exists. Sometimes the answer is "not worth the implementation cost."

{InsightGuardrails.get_pre_prompt_guardrails()}
"""
        return prompt
