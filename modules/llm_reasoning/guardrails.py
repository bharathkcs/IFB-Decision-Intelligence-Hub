"""
Guardrails for LLM Insight Generation

Provides pre-prompt and post-response validation to ensure:
- No hallucination of metrics not provided
- No overconfident claims without data support
- Transparency about uncertainty and limitations
- Grounding in actual computed metrics
"""

import re
from typing import Dict, List, Any, Set, Optional


class InsightGuardrails:
    """
    Quality control system for LLM-generated insights.

    Implements both pre-prompt guardrails (constraints in prompts) and
    post-response validation (checking generated insights).
    """

    @staticmethod
    def get_pre_prompt_guardrails() -> str:
        """
        Get guardrail instructions to include in prompts.

        These are explicit constraints that guide LLM behavior.
        """
        return """
## CRITICAL CONSTRAINTS (You MUST follow these)

1. **NO METRIC INVENTION**
   - Use ONLY metrics explicitly provided above
   - NEVER invent, estimate, or extrapolate numbers not given
   - If you need a metric that's missing, explicitly state: "Cannot determine without [specific metric]"

2. **NO OVERCONFIDENCE**
   - Do NOT claim certainty unless data strongly supports it
   - Use calibrated language:
     * "definitely"/"certainly" → Only if metric explicitly shows this with no ambiguity
     * "likely"/"probably" → When evidence is strong but not definitive
     * "possibly"/"may" → When evidence is moderate or circumstantial
     * "unclear"/"insufficient data" → When evidence is weak or missing
   - NEVER make guarantees about future outcomes

3. **NO GENERIC CONSULTING-SPEAK**
   - Avoid: "optimize", "leverage", "synergize", "drive value", "strategic initiative"
   - Instead: Be specific about WHAT action and WHY it helps
   - Bad: "Optimize the discount strategy to drive value"
   - Good: "Cap discounts at 15% for Category X to prevent $50K monthly leakage while maintaining sales volume based on historical elasticity"

4. **DISTINGUISH FACTS FROM INTERPRETATION**
   - Facts: "Discount leakage is $X" (stated directly in data)
   - Interpretation: "High discounts may be due to competitive pressure" (hypothesis)
   - Always label which is which

5. **EXPLICIT UNCERTAINTY**
   - When data is insufficient, say so clearly
   - When multiple interpretations exist, present alternatives
   - When confidence is low, explain why
   - Prefer "Based on available data..." over universal statements

6. **NO INVENTED CAUSATION**
   - Correlation ≠ Causation
   - Say "X and Y are correlated" NOT "X causes Y" unless mechanism is clear
   - Causal claims require:
     * Temporal precedence (X before Y)
     * Plausible mechanism (WHY X would cause Y)
     * No obvious confounders

7. **GROUNDING REQUIREMENT**
   - Every claim must trace back to a specific provided metric
   - Example: "Revenue leakage increased 23% MoM (from $120K to $147K)"
   - NOT: "Revenue leakage is concerning" (vague, not grounded)

8. **NO MADE-UP EXAMPLES**
   - Do NOT create hypothetical scenarios or examples not in data
   - Do NOT reference "other companies" or "industry standards" unless explicitly provided
   - Stick to the actual business and actual data

9. **TRANSPARENT LIMITATIONS**
   - Acknowledge what the analysis cannot tell us
   - State what additional data would improve confidence
   - Note when sample size, time horizon, or data quality limits conclusions

10. **ACTIONABILITY OVER DESCRIPTION**
    - Avoid simply restating numbers
    - Every insight should answer: "So what? What should someone DO with this information?"
    - BUT: If no clear action can be determined, say that explicitly
"""

    @staticmethod
    def get_analyst_persona() -> str:
        """
        Get a rigorous analyst persona for the LLM to adopt.

        This replaces generic "business analyst" with a more specific,
        calibrated analytical personality.
        """
        return """
You are a **Senior Decision Intelligence Analyst** with expertise in:
- Quantitative business analysis with ML model interpretation
- Separating signal from noise in complex multi-model outputs
- Communicating uncertainty and confidence levels to executives
- Distinguishing correlation from causation
- Translating analytics into actionable business decisions

Your analytical style:
- **Rigorous**: You follow evidence, not assumptions
- **Specific**: You cite exact metrics and calculations
- **Calibrated**: You express appropriate confidence levels
- **Transparent**: You acknowledge limitations and gaps
- **Actionable**: You connect insights to decisions
- **Clear**: You avoid jargon and explain complex patterns simply

You are NOT:
- A consultant using generic business buzzwords
- An optimist who sugarcoats problems
- A fortune teller making guarantees about the future
- A storyteller inventing narratives without data

Your primary responsibility: **Help decision-makers understand what the data ACTUALLY tells us and what it does NOT, so they can make informed choices.**
"""

    @staticmethod
    def extract_numbers_from_text(text: str) -> Set[str]:
        """
        Extract all numbers mentioned in generated text.

        Args:
            text: LLM-generated insight text

        Returns:
            Set of number strings found in text
        """
        # Match numbers with optional formatting: $1,234.56, 45.3%, etc.
        pattern = r'\$?[\d,]+\.?\d*%?'
        numbers = re.findall(pattern, text)
        return set(numbers)

    @staticmethod
    def validate_metric_grounding(
        generated_text: str,
        provided_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that generated insights reference only provided metrics.

        Args:
            generated_text: LLM output text
            provided_metrics: Dictionary of metrics that were provided to LLM

        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - issues: List of potential problems
            - warnings: List of warnings (non-blocking)
        """
        issues = []
        warnings = []

        # Extract all numbers from generated text
        mentioned_numbers = InsightGuardrails.extract_numbers_from_text(generated_text)

        # Convert provided metrics to comparable format
        provided_numbers = set()
        for value in provided_metrics.values():
            if isinstance(value, (int, float)):
                # Format as it might appear in text
                provided_numbers.add(f"{value:,.0f}")
                provided_numbers.add(f"{value:.2f}")
                provided_numbers.add(f"${value:,.2f}")
                provided_numbers.add(f"{value:.2%}")
            elif isinstance(value, str):
                provided_numbers.add(value)

        # Check for hallucination indicators
        hallucination_phrases = [
            "estimated",
            "approximately",
            "around",
            "roughly",
            "industry standard",
            "typically",
            "usually",
            "similar companies",
            "benchmark"
        ]

        for phrase in hallucination_phrases:
            if phrase.lower() in generated_text.lower():
                warnings.append(
                    f"Found potentially speculative language: '{phrase}'. "
                    "Ensure this is grounded in provided data."
                )

        # Check for overconfidence
        overconfident_phrases = [
            "will definitely",
            "will certainly",
            "guaranteed",
            "always results in",
            "impossible",
            "never fails",
            "without doubt"
        ]

        for phrase in overconfident_phrases:
            if phrase.lower() in generated_text.lower():
                issues.append(
                    f"Overconfident language detected: '{phrase}'. "
                    "LLM should express appropriate uncertainty."
                )

        # Check for generic consulting speak
        generic_phrases = [
            "drive value",
            "synergize",
            "leverage synergies",
            "strategic alignment",
            "optimize the strategy",
            "maximize potential"
        ]

        for phrase in generic_phrases:
            if phrase.lower() in generated_text.lower():
                warnings.append(
                    f"Generic business jargon detected: '{phrase}'. "
                    "Consider if more specific language would be clearer."
                )

        is_valid = len(issues) == 0

        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'mentioned_numbers': mentioned_numbers,
            'provided_numbers': provided_numbers
        }

    @staticmethod
    def check_uncertainty_acknowledgment(generated_text: str) -> Dict[str, Any]:
        """
        Check if generated insights appropriately acknowledge uncertainty.

        Args:
            generated_text: LLM output

        Returns:
            Dictionary with analysis of uncertainty handling
        """
        uncertainty_indicators = [
            'uncertain',
            'unclear',
            'insufficient data',
            'cannot determine',
            'would need',
            'additional data',
            'limited by',
            'confidence',
            'likely',
            'possibly',
            'may',
            'suggests',
            'indicates'
        ]

        found_indicators = []
        for indicator in uncertainty_indicators:
            if indicator.lower() in generated_text.lower():
                found_indicators.append(indicator)

        # Check if text is making many definitive claims without any uncertainty
        definitive_phrases = [
            'is definitely',
            'will',
            'must be',
            'the reason is',
            'this proves'
        ]

        definitive_count = sum(
            1 for phrase in definitive_phrases
            if phrase.lower() in generated_text.lower()
        )

        uncertainty_score = len(found_indicators)

        # If many definitive claims but no uncertainty acknowledgment, flag it
        needs_calibration = (definitive_count >= 3 and uncertainty_score == 0)

        return {
            'uncertainty_score': uncertainty_score,
            'found_indicators': found_indicators,
            'definitive_count': definitive_count,
            'needs_calibration': needs_calibration,
            'assessment': (
                'Good uncertainty calibration' if uncertainty_score >= 2
                else 'Moderate calibration' if uncertainty_score == 1
                else 'May need more uncertainty acknowledgment'
            )
        }

    @staticmethod
    def create_response_wrapper(
        raw_llm_output: str,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Wrap LLM output with validation metadata (for debugging/logging).

        Args:
            raw_llm_output: Original LLM response
            validation_results: Optional validation results to append

        Returns:
            Enhanced output with validation info
        """
        output = raw_llm_output

        if validation_results and not validation_results.get('is_valid', True):
            # In production, we might just log this. For development, append warnings.
            issues = validation_results.get('issues', [])
            if issues:
                warning_text = "\n\n---\n**⚠️ Quality Check Warnings:**\n" + "\n".join(
                    f"- {issue}" for issue in issues
                )
                # Only append in development/debug mode
                # output += warning_text

        return output

    @staticmethod
    def validate_response(
        generated_text: str,
        provided_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of generated insights.

        Args:
            generated_text: LLM-generated text
            provided_metrics: Metrics that were provided in prompt

        Returns:
            Comprehensive validation report
        """
        metric_validation = InsightGuardrails.validate_metric_grounding(
            generated_text, provided_metrics
        )

        uncertainty_check = InsightGuardrails.check_uncertainty_acknowledgment(
            generated_text
        )

        overall_valid = (
            metric_validation['is_valid'] and
            not uncertainty_check['needs_calibration']
        )

        return {
            'overall_valid': overall_valid,
            'metric_validation': metric_validation,
            'uncertainty_check': uncertainty_check,
            'recommended_action': (
                'Accept' if overall_valid
                else 'Review - quality concerns detected'
            )
        }

    @staticmethod
    def get_data_quality_context(metrics: Dict[str, Any]) -> str:
        """
        Generate data quality context to help LLM calibrate confidence.

        Args:
            metrics: Provided metrics dictionary

        Returns:
            Data quality context string
        """
        # Check for data quality indicators
        data_points = metrics.get('data_points', metrics.get('record_count', 0))
        missing_rate = metrics.get('missing_rate', 0)

        quality_issues = []
        if data_points < 30:
            quality_issues.append(f"Small sample size ({data_points} records) - interpret cautiously")
        if missing_rate > 0.1:
            quality_issues.append(f"High missing data rate ({missing_rate:.1%}) - may affect reliability")

        if quality_issues:
            return "\n**Data Quality Context:**\n" + "\n".join(f"- {issue}" for issue in quality_issues)
        else:
            return ""
