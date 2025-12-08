"""
Reasoning Scaffolds for LLM Insight Generation

Provides Chain-of-Thought patterns and reasoning structures that guide
LLMs to produce more analytical, rigorous, and well-structured insights.
"""

from typing import Dict, List, Any


class ReasoningScaffolds:
    """
    Reasoning scaffolding patterns for structured analytical thinking.

    These scaffolds guide the LLM through systematic reasoning processes,
    separating facts, interpretation, and implications.
    """

    @staticmethod
    def chain_of_thought_scaffold() -> str:
        """
        Chain-of-Thought reasoning pattern.

        Guides LLM to think step-by-step before drawing conclusions.
        """
        return """
**Think through this systematically:**

STEP 1: FACTS - What do the numbers actually show?
- List only objective observations from the data
- No interpretation yet, just what IS

STEP 2: PATTERNS - What patterns or relationships emerge?
- Compare metrics against each other
- Identify anomalies, trends, or concentrations
- Note what's surprising or expected

STEP 3: INTERPRETATION - What do these patterns suggest?
- Propose causal mechanisms (why might this be happening?)
- Consider multiple explanations
- Distinguish correlation from causation

STEP 4: IMPLICATIONS - What does this mean for decisions?
- Business impact assessment
- Risk and opportunity identification
- Action implications

STEP 5: CONFIDENCE - How certain are we?
- What supports this interpretation?
- What contradicts it or creates uncertainty?
- What additional data would help?
"""

    @staticmethod
    def fact_interpretation_implication_scaffold() -> str:
        """
        FII (Fact-Interpretation-Implication) separation pattern.

        Forces clear separation between what IS, what it MEANS, and what to DO.
        """
        return """
Structure your response with clear separation:

## FACTS (What the data shows)
- [Only objective statements: "X is $Y", "A is Z% higher than B"]
- [No interpretation, no "suggests" or "indicates"]
- [Specific numbers and precise comparisons]

## INTERPRETATION (What this means)
- [Why might these facts be occurring? Causal mechanisms]
- [What patterns do they reveal about business processes?]
- [How do different signals relate to each other?]
- [Note: Use "likely", "suggests", "may indicate" appropriately]

## IMPLICATIONS (What to do about it)
- [Decision impact: How should this change actions?]
- [Risk assessment: What are the consequences of ignoring this?]
- [Opportunity identification: What can be gained?]
- [Prioritization: What's urgent vs. important?]

## CONFIDENCE & LIMITATIONS
- [What are we confident about and why?]
- [What remains uncertain?]
- [What data gaps limit our conclusions?]
"""

    @staticmethod
    def cross_signal_synthesis_scaffold() -> str:
        """
        Multi-model synthesis reasoning pattern.

        For integrating insights from multiple analytical models.
        """
        return """
Synthesize multiple analytical signals systematically:

## SIGNAL INVENTORY
For each model output:
- What is it telling us? (Main message)
- How confident is it? (Based on model metrics)
- What does it NOT tell us? (Limitations)

## CONVERGENCE ANALYSIS
- Where do multiple signals agree?
- When models agree, what unified story emerges?
- Does convergence increase our confidence?

## DIVERGENCE ANALYSIS
- Where do signals disagree or point different directions?
- What might explain these divergences?
- Do divergences reveal different time horizons (current vs. future)?
- Do divergences suggest data quality issues?

## EMERGENT INSIGHTS
- What patterns only become visible when viewing all signals together?
- Are we seeing systemic issues or isolated incidents?
- Structural problems (long-term) vs. tactical problems (short-term)?

## INTEGRATED INTERPRETATION
- What is the single most important insight from all signals combined?
- How should decision-makers weight different signals?
- What actions does this integrated view suggest?
"""

    @staticmethod
    def root_cause_reasoning_scaffold() -> str:
        """
        Root cause analysis reasoning pattern.

        Guides systematic WHY analysis, distinguishing causes from symptoms.
        """
        return """
Conduct systematic root cause analysis:

## SYMPTOM IDENTIFICATION
- What are we observing? (The "WHAT")
- Is this symptom or cause? (Be careful not to confuse them)

## HYPOTHESIS GENERATION
Generate 3-5 potential root causes:
1. [Hypothesis 1: Mechanism explaining WHY this occurs]
   - Supporting evidence from data
   - Contradicting evidence or gaps
   - Confidence level (High/Medium/Low)

2. [Hypothesis 2: Alternative mechanism]
   - Supporting evidence
   - Contradicting evidence
   - Confidence level

[Continue for 3-5 hypotheses]

## MECHANISM EXPLANATION
For each likely root cause:
- Explain the causal chain: X causes Y which leads to Z
- Why is this mechanism plausible given business context?
- What would we expect to see if this is the true cause?

## PRIORITIZATION
Rank root causes by:
- Probability (How likely is this the true cause?)
- Impact (How much does this contribute to the problem?)
- Actionability (Can we address this?)

## VALIDATION NEEDS
- What additional data would confirm/refute each hypothesis?
- What experiments or analyses would help?
- Can we test these hypotheses with existing data?
"""

    @staticmethod
    def decision_priority_scaffold() -> str:
        """
        Decision prioritization reasoning pattern.

        Translates analysis into actionable priorities with clear rationale.
        """
        return """
Structure decision priorities with clear reasoning:

For EACH priority level (Critical / Important / Monitor):

## [Priority Level]: [Specific Action]

**Decision Rationale:**
- WHY this priority level? (Urgency × Impact calculation)
- What specific threshold or signal triggered this classification?
- What happens if we act now vs. wait?

**Evidence Basis:**
- Which metrics/signals support this priority?
- How strong is the evidence? (Model confidence, data quality)
- What assumptions are we making?

**Action Specificity:**
- WHAT exactly should be done? (Not "improve X" but "do Y to improve X")
- WHO should own this action?
- WHEN should it be completed? (Specific timeframe, not "soon")
- HOW will we know if it worked? (Success metrics)

**Risk Assessment:**
- Risk of acting: What could go wrong with this action?
- Risk of not acting: What deteriorates if we ignore this?
- Reversibility: Can we undo this if wrong?

**Confidence Level:**
- How confident are we this is the right priority? (High/Med/Low)
- What makes us confident or uncertain?
- At what point should we reassess?
"""

    @staticmethod
    def uncertainty_calibration_scaffold() -> str:
        """
        Uncertainty and confidence calibration pattern.

        Forces explicit reasoning about confidence levels and data limitations.
        """
        return """
Calibrate confidence and uncertainty explicitly:

## EVIDENCE STRENGTH
Assess the quality of evidence:
- Data sufficiency: Do we have enough data points?
- Data quality: Are measurements reliable?
- Model performance: How accurate are predictions?
- Historical validation: Have similar analyses been accurate before?

## CONFIDENCE LEVELS
For each major conclusion:

**HIGH CONFIDENCE** (We can be quite certain)
- [Conclusion X]
- Basis: [Why we're confident - specific metrics, validation, consistency]
- Safe actions: [What can be decided based on this alone]

**MODERATE CONFIDENCE** (Probable but not certain)
- [Conclusion Y]
- Basis: [Evidence + uncertainty factors]
- Recommended approach: [Validate before major decisions, pilot tests]

**LOW CONFIDENCE** (Speculative or insufficient data)
- [Conclusion Z]
- Basis: [Why uncertain - data gaps, conflicting signals, model limitations]
- Do NOT act on: [What decisions should NOT be made yet]

## UNCERTAINTY SOURCES
What creates uncertainty in our analysis?
- Data limitations: [Specific gaps or quality issues]
- Model limitations: [What the models can't capture]
- External factors: [Market changes, unmeasured variables]
- Time horizon: [Forecast uncertainty increases over time]

## IMPROVING CONFIDENCE
What would reduce uncertainty?
- Additional data needed: [Specific data that would help]
- Validation approaches: [How to test our conclusions]
- Alternative analyses: [Other methods that could confirm/refute]
"""

    @staticmethod
    def scenario_analysis_scaffold() -> str:
        """
        Scenario interpretation reasoning pattern.

        For what-if analysis and business impact assessment.
        """
        return """
Analyze scenario impact systematically:

## SCENARIO CONTEXT
- What changed? (Specific parameter modifications)
- Why was this scenario tested? (Business question)
- Assumptions made in this scenario

## DIRECT EFFECTS
- What metrics changed and by how much?
- Are changes in expected direction?
- Magnitude: Significant or marginal?

## TRADE-OFF ANALYSIS
Create explicit trade-off matrix:

| Gained | Lost | Net Assessment |
|--------|------|----------------|
| [Benefit 1: $X or Y%] | [Cost 1: $A or B%] | [Better/Worse/Neutral] |
| [Benefit 2] | [Cost 2] | [Assessment] |

**Overall Trade-off:** [Are we better or worse off? By how much?]

## SECOND-ORDER EFFECTS
What ripple effects might occur?
- Customer behavior changes
- Competitive response
- Operational impacts
- Cash flow/timing effects
- Stakeholder reactions (sales team, finance, customers)

## FEASIBILITY ASSESSMENT
- Implementation complexity: [Easy/Medium/Hard + specifics]
- Resource requirements: [What's needed to execute]
- Time to implement: [Realistic timeline]
- Organizational resistance: [Who might oppose and why]

## RECOMMENDATION
**Should we implement this scenario?**
- [ ] YES - [Specific reasoning + suggested modifications]
- [ ] NO - [Why not + what to try instead]
- [ ] MAYBE - [What validation needed before deciding]

**Confidence in recommendation:** [High/Medium/Low + reasoning]
"""

    @staticmethod
    def get_scaffold(scaffold_type: str) -> str:
        """
        Get a specific reasoning scaffold by type.

        Args:
            scaffold_type: Type of scaffold (cot, fii, cross_signal, root_cause,
                          decision_priority, uncertainty, scenario)

        Returns:
            Scaffold template string
        """
        scaffolds = {
            'cot': ReasoningScaffolds.chain_of_thought_scaffold,
            'fii': ReasoningScaffolds.fact_interpretation_implication_scaffold,
            'cross_signal': ReasoningScaffolds.cross_signal_synthesis_scaffold,
            'root_cause': ReasoningScaffolds.root_cause_reasoning_scaffold,
            'decision_priority': ReasoningScaffolds.decision_priority_scaffold,
            'uncertainty': ReasoningScaffolds.uncertainty_calibration_scaffold,
            'scenario': ReasoningScaffolds.scenario_analysis_scaffold,
        }

        if scaffold_type not in scaffolds:
            raise ValueError(f"Unknown scaffold type: {scaffold_type}. "
                           f"Available: {list(scaffolds.keys())}")

        return scaffolds[scaffold_type]()

    @staticmethod
    def combine_scaffolds(scaffold_types: List[str]) -> str:
        """
        Combine multiple scaffolds into a comprehensive reasoning framework.

        Args:
            scaffold_types: List of scaffold type names

        Returns:
            Combined scaffold string
        """
        scaffolds = [ReasoningScaffolds.get_scaffold(st) for st in scaffold_types]
        return "\n\n---\n\n".join(scaffolds)
