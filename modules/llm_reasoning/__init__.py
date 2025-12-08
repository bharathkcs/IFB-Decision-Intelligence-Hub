"""
LLM Reasoning Enhancement Module

This module provides advanced LLM insight generation capabilities with:
- Reasoning scaffolding (Chain-of-Thought)
- Fact/Interpretation/Implication separation
- Data-grounded context assembly
- Guardrails against hallucination
- Uncertainty handling

Design Philosophy:
- Fully additive - does not modify analytics code
- Isolated to LLM layer only
- Grounded in actual computed metrics
- Transparent about limitations
"""

from .prompt_templates import PromptTemplates
from .context_builder import ContextBuilder
from .reasoning_scaffolds import ReasoningScaffolds
from .guardrails import InsightGuardrails
from .enhanced_insight_generator import EnhancedInsightGenerator

__all__ = [
    'PromptTemplates',
    'ContextBuilder',
    'ReasoningScaffolds',
    'InsightGuardrails',
    'EnhancedInsightGenerator'
]
