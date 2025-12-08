"""
Integration Helper for Enhanced LLM Reasoning

Provides easy integration of enhanced LLM insight generation into existing code.

Usage:
    from modules.llm_reasoning.integration import get_enhanced_insight_generator

    # Drop-in replacement for LLMInsightGenerator
    insight_generator = get_enhanced_insight_generator(llm_client)

    # Use exactly like the old generator
    summary = insight_generator.generate_executive_summary(metrics)
"""

from typing import Any
from .enhanced_insight_generator import EnhancedInsightGenerator


def get_enhanced_insight_generator(llm_client: Any) -> EnhancedInsightGenerator:
    """
    Get an enhanced insight generator instance.

    This is a drop-in replacement for LLMInsightGenerator that provides
    ChatGPT-level analytical depth.

    Args:
        llm_client: LLM client with conversational_response method

    Returns:
        EnhancedInsightGenerator instance

    Example:
        >>> from modules.llm_interface import LLMInterface
        >>> from modules.llm_reasoning.integration import get_enhanced_insight_generator
        >>>
        >>> llm = LLMInterface()
        >>> generator = get_enhanced_insight_generator(llm)
        >>> summary = generator.generate_executive_summary(metrics)
    """
    return EnhancedInsightGenerator(llm_client)


def enable_enhanced_insights_globally():
    """
    Enable enhanced insights globally by monkey-patching the import.

    This allows existing code to automatically use enhanced insights
    without any code changes.

    Usage:
        # In app.py or main module, before other imports:
        from modules.llm_reasoning.integration import enable_enhanced_insights_globally
        enable_enhanced_insights_globally()

        # Now all code that uses LLMInsightGenerator will get the enhanced version
    """
    import sys
    import modules.revenue_leakage_detector as rld_module

    # Replace LLMInsightGenerator with EnhancedInsightGenerator
    rld_module.LLMInsightGenerator = EnhancedInsightGenerator

    print("✅ Enhanced LLM insights enabled globally")


# Compatibility alias for backward compatibility
LLMInsightGenerator = EnhancedInsightGenerator


__all__ = [
    'get_enhanced_insight_generator',
    'enable_enhanced_insights_globally',
    'EnhancedInsightGenerator',
    'LLMInsightGenerator'  # Alias for compatibility
]
