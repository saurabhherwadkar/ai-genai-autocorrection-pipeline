"""
AI GenAI Auto-Correction Pipeline.

A pipeline that captures LLM query-response pairs, compares them against
ideal responses, and generates correction files for responses below the
confidence threshold to support future model training.
"""

# Package version identifier
__version__ = "0.1.0"
