"""
Correction capture subpackage.

Provides functionality to write low-confidence query-response pairs
to structured markdown files for future model training and correction.
"""

# Re-export primary correction interfaces for convenient imports
from autocorrection_pipeline.correction.capture import CorrectionCapture, CorrectionRecord

# Define public API for this subpackage
__all__ = ["CorrectionCapture", "CorrectionRecord"]
