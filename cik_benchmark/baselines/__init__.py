"""Minimal baselines for the custom CXR task family."""

from .cxr_baselines import DirectContextPromptBaseline, LastValueBaseline

__all__ = ["LastValueBaseline", "DirectContextPromptBaseline"]
