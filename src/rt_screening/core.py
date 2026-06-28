"""Public API for the response-time-aware screening pipeline.

Implementation is split into the engine module with small thematic wrapper
modules (policies, metrics, leakage, reporting) for easier navigation.
"""
from .engine import *  # noqa: F401,F403
