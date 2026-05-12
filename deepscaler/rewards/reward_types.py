"""
Reward type definitions for DeepScaler.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardResult:
    """Result of a reward function evaluation."""
    score: float
    is_correct: bool
    explanation: Optional[str] = None
