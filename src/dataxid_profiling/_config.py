from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VALID_MODES = ("complete", "overview")


@dataclass(frozen=True)
class ProfileConfig:
    """Profiling configuration. Immutable after creation."""

    title: str = "Dataset Report"

    # Type inference
    text_unique_ratio: float = 0.5  # unique/count > this → text-like, not categorical

    # Alert thresholds
    missing_threshold: float = 0.05  # > 5% missing → HIGH_MISSING alert
    cardinality_threshold: float = 0.95  # unique/count > 95% → HIGH_CARDINALITY alert
    correlation_threshold: float = 0.8  # |corr| > 0.8 → HIGH_CORRELATION alert
    constant_threshold: int = 1  # n_unique <= 1 → CONSTANT alert
    zero_threshold: float = 0.05  # > 5% zeros → HIGH_ZEROS alert
    skewness_threshold: float = 2.0  # |skewness| > 2 → SKEWED alert
    imbalance_threshold: float = 0.9  # top_value_pct > 90% → IMBALANCED alert
    duplicate_threshold: float = 0.0  # any duplicate → DUPLICATES alert
    uniform_pvalue_threshold: float = 0.05  # p > 0.05 → UNIFORM alert (can't reject uniform H0)

    # Interactions
    interaction_sample_size: int = 100_000  # sample above this row count
    interaction_sample_seed: int = 42  # reproducible sampling
    interaction_cardinality_limit: int = 50  # skip categorical cols with more unique values

    # Display
    n_top_values: int = 5  # value_counts'ta gösterilecek top N
    histogram_bins: int = 50

    # Profiling depth: "complete" (default) or "overview" (skip expensive computations)
    mode: Literal["complete", "overview"] = "complete"

    # Kolon bazlı override
    column_overrides: dict[str, dict] = field(default_factory=dict)

    @property
    def is_overview(self) -> bool:
        return self.mode == "overview"

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            msg = f"mode must be one of {VALID_MODES}, got '{self.mode}'"
            raise ValueError(msg)
        if not 0.0 <= self.text_unique_ratio <= 1.0:
            msg = f"text_unique_ratio must be in [0, 1], got {self.text_unique_ratio}"
            raise ValueError(msg)
        if not 0.0 <= self.missing_threshold <= 1.0:
            msg = f"missing_threshold must be in [0, 1], got {self.missing_threshold}"
            raise ValueError(msg)
        if not 0.0 <= self.cardinality_threshold <= 1.0:
            msg = f"cardinality_threshold must be in [0, 1], got {self.cardinality_threshold}"
            raise ValueError(msg)
        if self.n_top_values < 1:
            msg = f"n_top_values must be >= 1, got {self.n_top_values}"
            raise ValueError(msg)
        if self.histogram_bins < 2:
            msg = f"histogram_bins must be >= 2, got {self.histogram_bins}"
            raise ValueError(msg)
        if self.interaction_sample_size < 1000:
            msg = f"interaction_sample_size must be >= 1000, got {self.interaction_sample_size}"
            raise ValueError(msg)
        if self.interaction_cardinality_limit < 2:
            msg = (
                "interaction_cardinality_limit must be >= 2, "
                f"got {self.interaction_cardinality_limit}"
            )
            raise ValueError(msg)
