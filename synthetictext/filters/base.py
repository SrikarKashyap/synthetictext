"""Abstract base class for all filters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFilter(ABC):
    """Abstract base class that all filters must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this filter."""

    @abstractmethod
    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return a filtered subset of *df* without modifying the original.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least columns ``text`` (str) and ``label`` (int).
        **kwargs :
            Filter-specific options (e.g. ``original_df`` for dedup).

        Returns
        -------
        pd.DataFrame
            A (potentially smaller) copy of the input DataFrame.
        """
