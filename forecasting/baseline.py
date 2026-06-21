"""
Baseline models for Demand Forecasting

Implements naive baselines for comparison.

Requirements: FC-006, FC-007, FC-008
Tasks: P6-T002
"""

import pandas as pd


def seasonal_naive_baseline(data: pd.DataFrame, test_weeks: int = 4) -> pd.DataFrame:
    """
    Seasonal naive baseline: predict same value from last year.
    
    Args:
        data: Sales history data
        test_weeks: Number of weeks to predict
    
    Returns:
        pd.DataFrame: Predictions
    """
    raise NotImplementedError("Implement if doing Part B")


def moving_average_baseline(data: pd.DataFrame, window: int = 4, test_weeks: int = 4) -> pd.DataFrame:
    """
    Moving average baseline.
    
    Args:
        data: Sales history data
        window: Moving average window size
        test_weeks: Number of weeks to predict
    
    Returns:
        pd.DataFrame: Predictions
    """
    raise NotImplementedError("Implement if doing Part B")


def last_value_baseline(data: pd.DataFrame, test_weeks: int = 4) -> pd.DataFrame:
    """
    Last value baseline: predict last observed value.
    
    Args:
        data: Sales history data
        test_weeks: Number of weeks to predict
    
    Returns:
        pd.DataFrame: Predictions
    """
    raise NotImplementedError("Implement if doing Part B")
