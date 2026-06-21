"""
Demand Forecasting for VIKMO Assignment

Implements forecasting models for sales history data.

Requirements: FC-001 to FC-015
Tasks: P6-T003, P6-T004, P6-T005
"""

import pandas as pd
from typing import Optional


def load_sales_data(path: str = "sales_history.csv") -> pd.DataFrame:
    """
    Load sales history data.
    
    Args:
        path: Path to sales history CSV
    
    Returns:
        pd.DataFrame: Sales data
    
    Requirements: FC-012 to FC-015
    Task: P6-T001
    """
    raise NotImplementedError("Implement if doing Part B")


def train_forecast_model(data: pd.DataFrame, test_weeks: int = 4) -> dict:
    """
    Train forecasting model and evaluate.
    
    Args:
        data: Sales history data
        test_weeks: Number of weeks to hold out for testing
    
    Returns:
        dict: Model, predictions, metrics
    
    Requirements: FC-001 to FC-010
    Task: P6-T003, P6-T004, P6-T005
    """
    raise NotImplementedError("Implement if doing Part B")


def compute_naive_baseline(data: pd.DataFrame, test_weeks: int = 4) -> dict:
    """
    Compute naive baseline (seasonal naive or moving average).
    
    Args:
        data: Sales history data
        test_weeks: Number of weeks to predict
    
    Returns:
        dict: Baseline predictions and error metrics
    
    Requirements: FC-006, FC-007, FC-008
    Task: P6-T002
    """
    raise NotImplementedError("Implement if doing Part B")
