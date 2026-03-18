"""
Event study utilities for computing Cumulative Abnormal Returns (CARs)
and Cumulative Abnormal Volumes (CAVs) around litigation filing dates.

Methodology:
    - Fama-French 3-factor model for expected returns estimation
    - Estimation window: [-302, -52] business days before event (configurable)
    - Event window: [-2, +2] business days around filing date (configurable)
    - Abnormal volume following Chae (2005): log turnover minus mean log turnover
      over [-40, -11] pre-event window
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy import asarray, isnan
from pandas.tseries.offsets import BDay
from statsmodels.api import OLS, add_constant

from src.utils.config import EVENT_STUDY_PARAMS, FF_FACTORS


def estimate_factor_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    order: List[str] = ["const"] + FF_FACTORS,
) -> Optional[pd.Series]:
    """
    Estimate factor loadings via OLS over the estimation window.

    Parameters
    ----------
    returns : pd.Series
        Daily stock returns for a single firm over the estimation period.
    factors : pd.DataFrame
        Fama-French factor returns aligned to the same dates.
    order : list
        Column ordering for the design matrix.

    Returns
    -------
    pd.Series or None
        Estimated betas (including intercept), or None if estimation fails.
    """
    y = asarray(returns.dropna())
    if len(y) < 60:  # Minimum observations for reliable estimation
        return None

    X = add_constant(factors[FF_FACTORS])
    try:
        model = OLS(y, X[order], missing="drop").fit()
        return model.params
    except Exception:
        return None


def compute_abnormal_returns(
    returns: pd.Series,
    factors: pd.DataFrame,
    betas: pd.Series,
) -> pd.Series:
    """
    Compute abnormal returns as realized minus model-implied returns.

    Parameters
    ----------
    returns : pd.Series
        Realized daily returns over the event window.
    factors : pd.DataFrame
        Factor returns over the event window.
    betas : pd.Series
        Estimated factor loadings from the estimation window.

    Returns
    -------
    pd.Series
        Abnormal returns for each day in the event window.
    """
    X = add_constant(factors[FF_FACTORS])
    expected = X[["const"] + FF_FACTORS].dot(betas)
    return returns - expected


def compute_car(
    litigation_events: pd.DataFrame,
    daily_returns: pd.DataFrame,
    ff_factors: pd.DataFrame,
    event_window_pre: int = EVENT_STUDY_PARAMS["event_window_pre"],
    event_window_post: int = EVENT_STUDY_PARAMS["event_window_post"],
    estimation_gap: int = EVENT_STUDY_PARAMS["estimation_gap"],
    estimation_length: int = EVENT_STUDY_PARAMS["estimation_length"],
    event_date_col: str = "First Litigation Event",
    firm_id_col: str = "gvkey_side_B",
) -> pd.DataFrame:
    """
    Compute Cumulative Abnormal Returns for each litigation-firm pair.

    Parameters
    ----------
    litigation_events : pd.DataFrame
        One row per litigation event. Must contain `event_date_col` and `firm_id_col`.
    daily_returns : pd.DataFrame
        Daily returns indexed by (date, gvkey) or with gvkey as columns.
    ff_factors : pd.DataFrame
        Daily Fama-French factors indexed by date.
    event_window_pre, event_window_post : int
        Number of business days before/after the event.
    estimation_gap : int
        Gap in business days between estimation and event windows.
    estimation_length : int
        Number of business days in the estimation window.
    event_date_col, firm_id_col : str
        Column names for event dates and firm identifiers.

    Returns
    -------
    pd.DataFrame
        Columns: firm_id, event_date, CAR, n_days_event, n_days_estimation,
        plus daily abnormal returns AR_-2 through AR_+2.
    """
    results = []

    for idx, event in litigation_events.iterrows():
        event_day = pd.Timestamp(event[event_date_col])
        gvkey = event[firm_id_col]

        if pd.isna(gvkey):
            continue

        # Define windows
        est_end = event_day - BDay(event_window_pre + estimation_gap)
        est_start = est_end - BDay(estimation_length)
        evt_start = event_day - BDay(event_window_pre)
        evt_end = event_day + BDay(event_window_post)

        # Extract estimation-window data
        mask_est = (ff_factors.index >= est_start) & (ff_factors.index < est_end)
        ff_est = ff_factors.loc[mask_est]

        try:
            ret_est = daily_returns.loc[mask_est, gvkey]
        except KeyError:
            continue

        # Estimate betas
        betas = estimate_factor_betas(ret_est, ff_est)
        if betas is None:
            continue

        # Extract event-window data
        mask_evt = (ff_factors.index >= evt_start) & (ff_factors.index <= evt_end)
        ff_evt = ff_factors.loc[mask_evt]

        try:
            ret_evt = daily_returns.loc[mask_evt, gvkey]
        except KeyError:
            continue

        # Compute abnormal returns
        ar = compute_abnormal_returns(ret_evt, ff_evt, betas)
        car = ar.sum()

        result = {
            "gvkey": gvkey,
            "event_date": event_day,
            "CAR": car,
            "n_days_event": len(ar),
            "n_days_estimation": mask_est.sum(),
        }

        # Store daily ARs
        for i, (date, ar_val) in enumerate(ar.items()):
            result[f"AR_day_{i - event_window_pre}"] = ar_val

        results.append(result)

    return pd.DataFrame(results)


def compute_cav(
    litigation_events: pd.DataFrame,
    daily_volume: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    event_window_pre: int = 2,
    event_window_post: int = 2,
    baseline_window: Tuple[int, int] = (-40, -11),
    event_date_col: str = "First Litigation Event",
    firm_id_col: str = "gvkey_side_B",
) -> pd.DataFrame:
    """
    Compute Cumulative Abnormal Volume following Chae (2005).

    Abnormal turnover = log(turnover_t) - mean(log(turnover)) over baseline window.
    CAV = sum of abnormal turnover over the event window.

    Parameters
    ----------
    daily_volume : pd.DataFrame
        Daily trading volume, indexed by date with gvkey columns.
    shares_outstanding : pd.DataFrame
        Shares outstanding, aligned to daily_volume.
    baseline_window : tuple
        (start, end) business days relative to event for baseline mean.

    Returns
    -------
    pd.DataFrame
        Columns: gvkey, event_date, CAV.
    """
    results = []

    for idx, event in litigation_events.iterrows():
        event_day = pd.Timestamp(event[event_date_col])
        gvkey = event[firm_id_col]

        if pd.isna(gvkey):
            continue

        try:
            vol = daily_volume[gvkey]
            shares = shares_outstanding[gvkey]
        except KeyError:
            continue

        turnover = vol / shares
        log_turnover = np.log(turnover.replace(0, np.nan))

        # Baseline window
        bl_start = event_day + BDay(baseline_window[0])
        bl_end = event_day + BDay(baseline_window[1])
        baseline = log_turnover[(log_turnover.index >= bl_start) & (log_turnover.index <= bl_end)]
        baseline_mean = baseline.mean()

        # Event window
        evt_start = event_day - BDay(event_window_pre)
        evt_end = event_day + BDay(event_window_post)
        event_period = log_turnover[(log_turnover.index >= evt_start) & (log_turnover.index <= evt_end)]

        abnormal_turnover = event_period - baseline_mean
        cav = abnormal_turnover.sum()

        results.append({
            "gvkey": gvkey,
            "event_date": event_day,
            "CAV": cav,
        })

    return pd.DataFrame(results)
