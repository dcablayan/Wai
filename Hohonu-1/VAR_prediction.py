# Authors:
# MengChen Chung <mengchenc@uchicago.edu>

"""
VAR model for predicting water levels in Hohonu and NOAA data, 
including some time series data examinations.
Some of the functions included come from:
https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
"""

# Standard lib
import warnings

# Third-party
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


def grangers_causation_matrix(
    data, variables, test="ssr_chi2test", maxlag=12, verbose=False
):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    Input:
        -(pd.DataFrame) data: time series data
        -(list) :names of the time series variables

    Output:
        -(pd.DataFrame) grangers causation outcome
    """

    df = pd.DataFrame(
        np.zeros((len(variables), len(variables))), columns=variables, index=variables
    )
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(
                data[[r, c]], maxlag=maxlag, verbose=False
            )
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f"Y = {r}, X = {c}, P Values = {p_values}")
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + "_x" for var in variables]
    df.index = [var + "_y" for var in variables]
    return df


def adfuller_test(series, signif=0.05, name="", verbose=False):
    """
    Perform ADFuller to test for Stationarity of given series and print report

    Input:
        -(pd.series) series: water level data of the Hohonu station or NOAA station
        -(float) signif: significant level
        -(string) name: station's name for testing
    """

    r = adfuller(series, autolag="AIC")
    output = {
        "test_statistic": round(r[0], 4),
        "pvalue": round(r[1], 4),
        "n_lags": round(r[2], 4),
        "n_obs": r[3],
    }
    p_value = output["pvalue"]

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", "-" * 47)
    print(f" Null Hypothesis: Data has unit root. Non-Stationary.")
    print(f" Significance Level    = {signif}")
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f" Critical value {adjust(key)} = {round(val, 3)}")

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


def forecast_accuracy(forecast, actual):
    """
    Print out the performance of forecast values`

    Input:
        -(np.array) forecast: predicted water level data of the Hohonu station or NOAA station
        -(np.array) actual: true water level data of the Hohonu station or NOAA station

    Output:
        -(dictionary) accuracy on different criteria
    """
    forecast = np.asarray(forecast).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    if forecast.shape != actual.shape:
        raise ValueError("Forecast and actual arrays must have the same shape")

    diff = forecast - actual
    mape = (
        np.mean(np.abs(diff))
        / np.mean(np.abs(actual))
        if not np.isclose(np.mean(np.abs(actual)), 0)
        else np.nan
    )  # MAPE
    me = np.mean(diff)  # ME
    mae = np.mean(np.abs(diff))  # MAE
    mpe = np.mean(diff) / np.mean(actual) if not np.isclose(np.mean(actual), 0) else np.nan  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** 0.5  # RMSE
    actual_var = np.sum((actual - np.mean(actual)) ** 2)
    r_square = 1 - (np.sum(diff ** 2) / actual_var) if actual_var != 0 else np.nan
    nse = (
        1 - (np.sum(diff ** 2) / actual_var)
        if not np.isclose(actual_var, 0)
        else np.nan
    )
    corr = (
        np.corrcoef(forecast, actual)[0, 1]
        if np.std(forecast) != 0 and np.std(actual) != 0
        else np.nan
    )  # corr
    mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    ratio = np.divide(mins, maxs, out=np.zeros_like(mins, dtype=float), where=maxs != 0)
    minmax = 1 - np.mean(ratio) if np.any(maxs != 0) else np.nan  # minmax
    return {
        "mape": mape,
        "me": me,
        "mae": mae,
        "mpe": mpe,
        "rmse": rmse,
        "r_square": r_square,
        "corr": corr,
        "nse": nse,
        "minmax": minmax,
    }


def predict_water_level(combined_data, steps=960, rmse_threshold=0.3):
    """
    Predict water level data with VAR model.
    Note: in 6 min time interval, 1 day = 240 obs, 1 mon (30 days) = 7200 obs.

    Input:
        -(pd.DataFrame) combined_data: water level data with at least 2 variables/columns
                                       need to be in period
        -(int) steps: the number of observations to be predicted
        -(float) rmse_threshold: the expected rmse performance

    Output:
        -(np.arrary) prediction: 2D array with predicted values for all variables/columns
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    minimum_rows = 7200 + int(steps)
    if len(combined_data.index) < minimum_rows:
        print(
            "Not enough data for prediction. "
            f"Require at least one month plus a {steps}-row validation holdout "
            f"({minimum_rows} rows)."
        )
        return
    elif len(combined_data.index) < 21600:
        warnings.warn("More than 3 months data are strongly suggested for prediction!")
    # Select the training-window length on a true forward holdout.  The legacy
    # implementation trained through the end of the series, forecast beyond
    # it, and compared that forecast with rows from the beginning of the same
    # training window.  That number was not an out-of-sample error estimate.
    lag = 248  # every 24 hrs and 50 mins the moon goes back to the same position, and creates the same tides
    validation_start = len(combined_data) - int(steps)
    validation_actual = combined_data.iloc[validation_start:].to_numpy(dtype=float)
    max_extra_days = min(60, max(0, (validation_start - 7200) // 240))
    candidate_days = range(0, max_extra_days + 1)
    selected_window = None
    selected_rmses = None
    selected_score = float("inf")

    for extra_days in candidate_days:
        window = 7200 + 240 * extra_days
        train_data = combined_data.iloc[validation_start - window : validation_start]
        try:
            model_fitted = VAR(train_data).fit(lag)
            holdout_forecast = model_fitted.forecast(
                y=train_data.values[-lag:], steps=steps
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        rmses = np.sqrt(
            np.mean((holdout_forecast[:, :2] - validation_actual[:, :2]) ** 2, axis=0)
        )
        score = float(np.mean(rmses))
        if score < selected_score:
            selected_score = score
            selected_window = window
            selected_rmses = rmses
        if np.all(rmses <= rmse_threshold):
            break

    if selected_window is None or selected_rmses is None:
        print("Unable to fit a finite VAR model on any validation window.")
        return

    # Refit the selected configuration through the latest observation, then
    # issue the actual future forecast requested by the caller.
    final_train = combined_data.iloc[-selected_window:]
    try:
        final_model = VAR(final_train).fit(lag)
        selected_fc = final_model.forecast(y=final_train.values[-lag:], steps=steps)
    except (ValueError, np.linalg.LinAlgError):
        print("Unable to refit the selected VAR model on all available data.")
        return

    extra_days = (selected_window - 7200) // 240
    selected_RMSE_1 = float(selected_rmses[0])
    selected_RMSE_2 = float(selected_rmses[1])
    print("extra days for model training:", extra_days)
    print("forward-holdout RMSE in 1st column:", selected_RMSE_1)
    print("forward-holdout RMSE in 2nd column:", selected_RMSE_2)
    return selected_fc
