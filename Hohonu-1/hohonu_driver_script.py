"""
DATA PIPELINE DEVELOPMENT
Version: Winter 2021
Putting together the pieces we've been working on winter-21 quarter.
"""

# Authors:
# MengChen Chung <mengchenc@uchicago.edu>
# Charlie Sheils <casheils@uchicago.edu>
# Jesica Maria Ramirez Toscano <jramireztoscano@uchicago.edu>

# Standard lib
from datetime import datetime, timedelta
from pathlib import Path
import re
from types import SimpleNamespace

# Third-party
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd


NODE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_node_name(value: str) -> str:
    """Return a safe local station identifier or fail before file lookup."""

    node_name = str(value).strip()
    if not NODE_NAME_PATTERN.fullmatch(node_name):
        raise ValueError(
            "node_name must be 1-64 characters using only letters, numbers, "
            "dot, underscore, or hyphen"
        )
    return node_name

# Locals
try:
    from dataproc import find_nearest_noaa_station as find_noaa
    from dataproc import TAD_algorithm as tad
    from dataproc import impute_hohonu_missing_data as impute
    from dataproc import VAR_prediction
    from dataproc import qartod_tests as qt
    from dataproc import noaa_stations
    from dataproc import utime
    from dataproc import dflib
except ModuleNotFoundError:
    # Lightweight fallbacks for working in the extracted `hohonu-1` directory.
    from noaa_stations import fetch_noaa_data as _fallback_fetch_noaa

    class _FallbackNoaaStations:
        @staticmethod
        def fetch_noaa_data(noaa_id, variable, begin, end, datum="metric"):
            return _fallback_fetch_noaa(noaa_id, variable, begin, end, datum)

    class _FallbackQartodTests:
        @staticmethod
        def _resample_node(df):
            if df.empty:
                return df
            df = df.copy()
            df = df.astype(float)
            return df.resample("6min").mean().ffill()

        @staticmethod
        def preprocess_node_data(node_name, begin, end):
            node_name = validate_node_name(node_name)
            candidates = [
                Path("./data") / f"{node_name}.csv",
                Path("./data") / f"{node_name}.tsv",
                Path("./data") / f"{node_name}_hohonu.csv",
            ]
            node_path = next((p for p in candidates if p.exists()), None)
            if node_path is None:
                raise FileNotFoundError(
                    f"No local Hohonu file found for node '{node_name}' in ./data."
                )

            if node_path.suffix == ".tsv":
                raw = pd.read_csv(node_path, sep="\t")
            else:
                raw = pd.read_csv(node_path)

            timestamp_col = next(
                (c for c in raw.columns if c.lower() in {"time", "timestamp", "datetime"}),
                None,
            )
            if timestamp_col is None:
                raise ValueError(
                    f"Could not find timestamp column in {node_path.name}. "
                    "Expected one of time/timestamp/datetime."
                )

            raw[timestamp_col] = pd.to_datetime(
                raw[timestamp_col], errors="coerce", utc=True
            )
            value_col = next((c for c in raw.columns if c != timestamp_col), raw.columns[1])
            raw = (
                raw.loc[:, [timestamp_col, value_col]]
                .set_index(timestamp_col)
                .sort_index()
            )
            begin_ts = pd.to_datetime(begin, unit="s", utc=True)
            end_ts = pd.to_datetime(end, unit="s", utc=True)
            raw = raw[(raw.index >= begin_ts) & (raw.index <= end_ts)]
            raw.columns = [node_name]
            return _FallbackQartodTests._resample_node(raw)

        @staticmethod
        def basic_test(combined_df, node_name, noaa_name):
            node = combined_df[node_name].copy()
            med = node.median()
            mad = (node - med).abs().median()
            if mad == 0 or pd.isna(mad):
                return node
            return node.where((node - med).abs() < 8.0 * mad, other=np.nan)

        @staticmethod
        def advanced_test(combined_df, clean_col, noaa_name):
            s = combined_df[clean_col].copy()
            return s.rolling(12, min_periods=1).median()

    class _FallbackFindNearest:
        @staticmethod
        def run():
            from load_hohonu_devices import run as _load_devices

            _load_devices()

    class _FallbackTAD:
        @staticmethod
        def get_local_mllw_TAD(hohonu_data, noaa_data):
            both = pd.concat([hohonu_data, noaa_data], axis=1).dropna()
            if both.empty:
                return 0.0
            return float((both.iloc[:, 0] - both.iloc[:, 1]).median())

    class _FallbackUTime:
        @staticmethod
        def get_monthly_timestamp_interval(number_of_months=1, offset_days_backward=0):
            end = datetime.now() - timedelta(days=offset_days_backward)
            begin = end - timedelta(days=30 * float(number_of_months))
            return int(begin.timestamp()), int(end.timestamp())

        @staticmethod
        def get_timestamp_interval_from_iso(begin_str, end_str):
            return (
                int(pd.to_datetime(begin_str).timestamp()),
                int(pd.to_datetime(end_str).timestamp()),
            )

    class _FallbackImputer:
        @staticmethod
        def fit_transform(data):
            from sklearn.impute import IterativeImputer

            return IterativeImputer(random_state=0).fit_transform(data)

    class _FallbackVARPrediction:
        @staticmethod
        def predict_water_level(combined_data, steps=960, rmse_threshold=0.3):
            from VAR_prediction import predict_water_level

            return predict_water_level(combined_data, steps=steps, rmse_threshold=rmse_threshold)

    find_noaa = _FallbackFindNearest()
    tad = _FallbackTAD()
    impute = _FallbackImputer
    VAR_prediction = _FallbackVARPrediction
    qt = _FallbackQartodTests()
    noaa_stations = _FallbackNoaaStations()
    utime = _FallbackUTime()
    dflib = SimpleNamespace()  # placeholder for legacy imports

from tide_ml_engine import (
    MODEL_FAMILY_HELP_TEXT,
    get_default_candidate_grid,
    run_auto_ml_search,
)


def _coerce_candidate_families(raw):
    if raw is None:
        return None

    tokens = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        for token in text.replace(" ", ",").split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens if tokens else None


def _read_device_table():
    for path in [Path("./data/hohonu_stations.tsv"), Path("../data/hohonu_stations.tsv")]:
        if path.exists():
            return pd.read_csv(path, sep="\t")
    raise FileNotFoundError("Could not find hohonu_stations.tsv in ./data or ../data")


def get_nearestNoaa_by_node(node_name, begin, end, datum="metric"):
    """
    Helper Function for Step 2
    This function gets the nearest noaa station timeseries data in standard datum, GMT, and meters
    Inputs:
        (string) node_name: hohonu station's id
        (float) begin, end: beginning/ending time for data collection,
            e.g. begin, end = utime.get_timestamp_interval_from_iso('2020-12-01','2020-12-31')
    Outputs:
        (panda dataframe) the nearest noaa station timeseries data to the input hohonu station,
        in standard datum, GMT, and meters
    """

    device = _read_device_table()
    matching_nodes = device.loc[
        device["DeviceId (string)"] == node_name, "N-NoaaStation (int)"
    ]
    if matching_nodes.empty:
        raise ValueError(
            f"Hohonu node '{node_name}' is not present in ./data/hohonu_stations.tsv."
        )

    nearest_noaa_station = matching_nodes.iloc[0]
    if pd.isna(nearest_noaa_station):
        raise ValueError(
            f"The nearest NOAA station for Hohonu node '{node_name}' is missing."
        )

    nearest_noaa_station = int(nearest_noaa_station)
    print("The nearest NOAA station is", nearest_noaa_station)
    return noaa_stations.fetch_noaa_data(
        nearest_noaa_station, "water_level", begin, end, datum
    )


def get_mllw_scalar(node_name, hohonu_data, noaa_data):
    """
    Helper Function for Step 5: Convert to MLLW standard.
    Tries to get local MLLW datum information for a Hohonu node,
    if Device file doesn't have local_mllw for a hohonu node, then we run TAD and record MLLW in the database
    Inputs:
        -node_name (string): Name of the Hohonu node
        -hohonu_data (pd.DataFrame): DataFrame with the Water Levels of the Hohonu Station
        -noaa_data (pd.DataFrame): DataFrame with the Water Levels of the Nearest NOAA Station
    Outpus:
        -local_mllw (float): Local MLLW scalar to convert local water measurements of the Hohonu Station
    """

    device = pd.read_csv("./data/hohonu_stations.tsv", sep="\t")
    matches = device[device["DeviceId (string)"] == node_name]
    if matches.empty:
        raise ValueError(
            f"Hohonu node '{node_name}' is not present in ./data/hohonu_stations.tsv."
        )

    index_num = matches.index[0]
    local_mllw = matches["Local-MLLW (float)"].iloc[0]

    if pd.isna(local_mllw):

        # Run TAD
        print("No local MLLW found >> Running TAD algorithm")
        local_mllw = tad.get_local_mllw_TAD(hohonu_data, noaa_data)

        # Save Data
        print("Saving data in ./data/hohonu_stations.tsv and Device.json")
        device.loc[index_num, "Local-MLLW (float)"] = local_mllw
        # Persist the updated station table back to whichever local file exists.
        saved = False
        for target in [Path("./data/hohonu_stations.tsv"), Path("../data/hohonu_stations.tsv")]:
            if target.exists():
                device.to_csv(target, sep="\t", index=False)
                saved = True
                break
        if not saved:
            Path("./data").mkdir(parents=True, exist_ok=True)
            device.to_csv("./data/hohonu_stations.tsv", sep="\t", index=False)
        device["DeviceId (string)2"] = device["DeviceId (string)"]

        # device.set_index('DeviceId (string)2').to_json(
        #     './data/Device.json', orient='index', indent=2, force_ascii=False)

    return local_mllw


def impute_cleaned_data(noaa_hohonu):
    """
    Returns Pandas dataframe with missing values imputed from
    nearest NOAA station.
    Assumes that hohonu_df has been resampled to every 6 minutes
    Input:
    -noaa_hohonu: A two column dataframe with NOAA data first and Cleaned Hohonu Data second.
    Output:
    -merged_df_imputed[:, 1]: Hohonu Series with missing values imputed
    """
   
    # noaa_hohonu[:200]['node-10027'].replace({pd.NaT: None})

    # print(noaa_hohonu[:200]['node-10027'].tolist(), flush=True)

    # print(noaa_hohonu, flush=True)

    # pd.set_option('display.max_rows', noaa_hohonu.shape[0]+1)
    # print(noaa_hohonu[:200], flush=True)

    # Fit the imputer and impute the missing valu   es
    imp_mean = IterativeImputer(random_state=0)

    merged_df_imputed = pd.DataFrame(
        imp_mean.fit_transform(noaa_hohonu),
        index=noaa_hohonu.index,
        columns=noaa_hohonu.columns,
    )

    hohonu_imputed = merged_df_imputed.iloc[:, 1]

    # Return Hohonu Data column
    return hohonu_imputed


def _prepare_combined_series(node_name, number_of_months=8, offset_days_backward=20):
    """
    Shared pre-processing used by both VAR and ML branches.
    """
    # Step 0: Set begin and end dates to fetch
    begin, end = utime.get_monthly_timestamp_interval(
        number_of_months=number_of_months, offset_days_backward=offset_days_backward
    )

    print("Beginning Time:", datetime.fromtimestamp(begin))
    print("Ending Time:", datetime.fromtimestamp(end))

    # Step 1: Fetch data from requested node
    node_data = qt.preprocess_node_data(node_name, begin, end)

    # Step 2: Fetch data from nearest NOAA station
    try:
        noaa_data = get_nearestNoaa_by_node(node_name, begin, end, "metric")
    except (KeyError, ValueError, IndexError) as exc:
        # If it fails calculate the nearest NOAA for that node and record it
        print(
            f"Could not retrieve NOAA data for node '{node_name}': {exc}. "
            "Recomputing nearest stations and retrying."
        )
        find_noaa.run()
        noaa_data = get_nearestNoaa_by_node(node_name, begin, end, "metric")

    noaa_name = noaa_data.columns[0]

    # Step 2.5: Merge NOAA and Hohonu data
    noaa_data = noaa_data[
        ~noaa_data.index.duplicated(keep="first")
    ]  # Remove duplication in NOAA
    combined_df = node_data.merge(
        noaa_data, how="outer", left_index=True, right_index=True, sort=True
    )
    if combined_df.empty:
        raise ValueError(
            f"No merged observations found for node '{node_name}' in selected date range."
        )

    # Step 3: Run basic qartod tests on the hohonu node and flag bad data
    combined_df["clean_node"] = qt.basic_test(combined_df, node_name, noaa_name)
    combined_df.loc[combined_df.index[0], "clean_node"] = combined_df.loc[
        combined_df.index[0], node_name
    ]

    # Step 4: Run TAD Algorithm and Convert to MLLW standard
    estimated_mllw_scalar = get_mllw_scalar(
        node_name, combined_df[["clean_node"]], combined_df[[noaa_name]]
    )
    combined_df["clean_node"] = -combined_df["clean_node"] - estimated_mllw_scalar

    # Step 5: Get both the NOAA and hohonu data in MLLW for the last three months
    new_begin = datetime.now() - timedelta(days=111)
    combined_df = combined_df[combined_df.index > new_begin]

    # Step 6: Run advanced qartod tests on the hohonu node and flag bad data
    combined_df["clean_node"] = qt.advanced_test(combined_df, "clean_node", noaa_name)
    combined_df.loc[combined_df.index[0], "clean_node"] = combined_df.loc[
        combined_df.index[0], node_name
    ]

    # Step 7: Impute missing data
    combined_df["clean_node"] = impute_cleaned_data(combined_df.iloc[:, 1:])

    # Step 8: Format output
    combined_df.index = pd.DatetimeIndex(combined_df.index).to_period("min")
    combined_df.drop(columns=[node_name], inplace=True)
    return combined_df, noaa_name


def fetch_predictions(
    node_name,
    model_strategy="var",
    steps=960,
    use_digital_twin=True,
    return_metadata=False,
    ensemble_size=3,
    include_lstm=False,
    include_pinn=False,
    candidate_grid=None,
    meta_top_k=4,
    meta_holdout_ratio=0.2,
    candidate_profile="compact",
    candidate_model_families=None,
    candidate_mix_max_size: int = 4,
):
    """
    Data Processing + prediction pipeline.

    model_strategy:
        - "var": legacy VAR pipeline
        - "auto", "auto-ml", "ensemble", "meta", "mix", "ml": auto-ML variants
    """
    combined_df, _ = _prepare_combined_series(node_name)

    if model_strategy in {"var", "var-baseline"}:
        prediction = VAR_prediction.predict_water_level(combined_df, steps=steps)
        if return_metadata:
            return {
                "prediction": prediction,
                "model_name": "VAR",
                "selected_rmse": None,
                "selected_mae": None,
                "selected_mape": None,
                "selected_r2": None,
                "selected_corr": None,
                "selected_mpe": None,
                "selected_me": None,
                "selected_minmax": None,
                "selected_nse": None,
                "selected_qa_score": None,
                "scores": [],
                "digital_twin_used": False,
            }
        return prediction

    if candidate_grid is None and model_strategy in {
        "auto",
        "auto-ml",
        "ml",
        "ensemble",
        "meta",
        "mix",
    }:
        candidate_grid = get_default_candidate_grid(
            include_lstm=bool(include_lstm),
            include_pinn=bool(include_pinn),
            profile=candidate_profile,
        )

    if model_strategy in {"auto", "auto-ml", "ml"}:
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="best",
            candidate_grid=candidate_grid,
            candidate_model_families=candidate_model_families,
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    if model_strategy == "ensemble":
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="ensemble",
            ensemble_size=ensemble_size,
            candidate_grid=candidate_grid,
            candidate_model_families=candidate_model_families,
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    if model_strategy == "meta":
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="meta",
            candidate_grid=candidate_grid,
            meta_top_k=meta_top_k,
            meta_holdout_ratio=meta_holdout_ratio,
            candidate_model_families=candidate_model_families,
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    if model_strategy == "mix":
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="mix",
            candidate_grid=candidate_grid,
            candidate_model_families=candidate_model_families,
            candidate_mix_max_size=max(2, candidate_mix_max_size),
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    raise ValueError(
        "model_strategy must be one of: 'var', 'auto', 'auto-ml', 'ensemble', 'meta', 'mix', 'ml'."
    )


def predictions(
    node_name,
    noaa_name,
    combined_df,
    model_strategy="var",
    steps=960,
    use_digital_twin=True,
    return_metadata=False,
    ensemble_size=3,
    include_lstm=False,
    include_pinn=False,
    candidate_grid=None,
    meta_top_k=4,
    meta_holdout_ratio=0.2,
    candidate_profile="compact",
    candidate_model_families=None,
    candidate_mix_max_size: int = 4,
):
    combined_df["clean_node"] = combined_df[node_name]
    combined_df["clean_node"] = impute_cleaned_data(combined_df.iloc[:, 1:])
    combined_df.index = pd.DatetimeIndex(combined_df.index).to_period("min")
    combined_df.drop(columns=[node_name], inplace=True)

    if model_strategy in {"var", "var-baseline"}:
        prediction = VAR_prediction.predict_water_level(combined_df, steps=steps)
        if return_metadata:
            return {
                "prediction": prediction,
                "model_name": "VAR",
                "selected_rmse": None,
                "selected_mae": None,
                "selected_mape": None,
                "selected_r2": None,
                "selected_corr": None,
                "selected_mpe": None,
                "selected_me": None,
                "selected_minmax": None,
                "selected_nse": None,
                "selected_qa_score": None,
                "scores": [],
                "digital_twin_used": False,
            }
        return prediction

    if candidate_grid is None and model_strategy in {
        "auto",
        "auto-ml",
        "ml",
        "ensemble",
        "meta",
        "mix",
    }:
        candidate_grid = get_default_candidate_grid(
            include_lstm=bool(include_lstm),
            include_pinn=bool(include_pinn),
            profile=candidate_profile,
        )

    if model_strategy in {"auto", "auto-ml", "ml"}:
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="best",
            candidate_grid=candidate_grid,
            candidate_model_families=candidate_model_families,
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    if model_strategy == "ensemble":
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="ensemble",
            ensemble_size=ensemble_size,
            candidate_grid=candidate_grid,
            candidate_model_families=candidate_model_families,
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    if model_strategy == "meta":
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="meta",
            candidate_grid=candidate_grid,
            meta_top_k=meta_top_k,
            meta_holdout_ratio=meta_holdout_ratio,
            candidate_model_families=candidate_model_families,
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    if model_strategy == "mix":
        result = run_auto_ml_search(
            combined_df,
            steps=steps,
            use_digital_twin=use_digital_twin,
            strategy="mix",
            candidate_grid=candidate_grid,
            candidate_model_families=candidate_model_families,
            candidate_mix_max_size=max(2, candidate_mix_max_size),
        )
        if return_metadata:
            result = dict(result)
            result["prediction"] = result["forecast"]
            result["candidate_profile"] = candidate_profile
            result["candidate_model_families"] = candidate_model_families
            return result
        return result["forecast"]

    raise ValueError(
        "model_strategy must be one of: 'var', 'auto', 'auto-ml', 'ensemble', 'meta', 'mix', 'ml'."
    )


def fetch_predictions_multiple_nodes(
    node_array,
    model_strategy="var",
    steps=960,
    use_digital_twin=True,
    ensemble_size=3,
    include_lstm=False,
    include_pinn=False,
    candidate_grid=None,
    meta_top_k=4,
    meta_holdout_ratio=0.2,
    candidate_profile="compact",
    candidate_model_families=None,
):
    """
    Fetch predictions on multiple Hohonu stations and NOAA stations.

    Input:
        -(array-like): the array of Hohonu names to be predicted

    Output:
        -(array-like): 4-day predicted values (960 obs) in all imput Hohonu stations and their Nearest NOAA data
    """

    prediction_list = []
    for node in node_array:
        prediction_list.append(
            fetch_predictions(
                node,
                model_strategy=model_strategy,
                steps=steps,
                use_digital_twin=use_digital_twin,
                ensemble_size=ensemble_size,
                candidate_profile=candidate_profile,
                meta_top_k=meta_top_k,
                meta_holdout_ratio=meta_holdout_ratio,
                include_lstm=include_lstm,
                include_pinn=include_pinn,
                candidate_grid=candidate_grid,
                candidate_model_families=candidate_model_families,
            )
        )
    return prediction_list


def run_pipeline(
    node_name: str,
    model_strategy: str = "auto",
    steps: int = 960,
    use_digital_twin: bool = True,
    ensemble_size: int = 3,
    return_metadata: bool = False,
    include_lstm: bool = False,
    include_pinn: bool = False,
    meta_top_k: int = 4,
    meta_holdout_ratio: float = 0.2,
    candidate_profile: str = "compact",
    candidate_model_families=None,
    candidate_mix_max_size: int = 4,
):
    """Single entrypoint for generating a complete prediction for one node."""
    node_name = validate_node_name(node_name)
    return fetch_predictions(
        node_name=node_name,
        model_strategy=model_strategy,
        steps=steps,
        use_digital_twin=use_digital_twin,
        return_metadata=return_metadata,
        ensemble_size=ensemble_size,
        candidate_profile=candidate_profile,
        meta_top_k=meta_top_k,
        meta_holdout_ratio=meta_holdout_ratio,
        include_lstm=include_lstm,
        include_pinn=include_pinn,
        candidate_model_families=candidate_model_families,
        candidate_mix_max_size=candidate_mix_max_size,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Hohonu tide prediction pipeline."
    )
    parser.add_argument("node_name", help="Hohonu node id")
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["var", "auto", "auto-ml", "ensemble", "meta", "mix", "ml"],
        help="Prediction strategy",
    )
    parser.add_argument(
        "--steps", type=int, default=960, help="Number of 6-minute steps to forecast"
    )
    parser.add_argument(
        "--digital-twin",
        dest="use_digital_twin",
        action="store_true",
        default=True,
        help="Enable NOAA-based digital twin augmentation",
    )
    parser.add_argument(
        "--no-digital-twin",
        dest="use_digital_twin",
        action="store_false",
        help="Disable NOAA-based digital twin augmentation",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Number of top models to combine when using strategy=ensemble",
    )
    parser.add_argument(
        "--candidate-profile",
        type=str,
        default="compact",
        choices=["compact", "auto", "broad"],
        help="Candidate search profile: compact, auto (subsampled broad), or broad",
    )
    parser.add_argument(
        "--model-families",
        type=str,
        nargs="*",
        default=None,
        help=MODEL_FAMILY_HELP_TEXT,
    )
    parser.add_argument(
        "--mix-size",
        type=int,
        default=4,
        help="Max number of models to include in mix strategy.",
    )
    parser.add_argument(
        "--meta-top-k",
        type=int,
        default=4,
        help="Number of top candidates used by meta-stacker strategy",
    )
    parser.add_argument(
        "--meta-holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for fitting meta-stacker",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable LSTM candidates in auto/ml search.",
    )
    parser.add_argument(
        "--use-pinn",
        action="store_true",
        help="Enable PINN-like candidates in auto/ml search.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Print JSON-like metadata about selected model and score",
    )

    args = parser.parse_args()

    model_families = _coerce_candidate_families(args.model_families)

    output = run_pipeline(
        node_name=args.node_name,
        model_strategy=args.strategy,
        steps=args.steps,
        use_digital_twin=args.use_digital_twin,
        ensemble_size=args.ensemble_size,
        candidate_profile=args.candidate_profile,
        meta_top_k=args.meta_top_k,
        meta_holdout_ratio=args.meta_holdout_ratio,
        return_metadata=args.metadata,
        include_lstm=args.use_lstm,
        include_pinn=args.use_pinn,
        candidate_model_families=model_families,
        candidate_mix_max_size=args.mix_size,
    )
    print(output)
