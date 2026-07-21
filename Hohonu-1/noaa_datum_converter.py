# Authors: MengChen Chung <mengchenc@uchicago.edu>

from __future__ import annotations

'''
Functions converting noaa time series data to different datums.
'''

# Third-party
import pandas as pd
from pathlib import Path

# Locals
try:
    import noaa_stations
except ModuleNotFoundError:
    class _FallbackNoaaStations:
        @staticmethod
        def fetch_noaa_data(noaa_id, product, begin, end, units="metric"):
            raise KeyError(
                "noaa_stations module is missing and no local NOAA cache is available."
            )

    noaa_stations = _FallbackNoaaStations()

try:
    import utime
except ModuleNotFoundError:
    utime = None
    
try:
    import hohonu_stations
except ModuleNotFoundError:
    hohonu_stations = None



def _locate_noaa_stations_file():
    for candidate in [Path("./data/noaa_stations.tsv"), Path("../data/noaa_stations.tsv")]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find noaa_stations.tsv in ./data or ../data")


def _safe_station_scalar(noaa_id: int, datum_key: str):
    '''
    Returns datum scalar for a NOAA station id.
    '''
    noaa_stations_df = pd.read_csv(_locate_noaa_stations_file(), sep="\t")
    scalar = noaa_stations_df.loc[
        noaa_stations_df['station_id'].astype(str) == str(noaa_id), datum_key
    ]
    if scalar.empty:
        print(f"No NOAA station row found for station id {noaa_id}.")
        return None

    scalar_value = scalar.values[0]
    if pd.isna(scalar_value):
        print(f"This NOAA station does not contain {datum_key} datum.")
        return None

    return scalar_value


def noaa_stnd_to_navd88(
    noaa_id: str, begin: float, end: float, units: str = "metric"
) -> pd.DataFrame | None:
    
    '''
    This function convert noaa time series data from stnd to navd88
    
    Inputs:
        (string) noaa_id
        (float) begin, end: beginning/ending time for data collection,
                e.g. begin, end = utime.get_timestamp_interval_from_iso('2020-12-01','2020-12-31')
        (string) units: the measurement unit, can be either 'metric' or 'english'
    Outputs:
        (pandas.DataFrame) noaa time series data
    '''
    # examine the station's availability
    try:
        noaa_data = noaa_stations.fetch_noaa_data(noaa_id, 'water_level', begin, end, units = units)
    except KeyError:
        print("This noaa station is not available.")
        return
    
    # get noaa datum scalar
    try:
        noaa_id = int(noaa_id)
    except (TypeError, ValueError):
        print("Invalid NOAA station id provided for NAVD88 conversion.")
        return

    navd88_scalar = _safe_station_scalar(noaa_id, 'NAVD88')
    if navd88_scalar is None:
        return
    
    # determine the unit and make the conversion
    if (units == 'metric'):
        return noaa_data - navd88_scalar
    elif units == 'english':
        return noaa_data - navd88_scalar * 3.28084
    else:
        print("Invalid units value. Use 'metric' or 'english'.")
        return


def noaa_stnd_to_mllw(
    noaa_id: str, begin: float, end: float, units: str = "metric"
) -> pd.DataFrame | None:
    
    '''
    This function convert noaa time series data from stnd to mllw
    
    Inputs:
        (string) noaa_id
        (float) begin, end: beginning/ending time for data collection,
                e.g. begin, end = utime.get_timestamp_interval_from_iso('2020-12-01','2020-12-31')
        (string) units: the measurement unit, can be either 'metric' or 'english'
    Outputs:
        (pandas.DataFrame) noaa time series data
    '''
    # examine the station's availability
    try:
        noaa_data = noaa_stations.fetch_noaa_data(noaa_id, 'water_level', begin, end, units = units)
    except KeyError:
        print("This noaa station is not available.")
        return
    
    # get noaa datum scalar
    try:
        noaa_id = int(noaa_id)
    except (TypeError, ValueError):
        print("Invalid NOAA station id provided for MLLW conversion.")
        return

    mllw_scalar = _safe_station_scalar(noaa_id, 'MLLW')
    if mllw_scalar is None:
        return
    
    # determine the unit and make the conversion
    if (units == 'metric'):
        return noaa_data - mllw_scalar
    elif units == 'english':
        return noaa_data - mllw_scalar * 3.28084
    else:
        print("Invalid units value. Use 'metric' or 'english'.")
        return
