"""
Script for loading the Hohonu station data from nodes.json and
nodes_standardized.json into Device.tsv and Device.json.
Author: Charlie Sheils (casheils@uchicago.edu)
Last Updated: Feb. 17, 2021
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from timezonefinder import TimezoneFinder

try:
    from find_nearest_noaa_station import find_nearest_noaa_station
except ModuleNotFoundError:
    def find_nearest_noaa_station(
        latitude, longitude, active_noaa_df, top_n: int = 1
    ):
        '''
        Fallback nearest NOAA lookup when external helper module is unavailable.
        '''
        if (
            pd.isna(latitude)
            or pd.isna(longitude)
            or active_noaa_df is None
            or active_noaa_df.empty
        ):
            return None

        lat_col = next(
            (
                col
                for col in [
                    "latitude",
                    "Latitude (float)",
                    "Lat",
                    "lat",
                    "latitude_dd",
                ]
                if col in active_noaa_df.columns
            ),
            None,
        )
        lon_col = next(
            (
                col
                for col in [
                    "longitude",
                    "Longitude (float)",
                    "Lon",
                    "lon",
                    "longitude_dd",
                ]
                if col in active_noaa_df.columns
            ),
            None,
        )
        station_col = next(
            (
                col
                for col in [
                    "station_id",
                    "stationId",
                    "id",
                    "ID",
                    "StationId",
                ]
                if col in active_noaa_df.columns
            ),
            None,
        )
        if lat_col is None or lon_col is None or station_col is None:
            return None

        active_noaa_df = active_noaa_df.copy()
        active_noaa_df["distance_km"] = (
            (
                (active_noaa_df[lat_col].astype(float) - float(latitude)) ** 2
                + (active_noaa_df[lon_col].astype(float) - float(longitude)) ** 2
            )
            ** 0.5
            * 111.0
        )
        nearest = active_noaa_df.sort_values("distance_km")[station_col].iloc[:top_n]
        return nearest.astype("Int64").iloc[0]

def _read_data_file(filenames):
    candidates = [Path("./data"), Path("../data")]
    for name in filenames:
        for root in candidates:
            path = root / name
            if path.exists():
                with open(path, "r") as f:
                    return path, json.load(f)
    raise FileNotFoundError(f"Could not find any of: {', '.join(filenames)} in ./data or ../data")


def combine_nodes_json_data(nodes, nodes_standardized):
    '''
    Combine data from nodes.json and nodes_standardized.json (loaded as
    dictionaries) into one nested dictionary.

    Inputs:
        nodes: dictionary from loading nodes.json
        nodes_standardized: dictionary from loading nodes_standardized.json
    '''
    tf = TimezoneFinder()

    # Create nested dictionary
    rv = {}
    for node in nodes_standardized.keys():
        if 'noaa' in node: # Skip over NOAA nodes
            continue
        rv[node] = {}
        rv[node]['DeviceId (string)'] = node
        rv[node]['UserId (string)'] = None
        rv[node]['Timetype (string)'] = nodes_standardized[node]['time_type']
        rv[node]['Cellular (boolean)'] = nodes_standardized[node]['cellular']

        # Latitude (Y) and longitude (X) are flipped in nodes_standardized.json
        latitude, longitude = nodes_standardized[node]['x'], \
            nodes_standardized[node]['y']
        if (latitude is None) or (longitude is None):
            localTimezone = None
        else:
            localTimezone = tf.timezone_at(lng=longitude, lat=latitude)

        rv[node]['Local-Timezone (string)'] = localTimezone
        rv[node]['State (string)'] = nodes_standardized[node]['location']
        rv[node]['Latitude (float)'] = latitude
        rv[node]['Longitude (float)'] = longitude
        # Keep legacy keys for any downstream code still expecting them
        rv[node]['latitude'] = latitude
        rv[node]['longitude'] = longitude
        navd88 = nodes[node]['navd88']
        if navd88 is not None and not pd.isna(navd88):
            rv[node]['NAVD88 (float)'] = navd88*0.3048 # Convert to meters
        else:
            rv[node]['NAVD88 (float)'] = None

        rv[node]['Local-MLLW (float)'] = None
        rv[node]['N-NoaaStation (int)'] = None

    return rv


def run():
    '''
    Converts data from nodes.json and nodes_standardized.json into Device.tsv
    and Device.json, adding field for nearest NOAA station (minimum Euclidean
    distance).
    '''
    # Load & combine Hohonu nodes data
    _, NODES = _read_data_file(["nodes.json"])
    _, NODES_STANDARDIZED = _read_data_file(["nodes_standardized.json"])
    node_dict_rvsd = combine_nodes_json_data(NODES, NODES_STANDARDIZED)
    device_df = pd.DataFrame.from_dict(node_dict_rvsd, orient="index")

    # Read in NOAA stations
    try:
        noaa_path, _ = _read_data_file(["noaa_stations.tsv"])
        noaa_stations_df = pd.read_csv(noaa_path, sep="\t")
    except FileNotFoundError:
        noaa_stations_df = pd.read_csv("../data/noaa_stations.tsv", sep="\t")

    # Filter NOAA stations to those that are still active (not removed)
    active_noaa_df = noaa_stations_df[noaa_stations_df['removed'].isna()]

    # Add nearest_noaa_station column to Device dataframe
    device_df['N-NoaaStation (int)'] = device_df.apply(lambda row:\
        find_nearest_noaa_station(row['Latitude (float)'],
                                  row['Longitude (float)'],
                                  active_noaa_df, 1), axis=1)
    # Cast column back to integer datatype
    device_df['N-NoaaStation (int)'] = pd.Series(
        device_df['N-NoaaStation (int)'], dtype='Int64')


    # Export updated TSV file
    device_df.to_csv("./data/hohonu_stations.tsv", sep='\t', index=False)

    # Export updated JSON file
    # device_df['DeviceId (string)2'] = device_df['DeviceId (string)']
    # device_df.set_index('DeviceId (string)2').to_json(
    #     './data/Device.json',
    #     orient='index',
    #     indent=2,
    #     force_ascii=False)


if __name__ == "__main__":
    run()
