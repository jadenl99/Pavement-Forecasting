import pandas as pd
import geopandas as gpd
import numpy as np
import math
import geopy.distance
def data_by_MM(
        df: pd.DataFrame, mile_col: str, 
        start_marker: int, end_marker: int, 
        reverse_count: bool=False):
    """Creates a dictionary with the key as the MM and the value as a dataframe of the data for that MM

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        mile_col (str): column name of where the position of the start of the slabs are stored
        start_marker (int): starting point of the milemarkers
        end_marker (int): ending point of the milemarkers
        reverse_count (bool, optional): Whether to count down by MM or not. Counting down can be useful if the interstate goes westbound or southbound. Defaults to False.

    Raises:
        Exception: if milemarkers are negative
        Exception: if milemarkers are not integers

    Returns:
        dict: dictionary of dataframes with the milemarker as the key
    """
    if start_marker < 0 or end_marker < 0:
        raise Exception("Milemarkers must be positive")
    if type(start_marker) != int or type(end_marker) != int:
        raise Exception("Milemarker must be an integer")
    
    df_by_MM = {}
    min_MM = min(start_marker, end_marker)
    max_MM = max(start_marker, end_marker)
    if reverse_count:
        for i in range(max_MM, min_MM, -1):
            mile_df = df[(df[mile_col] <= i) & (df[mile_col] > i - 1)]
            df_by_MM[i] = mile_df
    else:
        for i in range(min_MM, max_MM):
            mile_df = df[(df[mile_col] >= i) & (df[mile_col] < i + 1)]
            df_by_MM[i] = mile_df

    return df_by_MM




def create_TPM(df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Creates a TPM with the rows as the starting state and the columns as the ending state. States are NC, L1, T1, L2, T2, CC, SS.

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        year_start (int): year to start the TPM
        year_end (int): year to end the TPM

    Returns:
        pd.DataFrame: A TPM with the rows as the starting state and the columns as the ending state, with states NC, L1, T1, L2, T2, CC, SS.
    """
    freq = create_two_way_freq(df, year_start, year_end)
    # Calculate percentages
    TPM = freq.div(freq.sum(axis=1), axis=0).round(3)
    TPM = TPM.fillna(0)
    return TPM


def create_two_way_freq(df, year_start: int, year_end: int):
    """Creates a df with the number of slabs that transition from each of the one states to another

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        year_start (int): year to start 
        year_end (int): year to end 

    Returns:
        pd.DataFrame: A frequency table with the rows as the starting state and the columns as the ending state, with states NC, L1, T1, L2, T2, CC, SS.
    """
    # Creates a df with the number of slabs that transition from each of the one states to another
    freq = df.groupby([str(year_start), str(year_end)]).size().unstack()
    # Reindex so that the rows and columns so that the upper triangluar shape can take form
    freq = freq.reindex(['NC', 'L1', 'T1', 'L2', 'T2', 'CC', 'SS'])
    freq = freq.reindex(['NC', 'L1', 'T1', 'L2', 'T2', 'CC', 'SS'], axis=1)
    freq = freq.fillna(0)
    return freq


def filter_data(df):
    """Filters out the potential bridge slabs and slabs that were replaced recently, whether they were placed recently outside the time period or in the time period

    Args:
        df (pd.DataFrame): all the data for the slabs
    
    Returns:
        pd.DataFrame: filtered data
    """
    df = df[df['Replaced'].notnull() == False]
    df = df[df['Replaced and Broken'].notnull() == False]
    df = df[df['2014'] != 'R']
    df = df[df['possibly_replaced_or_bridge'] == 0]


def calc_xy_of_each_slab(df: pd.DataFrame, length_col: str, start_lat: float, start_long: float):
    pass

def calc_distance(x1, y1, x2, y2):
    c1 = (x1, y1)
    c2 = (x2, y2)
    return geopy.distance.distance(c1, c2).feet
def calc_distance(c1, c2):
    return geopy.distance.distance(c1, c2).feet

def get_bearing(c1, c2):
    long1 = c1[1]
    lat1 = c1[0]
    long2 = c2[1]
    lat2 = c2[0]
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    return brng


def calc_point_from_start(c, distance, bearing):
    start = (c[0], c[1])
    dest = geopy.distance.distance(feet=distance).destination(start, bearing)
    return dest




