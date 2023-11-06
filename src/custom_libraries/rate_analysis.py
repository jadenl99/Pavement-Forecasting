import numpy as np
import pandas as pd



def nc_distance_from_damaged(
        df: pd.DataFrame, 
        start_year: int,
        damaged_states: set = {'L1', 'T1', 'T2', 'L2', 'CC', 'SS', 'R'}
        ) -> np.ndarray:
    """Calculates the distance of each NC slab to its nearest damaged slab.     Damaged slab states default to L1, T1, T2, L2, CC, SS, and R.

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        start_year (int): the year of interest

    Returns:
        np.ndarray: a numpy array of the distance of each NC slab to its nearest damaged slab. Index 0 represents the first slab in the data, index 1 represents the second slab in the data, etc.
    """
    initial_slab_states = df[str(start_year)].to_numpy()
    num_slabs = len(initial_slab_states)
    # define damaged slab states (the default)

    # initilize all slabs to be an infinite distane from damaged slabs for min function to work 
    distance_to_nearest_damaged_slab = np.full(num_slabs, 1000000)
    

    last_damaged_index = None
    # calculate distance to nearest damaged slab from the left
    for i in range(num_slabs):
        if initial_slab_states[i] in damaged_states:
            last_damaged_index = i
            # damaged slabs obviously have a distance of 0
            distance_to_nearest_damaged_slab[i] = 0
        elif last_damaged_index is not None and initial_slab_states[i] == 'NC':
            distance_to_nearest_damaged_slab[i] = i - last_damaged_index
    
    last_damaged_index = None
    # calculate distance to nearest damaged slab from the right and comparing it to the left to see which is closer
    for i in range(num_slabs -1, -1, -1):
        if initial_slab_states[i] in damaged_states:
            last_damaged_index = i
        elif last_damaged_index is not None and initial_slab_states[i] == 'NC':
            # comparing left side distance vs. right side distance and taking the minimum
            distance_to_nearest_damaged_slab[i] = min(distance_to_nearest_damaged_slab[i], last_damaged_index - i)
    
    distance_to_nearest_damaged_slab[
        distance_to_nearest_damaged_slab == 1000000
        ] = 0
    return distance_to_nearest_damaged_slab


def nc_distance_vs_deterioration_rate(
        df: pd.DataFrame,
        distance_to_nearest_damaged_slab: np.ndarray,
        start_year: int,
        end_year: int
        ) -> pd.DataFrame:
    """Generates a table with the NC deterioration rate for each distance from a damaged slab, first column being distance, second column being deterioration rate.

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        distance_to_nearest_damaged_slab (np.ndarray): a numpy array of the distance of each NC slab to its nearest damaged slab. Index 0 represents the first slab in the data, index 1 represents the second slab in the data, etc.
        start_year (int): start year of the period of interest
        end_year (int): end year of the period of interest

    Returns:
        pd.DataFrame: a dataframe with the first column being distance to nearest damaged slab, and the second column being the NC deterioration rate for that distance.
    """
    # NC slabs that are still NC after 1 year are considered to have not deteriorated. An NC slab is considered to have deteriorated if it is NC in the first year and is either T1, L1, T2, L2, CC, SS, or R in the second year.
    damaged_states = {'L1', 'T1', 'T2', 'L2', 'CC', 'SS', 'R'}
    conditions = [
        (df[str(start_year)] == 'NC') & ~(df[str(end_year)].isin(damaged_states)),
        df[str(start_year)] != 'NC'
    ]

    choices = [0, 0]
   
    df['deteriorated'] = np.select(conditions, choices, default=1)
    df['dist'] = distance_to_nearest_damaged_slab

    
    df = df[df[str(start_year)] == 'NC']
    #dfg = df.groupby(by=['dist', 'deteriorated']).size().reset_index(name='count')
    dfg = pd.crosstab(df['dist'], df['deteriorated'])
    return dfg

def deteriorated_and_dist(
        df: pd.DataFrame,
        distance_to_nearest_damaged_slab: np.ndarray,
        start_year: int,
        end_year: int,
        damaged_states: set = {'L1', 'T1', 'T2', 'L2', 'CC', 'SS', 'R'}

        ) -> pd.DataFrame:
    """Generate a table with each NC slab's distance to its nearest damaged slab and whether or not it deteriorated. 0 represents not deteriorated, and 1 represents deteriorated.

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        distance_to_nearest_damaged_slab (np.ndarray): a numpy array of the distance of each NC slab to its nearest damaged slab. Index 0 represents the first slab in the data, index 1 represents the second slab in the data, etc.
        start_year (int): start year of the period of interest
        end_year (int): end year of the period of interest
        damaged_states (set, optional): Set of slab states to consider as damaged when marking whether an NC slab has deteriorated or not. Defaults to {'L1', 'T1', 'T2', 'L2', 'CC', 'SS', 'R'}.

    Returns:
        pd.DataFrame: a dataframe with each NC slab's distance to its nearest damaged slab and whether or not it deteriorated. 0 represents not deteriorated, and 1 represents deteriorated.
    """
    # NC slabs that are still NC after 1 year are considered to have not deteriorated. An NC slab is considered to have deteriorated if it is NC in the first year and is either T1, L1, T2, L2, CC, SS, or R in the second year.
    
    conditions = [
        (df[str(start_year)] == 'NC') & ~(df[str(end_year)].isin(damaged_states)),
        df[str(start_year)] != 'NC'
    ]
    
    choices = [0, 0]
    new_df = pd.DataFrame()
    new_df['initial'] = df[str(start_year)]
    new_df['dist'] = distance_to_nearest_damaged_slab
    new_df['deteriorated'] = np.select(conditions, choices, default=1)
    new_df = new_df[new_df['initial'] == 'NC']
    return new_df

def construct_nc_deterioration_rate_table(
        df: pd.DataFrame,
        start_year: int,
        end_year: int,
        interval: int, 
        damaged_states: set = {'L1', 'T1', 'T2', 'L2', 'CC', 'SS', 'R'}
        ) -> pd.DataFrame:
    """Constructs a table with the NC deterioration rate for each distance from a damaged slab.

    Args:
        df (pd.DataFrame): dataframe with all the slab state data
        start_year (int): start year
        end_year (int): end year
        interval (int): period between transitions
        damaged_states (set, optional): Defaults to {'L1', 'T1', 'T2', 'L2', 'CC', 'SS', 'R'}. List of slab states to consider "damaged" in analysis

    Returns:
        pd.DataFrame: A table with the NC rate for each distance from a damaged slab for each interval year.
    """
    freq_by_year = {}
    final_df = pd.DataFrame(columns=['distance', 'deterioration_rate', 'period'])
    first_end_interval = start_year + interval
    for e_year in range(first_end_interval, end_year + 1, interval):
        s_year = e_year - interval
        nc_distance_damaged = nc_distance_from_damaged(df, s_year, damaged_states)
        curr_interval_data = nc_distance_vs_deterioration_rate(df, nc_distance_damaged, s_year, e_year)
        curr_interval_data = curr_interval_data.div(curr_interval_data.sum(axis=1), axis=0)
        curr_interval_data = curr_interval_data.drop(0, axis=1)
        curr_interval_data['distance'] = curr_interval_data.index
        curr_interval_data['period'] = f'{s_year}-{e_year}'
        curr_interval_data = curr_interval_data.rename(columns={1 : 'deterioration_rate'})
        
        final_df = pd.concat([final_df, curr_interval_data], axis=0)
    
    return final_df