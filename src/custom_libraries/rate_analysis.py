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
        final_df = pd.concat([curr_interval_data.astype(final_df.dtypes), final_df.astype(curr_interval_data.dtypes)], axis=0)
        
    
    return final_df


def calc_average_state_by_slab(df: pd.DataFrame, curr_year: str, buffer: int,
                               nc_weight: float = 0, t1_weight: float = 1,
                               l1_weight: float = 1, t2_weight: float = 2,
                               l2_weight: float = 2, cc_weight: float = 2,
                               ss_weight: float = 3, r_weight: float = 2,
                               b_weight: float = 0):
    """
    Generates np array of average distance to each slab. Can attach the np
    array as a new column to the existing dataframe if need be.
    Args:
        df (pd.DataFrame): all the slab data
        curr_year (str): the year of interest with the slab state for each slab
        buffer (int): how many slabs left and right to consider
        nc_weight (float): given weight of NC. Defaults to 0.
        t1_weight (float): given weight of T1. Defaults to 1.
        l1_weight (float): given weight of L1. Defaults to 1.
        t2_weight (float): given weight of NC. Defaults to 2.
        l2_weight (float): given weight of NC. Defaults to 2.
        cc_weight (float): given weight of NC. Defaults to 2.
        ss_weight (float): given weight of NC. Defaults to 3.
        r_weight (float): given weight of NC. Defaults to 2.
        b_weight (float): given weight of NC. Defaults ot 0.

    Returns:
        np.array: np array of average slab states

    """
    slab_states = df[curr_year].to_numpy()

    if buffer >= len(slab_states):
        raise ValueError('Buffer is too large')
    if buffer <= 0:
        raise ValueError('Buffer is too small')

    # define the mappings
    slab_state_to_weight = {
        'NC': nc_weight,
        'T1': t1_weight,
        'T2': t2_weight,
        'L1': l1_weight,
        'L2': l2_weight,
        'CC': cc_weight,
        'SS': ss_weight,
        'R': r_weight,
        'B': b_weight
    }
    # pad the ends
    n = len(slab_states)
    slab_states = np.pad(slab_states, (buffer, buffer), mode='reflect')

    # map slab state to numerical weight
    vfunc = np.vectorize(lambda state: slab_state_to_weight[state])
    slab_states = vfunc(slab_states)

    res_average_slab_state = np.zeros(n)

    moving_sum = np.sum(slab_states[0:buffer]) + np.sum(
        slab_states[buffer+1:buffer+buffer+1])

    for i in range(buffer, n + buffer):
        if i != buffer:
            moving_sum = (moving_sum + slab_states[i + buffer] - slab_states[
                i - buffer - 1] + slab_states[i - 1] - slab_states[i])
        res_average_slab_state[i - buffer] = moving_sum / (buffer * 2)

    return res_average_slab_state


def has_deteriorated(df: pd.DataFrame, curr_year: str, end_year: str):
    """
    Generates np array of whether or not a slab has deteriorated. Can attach
    the np array as a new column to the existing dataframe if need be.
    Args:
        df (pd.DataFrame): all the slab data
        curr_year (str): the year of interest with the slab state for each slab
        end_year (str): the year to compare to with the slab state for each slab

    Returns:
        np.array: np array of whether or not a slab has deteriorated

    """

    NC_damaged = {'L1', 'T1', 'L2', 'T2', 'CC', 'SS', 'R'}
    L1_damaged = {'T1', 'L2', 'T2', 'CC', 'SS', 'R'}
    T1_damaged = {'L2', 'T2', 'CC', 'SS', 'R'}
    L2_damaged = {'T2', 'CC', 'SS', 'R'}
    T2_damaged = {'CC', 'SS', 'R'}
    CC_damaged = {'SS', 'R'}
    start_slab_states = df[curr_year].to_numpy()
    end_slab_states = df[end_year].to_numpy()

    res = np.zeros(len(start_slab_states))

    for i in range(len(start_slab_states)):
        if start_slab_states[i] == 'NC':
            if end_slab_states[i] in NC_damaged:
                res[i] = 1
        elif start_slab_states[i] == 'L1':
            if end_slab_states[i] in L1_damaged:
                res[i] = 1
        elif start_slab_states[i] == 'T1':
            if end_slab_states[i] in T1_damaged:
                res[i] = 1
        elif start_slab_states[i] == 'L2':
            if end_slab_states[i] in L2_damaged:
                res[i] = 1
        elif start_slab_states[i] == 'T2':
            if end_slab_states[i] in T2_damaged:
                res[i] = 1
        elif start_slab_states[i] == 'CC':
            if end_slab_states[i] in CC_damaged:
                res[i] = 1
        elif start_slab_states[i] == 'SS':
            if end_slab_states[i] == 'R':
                res[i] = 1
        else:
            res[i] = 0
    return res



def categorize_proximity(df: pd.DataFrame, curr_year: str):
   
    slab_states = df[curr_year].to_numpy()


    # define the mappings
    slab_state_to_weight = {
        'NC': 0,
        'T1': 1,
        'T2': 2,
        'L1': 1,
        'L2': 2,
        'CC': 2,
        'SS': 2,
        'R': 2,
        'B': 0
    }
    # pad the ends
    n = len(slab_states)
    slab_states = np.pad(slab_states, (1, 1), mode='reflect')

    # map slab state to numerical weight
    vfunc = np.vectorize(lambda state: slab_state_to_weight[state])
    slab_states = vfunc(slab_states)

    res_average_slab_state = np.zeros(n)

    

    for i in range(1, n + 1):
        state_set = {slab_states[i - 1], slab_states[i + 1]}
        if state_set == {0}:
            res_average_slab_state[i - 1] = 0
        elif state_set == {0, 1}:
            res_average_slab_state[i - 1] = 1
        elif state_set == {1}:
            res_average_slab_state[i - 1] = 2
        elif state_set == {0, 2}:
            res_average_slab_state[i - 1] = 3
        elif state_set == {1, 2}:
            res_average_slab_state[i - 1] = 4
        elif state_set == {2}:
            res_average_slab_state[i - 1] = 5
        else:
            res_average_slab_state[i - 1] = 6
    return res_average_slab_state

