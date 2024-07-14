import itertools as it
import numpy as np
import pandas as pd

def get_part_of_column_name(multicolumns, level=0, sep='~', position=1):
    '''
    Extracts a part of column names from multi-index columns.
    
    Parameters
    ----------
    multicolumns : MultiIndex
        MultiIndex columns to extract part from.
    level : int, default 0
        The level of the columns to extract part from.
    sep : str, default '~'
        Separator used in the column names.
    position : int, default 1
        Position of the part to extract after splitting by separator.

    Returns
    -------
    set
        Set of extracted parts of the column names.
    '''
    return set(col.split(sep)[position] for col in multicolumns.get_level_values(level))

def get_all_combinations(iterable_obj, orders=None):
    '''
    Gets all possible combinations of elements in an iterable object up to a specified order.
    
    Parameters
    ----------
    iterable_obj : iterable
        An iterable object to generate combinations from.
    orders : range, optional
        The maximum order of combinations to generate. If not provided, defaults to range(1, len(iterable_obj) + 1).

    Returns
    -------
    list of tuples
        List of all possible combinations up to the specified order.
    '''
    if not orders:
        orders = range(1, len(iterable_obj) + 1)

    combinations = []
    for i in orders:
        combinations += it.combinations(iterable_obj, i)
    
    return combinations

def get_dfs_combinations(df, combinations, level=0):
    '''
    Extracts DataFrames based on column combinations.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to extract columns from.
    combinations : list of tuples
        List of column combinations to extract.
    level : int, default 0
        The level of the columns to match combinations against.

    Returns
    -------
    list of DataFrames
        List of DataFrames with columns matching the combinations.
    '''
    dfs = []
    for combination in combinations:
        reg_exp = '~'
        for el in combination[:-1]:
            reg_exp += f'{el}|~'
        reg_exp += f'{combination[-1]}'

        dfs.append(df.loc[:, df.columns.get_level_values(level).str.contains(f'{reg_exp}')])

    return dfs

def create_relearner_dates_grid(df, relearn_duration: str, trade_duration: int = 1):
    '''
    Creates a grid of relearner dates.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the index to resample.
    relearn_duration : str
        The duration string to resample the index (e.g., '1M' for one month).
    trade_duration : int, default 1
        The trading duration to use for the resampling.

    Returns
    -------
    DataFrame
        DataFrame containing the entry and exit dates for relearner periods.
    '''
    n_of_periods = int("".join([el for el in relearn_duration if el.isdigit()]))
    type_of_periods = "".join([el for el in relearn_duration if el.isalpha()])
    df_tmp = df.index.to_series().resample(type_of_periods).agg(["first", "last"])
    np_tmp = df_tmp.to_numpy()
    entries = np_tmp[::trade_duration, 0]
    outs = np_tmp[n_of_periods - 1 : -1 : trade_duration, 1]

    relearner_dates_grid = pd.DataFrame(zip(entries, outs), columns=["entries", "outs"])

    return relearner_dates_grid

def create_trade_dates_grid(relearner_dates_grid, df):
    '''
    Creates a grid of trade dates.
    
    Parameters
    ----------
    relearner_dates_grid : DataFrame
        DataFrame containing the relearner dates.
    df : DataFrame
        The DataFrame containing the index to use for generating trade dates.

    Returns
    -------
    DataFrame
        DataFrame containing the entry and exit dates for trade periods.
    '''
    trade_dates_grid = np.array(
        [
            (entry, out)
            for entry, out in zip(
                relearner_dates_grid.values[:, 1], relearner_dates_grid.values[1:, 1]
            )
        ]
        + [
            (entry, out)
            for entry, out in zip(relearner_dates_grid.values[-1:, 1], df.index[-1:])
        ]
    )
    trade_dates_grid = pd.DataFrame(
        [
            (df.loc[entry:out].iloc[1:].index[0], df.loc[entry:out].iloc[1:].index[-1])
            for entry, out in trade_dates_grid
        ],
        columns=["entries", "outs"],
    )

    return trade_dates_grid

def DATES_GRID_BLOCK(symbol, ohlcv, relearn_duration, trade_duration, required_n_of_periods):
    '''
    Generates grids of relearner and trade dates, and identifies key timestamps for backtesting.
    
    Parameters
    ----------
    symbol : str
        The symbol associated with the data.
    ohlcv : DataFrame
        The OHLCV data to use for generating dates grids.
    relearn_duration : str
        The duration string to use for resampling in relearner grid.
    trade_duration : int
        The trading duration to use for resampling in trade grid.
    required_n_of_periods : int
        The number of periods required for backtesting.

    Returns
    -------
    tuple
        A tuple containing the relearner dates grid, trade dates grid, 
        start and end timestamps for in-sample backtesting, and the start timestamp for out-of-sample backtesting.
    '''
    print(f">>> START: DATES GRID BLOCK {symbol}")

    relearner_dates_grid = create_relearner_dates_grid(ohlcv, relearn_duration, trade_duration)
    print("Сетка дат для переобучения готова.")

    trade_dates_grid = create_trade_dates_grid(relearner_dates_grid, ohlcv)
    print("Сетка дат для торговли готова.")

    backtest_IS_start_timestamp, backtest_IS_end_timestamp = (
        trade_dates_grid.iloc[0]["entries"],
        trade_dates_grid.iloc[required_n_of_periods - 1]["outs"],
    )
    print(f"Даты, с которой стартует и заканчивается первый этап In Sample: {backtest_IS_start_timestamp, backtest_IS_end_timestamp}")

    backtest_OOS_start_timestamp = trade_dates_grid.iloc[required_n_of_periods]["entries"]
    print(f"Дата, с которой стартует <<реальная торговля>>: {backtest_OOS_start_timestamp}")

    print(f">>> DONE: DATES GRID BLOCK {symbol}")
    print("-" * 50)

    return (
        relearner_dates_grid,
        trade_dates_grid,
        backtest_IS_start_timestamp,
        backtest_IS_end_timestamp,
        backtest_OOS_start_timestamp,
    )

def join_time_pair(dates_grid, separator=" "):
    '''
    Joins pairs of dates into a single string with a separator.
    
    Parameters
    ----------
    dates_grid : DataFrame
        DataFrame containing pairs of dates to join.
    separator : str, default " "
        The separator to use for joining the dates.

    Returns
    -------
    Series
        Series of joined date strings.
    '''
    return dates_grid.astype(str).apply(separator.join, axis=1)

def create_time_labels(relearner_dates_grid, trade_dates_grid, separator=" "):
    '''
    Creates lists of time labels for files from relearner and trade dates grids.
    
    Parameters
    ----------
    relearner_dates_grid : DataFrame
        DataFrame containing the relearner dates.
    trade_dates_grid : DataFrame
        DataFrame containing the trade dates.
    separator : str, default " "
        The separator to use for joining the dates.

    Returns
    -------
    tuple
        A tuple containing lists of relearner and trade time labels,
        and dictionaries mapping relearner to trade and trade to relearner labels.
    '''
    relearn_time_labels = join_time_pair(relearner_dates_grid, separator)
    trade_time_labels = join_time_pair(trade_dates_grid, separator)

    relearn_to_trade_map = {
        el_1: el_2
        for el_1, el_2 in pd.concat([relearn_time_labels, trade_time_labels], axis=1).to_numpy()
    }

    trade_to_relearn_map = {
        el_1: el_2
        for el_1, el_2 in pd.concat([trade_time_labels, relearn_time_labels], axis=1).to_numpy()
    }

    return (
        relearn_time_labels,
        trade_time_labels,
        relearn_to_trade_map,
        trade_to_relearn_map,
    )
