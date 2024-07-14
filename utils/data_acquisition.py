import os
import time
import json
import requests
import datetime as dt
import numpy as np
import pandas as pd
from urllib.request import URLError, urlopen

def check_existance_of_file(path_to_file, file_name):
    return os.path.exists(path_to_file + file_name)

def create_history_length(start_date, freq=None, drop_weekends=True):
    """
    Enter the date, enter the frequency,
    """
    today = dt.datetime.now().date()
    if not freq:
        raise ValueError("Frequency not entered.")

    dates = pd.date_range(start=start_date, end=today, freq=freq)
    if drop_weekends:
        dates = dates[~dates.weekday.isin([5, 6])]

    return dates.shape[0]

def set_buffer(ADDRESS, n_bars, request_tail):
    """
    Set buffer on the server

    Args:
        n_bars(int, float, str): number of bars to request from the server
        request_tail(str): parameters for the server request, which include the symbol and timeframe
        (example - 'symbol=EURCHF&timeframe=M5')
    """
    respond = requests.get(
        ADDRESS + "fillbars?count=" + str(n_bars) + "&" + request_tail
    )
    if respond.ok:
        print(respond.text)
    else:
        print("Not Responded")

def get_data(address, request_tail):
    """
    The function retrieves all accumulated bars from the buffer on the server
    
    Args:
        request_tail(str): parameters for the server request, which include the symbol and timeframe
        (example - 'symbol=EURCHF&timeframe=M5')
        
    Returns:
        list: a list (empty or with dictionaries) of candles received from the server
    
    """
    try:
        return json.loads(urlopen(address + "getbars?" + request_tail).readlines()[0])
    except (URLError, OSError) as e:
        time.sleep(0.01)
        return json.loads(urlopen(address + "getbars?" + request_tail).readlines()[0])

def candles_to_df(data):
    """
    The function packages a list of dictionaries with bars into a DataFrame
    Args:
        data(list): a list (empty or with dictionaries) of bars received from the server using the get_data() function

    Returns:
        df(DataFrame): Bars prepared for work in the pipeline
    """
    df = pd.DataFrame(data, dtype=np.float)
    df["Time"] = df["Time"].astype("datetime64")
    df = df.rename(
        columns={
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Time": "timestamp",
            "Close": "close",
            "Open": "open",
        }
    ).set_index("timestamp")
    return df

def get_ohlcv(symbol, history_length, TIMEFRAME, address, query_id):
    """
    The function obtains a dataset with the number of candles history_length up to the current moment
    
    Args:
    
        history_length(int): the maximum parameter value for these indicators + 14 
        (for the number of candles requested for indicator calculation)
    
        request_tail(str): parameters for the server request, which include the symbol and timeframe
        (example - 'symbol=EURCHF&timeframe=M5') 
    
    Returns:
        ohlcv_dataset(DataFrame): Bars prepared for work in the pipeline in the required quantity
        
    """
    request_tail = (
        "symbol=" + str(symbol) + "&timeframe=" + str(TIMEFRAME) + f"&id={query_id}"
    )
    set_buffer(address, history_length, request_tail)
    ohlcv = pd.DataFrame()

    while ohlcv.shape[0] < history_length:
        print(ohlcv.shape[0])
        time.sleep(1)
        data = get_data(address, request_tail)
        if data != []:
            print("Get")
            new_candles = candles_to_df(data)
            ohlcv = pd.concat([ohlcv, new_candles])

    # region DATE CHECK
    #! Check that the last candle is no more than 2 days from today
    ohlcv.index = pd.DatetimeIndex(ohlcv.index)
    ohlcv.sort_index(inplace=True)  # ? We don't really need it, but ... Whatever

    today = np.datetime64(time.strftime("%Y-%m-%d", time.localtime(time.time())))
    last_candle_day = np.datetime64(ohlcv.index[-1].date())
    timedelta = today - last_candle_day

    if timedelta > np.timedelta64(2, "D"):
        raise ValueError(
            f"Last candle date is {last_candle_day}. Timedelta is {timedelta}. It's bigger than 2 days..."
        )

    print(
        "The last candle passed the date check (today - last candle <= 2 days)"
    )
    # endregion

    return ohlcv

def EK_get_ohlcv(
    symbol,
    history_length,
    date=None,
    MAX_INDICATOR_WINDOW=1000,
    TIMEFRAME=None,
    address=None,
    query_id=None
):
    """
    Trim the start date if necessary.
    Also checks that there are enough candles for the indicators before this start date.
    """
    df = get_ohlcv(symbol, history_length, TIMEFRAME, address, query_id)
    if date:
        date_in_index = date in df.index.floor("D")
        has_candles_for_indicators = df.loc[:date].shape[0] >= MAX_INDICATOR_WINDOW + 1

        if date_in_index and has_candles_for_indicators:
            start_date = (
                df.index.to_series().shift(MAX_INDICATOR_WINDOW).loc[date:].iloc[0]
            )
            df = df.loc[start_date:]
        else:
            raise ValueError(f"{date_in_index}, {has_candles_for_indicators}")

    return df

def DATA_BLOCK(
    symbol,
    FORCE,
    ticker_path,
    ohlcv_file,
    SAVE,
    start_date,
    freq,
    MAX_INDICATOR_WINDOW,
    TIMEFRAME,
    market_data_address,
    query_id,
    drop_weekends=True,

):
    print(f">>> START: DATA BLOCK {symbol}")

    # First, create the necessary folders, if they already exist, nothing will happen (the files inside will be okay)
    os.makedirs(f"{ticker_path}/", exist_ok=True)
    # os.makedirs(f"{ticker_path}/models/", exist_ok=True)
    # os.makedirs(f"{ticker_path}/optimized_indicators_params/", exist_ok=True)

    # First, create a log file if it does not exist, otherwise "somewhere far away" something will break.
    if not os.path.exists(f"{ticker_path}log.txt"):
        with open(f"{ticker_path}log.txt", "w"):
            pass

    file_exists = check_existance_of_file(
        path_to_file=ticker_path, file_name=ohlcv_file
    )

    if file_exists and not FORCE["OHLCV"]:
        print(
            f"Data exists, {FORCE['OHLCV']}, so the data will just be read from the file {ticker_path+ohlcv_file}"
        )
        ohlcv = pd.read_csv(
            ticker_path + ohlcv_file,
            parse_dates=["timestamp"],
            dayfirst=True,
            index_col=["timestamp"],
            sep=";",
        )
    else:
        print(f">>> DOWNLOADING {symbol}...")
        history_length = create_history_length(start_date, freq, drop_weekends)
        ohlcv = EK_get_ohlcv(
            symbol=symbol,
            history_length=history_length,
            date=start_date,
            MAX_INDICATOR_WINDOW=MAX_INDICATOR_WINDOW,
            TIMEFRAME=TIMEFRAME,
            address=market_data_address,
            query_id=query_id
        )

    # ? Put the columns in a generally accepted order.
    ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]

    # ? If you need to save - save.
    if SAVE["OHLCV"]:
        print(f">>> SAVING OHLCV {symbol} ...")
        ohlcv.to_csv(ticker_path + ohlcv_file, sep=";")
        print(f">>> SAVING OHLCV {symbol} ... DONE")

    print(f">>> DONE: DATA BLOCK {symbol}")
    print("-" * 50)

    return ohlcv
