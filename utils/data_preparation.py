"""
This script contains all functions for pre/post-processing of data.
"""
import os
import pandas as pd
import shutil
import datetime

def init_logger(ticker_path):
    import multiprocessing, logging
    
    name = ticker_path.split("/")[-1]
    # logger = logging.getLogger(name)
    logger = multiprocessing.get_logger(name)
    FORMAT = "%(asctime)s %(name)s: %(message)s"
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(f"{ticker_path}/log.log", "w", "utf-8")
    fh.setFormatter(logging.Formatter(FORMAT))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(ch)
    return logger

def write_log(log_info, ticker_path):
    """
    Takes a message to be logged - log_info (type str) 
    and ticker_path - the path to the ticker folder containing log.txt.
    Writes to the file: "2022-04-26 14:28:09 - {log_info}. Done in 0:00:23.625299"
    'Done in' is the delta between the time of the previous message and the current one.
    """
    
    with open(f'{ticker_path}/log.txt', 'r', encoding='utf-8') as log_file:
        now = datetime.datetime.now()
        for line in log_file:
            pass

        try:    
            str_time = line.split(" - ")[0]
            last_time = datetime.datetime.strptime(str_time, ("%Y-%m-%d %H:%M:%S"))
            delta = now - last_time
        except:
            delta = "work time not fixed"
    with open(f'{ticker_path}/log.txt', 'a', encoding='utf-8') as log_file:        
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")} -- {log_info}. Done in {delta}', file=log_file) 
    
def create_temp_folders(ticker_path):
    
    try:
        shutil.rmtree(ticker_path)
    except:
        pass
    
    os.makedirs(ticker_path, exist_ok=True)
    os.makedirs(f"{ticker_path}/optimization_borders", exist_ok=True)
    print(f"{ticker_path}/optimization_borders created")
    os.makedirs(f"{ticker_path}/trash", exist_ok=True)
    print(f"{ticker_path}/trash created")
    os.makedirs(f"{ticker_path}/pycaret", exist_ok=True)
    print(f"{ticker_path}/pycaret created")

def get_and_crop_df(
    file_path,
    start_date,
    end_date,
    adjClose=False,
    prec_candles=180,
):
    """
    Signature:
    get_and_crop_df(
        file_path,
        adjClose=False,
        start_year=START_YEAR,
        start_month=START_MONTH,
        end_year=END_YEAR,
        end_month=END_MONTH,
    )
    
    Docstring:
    Retrieve OHLCV from CSV and crop the required interval.
    
    Parameters
    ----------
    file_path: str
        CSV file name
    adjClose: bool, default False
        If the data has a column with "Adjusted Close"
    start_date: str
        Start date (inclusive)
    end_date: str
        End date (inclusive)
    prec_candles: int
        Number of candles before the start date that will be cropped 
        after target and indicators (NaNs due to sliding windows) are built
    Returns
    -------
    DataFrame
        A comma-separated values (CSV) file returned as a two-dimensional
        data structure with labeled axes.
    """
    
    if adjClose:
        columns = [
            "date",
            "time",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    else:
        columns = ["date", "time", "open", "high", "low", "close", "volume"]

    df = pd.read_csv(file_path, header=None, names=columns)
    df["timestamp"] = pd.to_datetime(df.date + " " + df.time)
    df.drop(columns=["date", "time"], inplace=True)
    df.set_index("timestamp", inplace=True)

    df = pd.concat(
        [
            df[:start_date].iloc[-prec_candles:],
            df[start_date:end_date]
        ]
    )
    return df

# def get_and_crop_df(
#     file_path,
#     adjClose=False,
#     start_year=START_YEAR,
#     start_month=START_MONTH,
#     end_year=END_YEAR,
#     end_month=END_MONTH,
#     prec_candles=30,
# ):
#     """
#     Signature:
#     get_and_crop_df(
#         file_path,
#         adjClose=False,
#         start_year=START_YEAR,
#         start_month=START_MONTH,
#         end_year=END_YEAR,
#         end_month=END_MONTH,
#     )
    
#     Docstring:
#     Retrieve OHLCV from CSV and crop the required interval.
    
#     Parameters
#     ----------
#     file_path: str
#         CSV file name
#     adjClose: bool, default False
#         If the data has a column with "Adjusted Close"
#     start_year: int
#         Start year of the dataset (inclusive)
#     start_month: int
#         Start month of the dataset (inclusive)
#     end_year: int
#         End year of the dataset (inclusive)
#     end_month: int
#         End month of the dataset (inclusive)
#     prec_candles: int
#         Number of candles before the start date that will be cropped 
#         after target and indicators (NaNs due to sliding windows) are built
#     Returns
#     -------
#     DataFrame
#         A comma-separated values (CSV) file returned as a two-dimensional
#         data structure with labeled axes.
#     """
#     start_date = datetime.datetime(start_year, start_month, 1)
#     end_date = datetime.datetime(
#         end_year, end_month, calendar.monthrange(END_YEAR, END_MONTH)[1]
#     ) + datetime.timedelta(days=1)

#     if adjClose:
#         columns = [
#             "date",
#             "time",
#             "open",
#             "high",
#             "low",
#             "close",
#             "adj_close",
#             "volume",
#         ]
#     else:
#         columns = ["date", "time", "open", "high", "low", "close", "volume"]

#     df = pd.read_csv(file_path, header=None, names=columns)
#     df["timestamp"] = pd.to_datetime(df.date + " " + df.time)
#     df.drop(columns=["date", "time"], inplace=True)
#     df.set_index("timestamp", inplace=True)
#     df = pd.concat(
#         [
#             df[(df.index < start_date)][-prec_candles:],
#             #df[:START_DATE][-prec_candles:],
#             df[(df.index >= start_date) & (df.index < end_date)],
#             #df[START_DATE:END_DATE]
#         ]
#     )
#     return df
