import configparser
import datetime as dt
import json
import os
import time
from urllib.request import URLError, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
from utils import features_engeneering as fe
from utils import modeling as md
from utils.data_acquisition import *
from utils.loop_funcs import *

config = configparser.ConfigParser()
config.read("config.ini")

market_data_adress = config["java_ip"]["market_data_ip"]

# Params
symbols = [
    "AUDUSD",
    "EURCHF",
    "EURGBP",
    "EURUSD",
    "GBPCHF",
    "NZDCAD",
    "USDCAD",
    "USDCHF",
]

TIMEFRAME = "H1"
freq = "H"

start_date = "2019-01-02"
MAX_INDICATOR_WINDOW = 1000
OPTIMIZE_INDICATORS = True
n_of_candles_to_add_to_target = 24
relearn_duration, trade_duration, required_n_of_periods = "36M", 3, 4

TICKERS_PATH = "work_dir/"
query_id = "rebalance"
ohlcv_file = "ohlcv.csv"
target_file = "target.csv"
indicators_file = "indicators.csv"

possible_models = {
    "PC": [
        "AdaBoostClassifier",
        "CatBoostClassifier",
        "DecisionTreeClassifier",
        "DummyClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "GradientBoostingClassifier",
        "KNeighborsClassifier",
        "LGBMClassifier",
        "LinearDiscriminantAnalysis",
        "LogisticRegression",
        "QuadraticDiscriminantAnalysis",
        "RandomForestClassifier",
        "RidgeClassifier",
        "SGDClassifier",
        "XGBClassifier",
    ],
}

current_models = {
    "PC": [
        "AdaBoostClassifier",
        "CatBoostClassifier",
        "DecisionTreeClassifier",
        "DummyClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "GradientBoostingClassifier",
        "KNeighborsClassifier",
        "LGBMClassifier",
        "LinearDiscriminantAnalysis",
        "LogisticRegression",
        "QuadraticDiscriminantAnalysis",
        "RandomForestClassifier",
        "RidgeClassifier",
        "SGDClassifier",
        "XGBClassifier",
    ],
}

possible_function_to_call_READ_MODELS = {
    # "AG": md.AG_READ_MODELS,
    "PC": md.PC_READ_MODELS,
}

possible_function_to_call_TRAIN = {
    # "AG": md.AG_TRAIN,
    "PC": md.PC_TRAIN
}

possible_function_to_call_TEST = {
    # "AG": md.AG_TEST,
    "PC": md.PC_TEST
}

current_libraries = list(current_models.keys())

for library in current_libraries:
    for model in current_models[library]:
        if model not in possible_models[library]:
            raise ValueError(f"Wrong model name! You specified {model}!")
            
FORCE = {
    "OHLCV": True,
    "TARGET": True,
    "INDICATORS": False,
    "MODELS_TRAIN": False,
    "MODELS_TEST": False,
    "TRADE": False

}
# FORCE = {key: True for key in FORCE.keys()}
SAVE = {
    "OHLCV": True,
    "TARGET": True,
    "INDICATORS": True,
    "MODELS": True,
    "PREDICTS": True,
    "PORTFOLIOS": False,
}

def time_me(func):
    def wrapper(*args, **kwargs):
        start_time = dt.datetime.now()
        result = func(*args, **kwargs)

        print(
            f"Elapsed time of block {func.__name__!r}: {dt.datetime.now() - start_time}"
        )
        print("-" * 50)
        return result

    return wrapper


# ----------------------------------------------------------------------------------------------------------------

# -------------------------------------------------Executable Blocks----------------------------------------------
@time_me
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
    market_data_adress,
    query_id,
    drop_weekends=True,
):
    print(f">>> START: DATA BLOCK {symbol}")

    os.makedirs(f"{ticker_path}/", exist_ok=True)
    # os.makedirs(f"{ticker_path}/models/", exist_ok=True)
    # os.makedirs(f"{ticker_path}/optimized_indicators_params/", exist_ok=True)

    if not os.path.exists(f"{ticker_path}log.txt"):
        with open(f"{ticker_path}log.txt", "w"):
            pass

    file_exists = check_existance_of_file(
        path_to_file=ticker_path, file_name=ohlcv_file
    )

    if file_exists and not FORCE["OHLCV"]:
        print(
            f"The data is available, {FORCE['OHLCV']}, so the data will simply be read from the file {ticker_path+ohlcv_file}"
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
            address=market_data_adress,
            query_id=query_id,
        )

    ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]

    if SAVE["OHLCV"]:
        print(f">>> SAVING OHLCV {symbol} ...")
        ohlcv.to_csv(ticker_path + ohlcv_file, sep=";")
        print(f">>> SAVING OHLCV {symbol} ... DONE")

    print(f">>> DONE: DATA BLOCK {symbol}")
    print("-" * 50)

    return ohlcv


@time_me
def DATES_GRID_BLOCK(
    symbol, ohlcv, relearn_duration, trade_duration, required_n_of_periods
):
    print(f">>> START: DATES GRID BLOCK {symbol}")

    relearner_dates_grid = create_relearner_dates_grid(
        ohlcv, relearn_duration, trade_duration
    )
    print("The date grid for retraining is ready.")

    trade_dates_grid = create_trade_dates_grid(relearner_dates_grid, ohlcv)
    print("The date grid for trading is ready.")

    backtest_IS_start_timestamp, backtest_IS_end_timestamp = (
        trade_dates_grid.iloc[0]["entries"],
        trade_dates_grid.iloc[required_n_of_periods - 1]["outs"],
    )
    print(
        f"The dates that mark the start and end of the first In-Sample stage: {backtest_IS_start_timestamp, backtest_IS_end_timestamp}"
    )

    backtest_OOS_start_timestamp = trade_dates_grid.iloc[required_n_of_periods][
        "entries"
    ]
    print(
        f"The date from which 'live trading' starts.: {backtest_OOS_start_timestamp}"
    )

    print(f">>> DONE: DATES GRID BLOCK {symbol}")
    print("-" * 50)

    return (
        relearner_dates_grid,
        trade_dates_grid,
        backtest_IS_start_timestamp,
        backtest_IS_end_timestamp,
        backtest_OOS_start_timestamp,
    )


def join_time_pair(dates_grid, separetor=" "):

    return dates_grid.astype(str).apply(separetor.join, axis=1)


@time_me
def create_time_labels(relearner_dates_grid, trade_dates_grid, separetor=" "):
    """
    Creates a list of date-names for files.
    """

    relearn_time_labels = join_time_pair(relearner_dates_grid, separetor=" ")
    trade_time_labels = join_time_pair(trade_dates_grid, separetor=" ")

    relearn_to_trade_map = {
        el_1: el_2
        for el_1, el_2 in pd.concat(
            [relearn_time_labels, trade_time_labels], axis=1
        ).to_numpy()
    }

    trade_to_relearn_map = {
        el_1: el_2
        for el_1, el_2 in pd.concat(
            [trade_time_labels, relearn_time_labels], axis=1
        ).to_numpy()
    }

    return (
        relearn_time_labels,
        trade_time_labels,
        relearn_to_trade_map,
        trade_to_relearn_map,
    )


@time_me
def TARGET_BLOCK(
    ohlcv,
    symbol,
    FORCE,
    SAVE,
    ticker_path,
    target_file,
    relearner_dates_grid,
    relearn_time_labels,
    n_of_candles_to_add_to_target,
    trend_side,
):
    print(f">>> START: TARGET BLOCK {symbol}")

    os.makedirs(f"{ticker_path}", exist_ok=True)
    file_exists = check_existance_of_file(
        path_to_file=ticker_path, file_name=target_file
    )
    # ? Если таргет уже посчитан и не надо пересчитывать.
    if file_exists and not FORCE["TARGET"]:
        print(
            f"The data is available, {FORCE['TARGET']=}, so it will simply be read from the file {ticker_path+target_file}"
        )
        targeted_df = pd.read_csv(
            ticker_path + target_file, index_col=[0], parse_dates=[0], sep=";"
        )
    else:
        print(f">>> CALCULATING TARGET {symbol}...")
        parts_targeted_dfs = []
        for relearn_entry, relearn_out in relearner_dates_grid.to_numpy():
            target_entry, target_out = (
                ohlcv.index.to_series()
                .shift(n_of_candles_to_add_to_target)
                .loc[relearn_entry],  #! <--- comma
                relearn_out,
            )

            # ? Labeling the target.
            # parts_targeted_dfs.append(
            #     fe.mark_target_new(
            #         ohlcv.loc[target_entry:target_out],
            #         ticker_path,
            #         p.adjClose,
            #         **p.target_params,
            #     ) 
            # )

            period_ohlcv = ohlcv.loc[target_entry:target_out]
            mix_target = cook_mix_target(period_ohlcv)

            trend_side_target = mix_target[trend_side]
            # trend_side_target.name = relearn_time_labels
            parts_targeted_dfs.append(trend_side_target)

        targeted_df = pd.concat(parts_targeted_dfs, axis=1, keys=relearn_time_labels)

        if SAVE["TARGET"]:
            print(f">>> SAVING TARGET {symbol}")
            targeted_df.to_csv(ticker_path + target_file, sep=";")

    print(f">>> DONE: TARGET BLOCK {symbol}")
    print("-" * 50)

    return targeted_df


@time_me
def INDICATORS_BLOCK(
    symbol,
    FORCE,
    SAVE,
    ticker_path,
    indicators_file,
    relearner_dates_grid,
    relearn_time_labels,
    trade_dates_grid,
    targeted_df,
    ohlcv,
    MAX_INDICATOR_WINDOW,
):
    print(f">>> START: INDICATORS BLOCK {symbol}")

    os.makedirs(f"{ticker_path}/optimized_indicators_params/", exist_ok=True)

    parts_indicators_df = []

    for i in range(relearner_dates_grid.shape[0]):

        indicators_optimized_params_folder = f"optimized_indicators_params/{relearn_time_labels.iloc[i].replace(':', '_')}/"

        os.makedirs(
            rf"{ticker_path}/{indicators_optimized_params_folder}", exist_ok=True
        )

        file_exists = check_existance_of_file(
            path_to_file=f"{ticker_path}/{indicators_optimized_params_folder}",
            file_name=indicators_file,
        )

        if file_exists and not FORCE["INDICATORS"]:
            print(
                f"Данные есть, {FORCE['INDICATORS']=}, поэтому данные просто прочитаются из файла {ticker_path}/{indicators_optimized_params_folder}{indicators_file}"
            )
            indicators_part_df = pd.read_csv(
                ticker_path + indicators_optimized_params_folder + indicators_file,
                sep=";",
                index_col=0,
                parse_dates=[0],
                dayfirst=True,
            )
            parts_indicators_df.append(indicators_part_df)

        else:
            print(f">>> CALCULATING INDICATORS {symbol}...")

            relearn_entry, relearn_out = relearner_dates_grid.iloc[i]
            trade_entry, trade_out = trade_dates_grid.iloc[i]
            indicators_entry_IS, indicators_out_IS = (
                ohlcv.index.to_series()
                .shift(MAX_INDICATOR_WINDOW)
                .loc[relearn_entry],  #! <--- comma
                relearn_out,
            )

            indicators_entry_OOS, indicators_out_OOS = (
                ohlcv.index.to_series()
                .shift(MAX_INDICATOR_WINDOW)
                .loc[trade_entry],  #! <--- comma
                trade_out,
            )

            if (
                str(relearn_entry)
                + " "
                + str(relearn_out)
                != relearn_time_labels.iloc[i]
            ):
                input(
                    "The dates and labels don't match, there might be an error. Submit at your own risk."
                )

            print(f"INDICATORS {relearn_entry, trade_out} CALCULATING...")

            ohlcv_for_indicators_IS = ohlcv.loc[indicators_entry_IS:indicators_out_IS]
            target_for_indicators_IS = targeted_df[relearn_time_labels.iloc[i]].dropna()

            df_for_indicators_IS = pd.concat(
                [ohlcv_for_indicators_IS, target_for_indicators_IS], axis=1
            )
            df_for_indicators_IS.columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "target",
            ]

            indicators_part_df = fe.generate_indis_new(
                df_for_indicators_IS,
                ticker_path,
                df_OOS=ohlcv.loc[indicators_entry_OOS:indicators_out_OOS],
                optimize_results_folder=indicators_optimized_params_folder,
                optimization=OPTIMIZE_INDICATORS,
                MAX_INDICATOR_WINDOW=1000,
            ).loc[
                relearn_entry:
            ] 

            parts_indicators_df.append(indicators_part_df)

            if SAVE["INDICATORS"]:
                indicators_part_df.to_csv(
                    f"{ticker_path}/{indicators_optimized_params_folder}{indicators_file}",
                    sep=";",
                )
            print(f"INDICATORS {relearn_entry, trade_out} ... DONE")

    indicators_df = pd.concat(parts_indicators_df, axis=1, keys=relearn_time_labels)

    return indicators_df


@time_me
def MODELS_BLOCK(
    symbol,
    relearner_dates_grid,
    trade_dates_grid,
    relearn_time_labels,
    current_libraries,
    ticker_path,
    FORCE,
    SAVE,
    possible_function_to_call_READ_MODELS,
    possible_function_to_call_TRAIN,
    indicators_df,
    targeted_df,
):
    os.makedirs(f"{ticker_path}/models/", exist_ok=True)
    print(f">>> START: MODELS BLOCK {symbol}")
    predicted_df_parts = []  # ? Here we will store pieces of the predicted_df.

    calculated_predicted_df = check_existance_of_file(
        path_to_file=ticker_path, file_name="predicted_df.csv"
    )

    for i in range(relearner_dates_grid.shape[0]):
        # ? Determining the dates we need to focus on.
        relearn_entry, relearn_out = relearner_dates_grid.iloc[i]
        current_date_range = relearn_time_labels.iloc[
            i
        ]  # The name of the column to lock onto.
        current_date_range_path = current_date_range.replace(":", "_")
        os.makedirs(
            ticker_path + f"models/{current_date_range_path}/PC/", exist_ok=True
        )
        # os.makedirs(ticker_path + f"models/{current_date_range_path}/AG/", exist_ok=True)

        calculated_models = True
        for library in current_libraries:
            for model in current_models[library]:
                if not os.path.exists(
                    ticker_path
                    + f"models/{current_date_range_path}/{library}/models/{model}/model.pkl"
                ):
                    calculated_models = False

        if (
            calculated_models and not FORCE["MODELS_TRAIN"]
        ): 
            print(
                f"All models for the symbol {symbol} and the time interval {current_date_range} are available and stored in their respective locations; they just need to be read."
            )
            if calculated_predicted_df and not FORCE["MODELS_TEST"]:
                print(
                    f"Since predicted_df for symbol {symbol} exists and {FORCE['MODELS_TEST']} is true, we don't even need to read the models"
                )
                predicted_df = pd.read_csv(
                    ticker_path + "predicted_df.csv",
                    sep=";",
                    index_col=[0],
                    parse_dates=[0],
                    dayfirst=True,
                    header=[0, 1],
                )

                return predicted_df

            elif os.path.exists(
                ticker_path + f"models/{current_date_range_path}/predicted_part_df.csv"
            ):
                predicted_df_part = pd.read_csv(
                    ticker_path
                    + f"models/{current_date_range_path}/predicted_part_df.csv",
                    sep=";",
                    parse_dates=[0],
                    header=[0, 1],
                    index_col=[0],
                    dayfirst=True,
                )
                predicted_df_parts.append(predicted_df_part)

                continue
            else:

                print(f">>> READING MODELS {symbol} за {current_date_range} ...")
                trained_models = md.READ_MODELS(
                    ticker_path,
                    current_date_range_path,
                    current_libraries,
                    possible_function_to_call_READ_MODELS,
                )
                print(f">>> READING MODELS {symbol} за {current_date_range} ... DONE")
        else:
            print(f">>> MODELS TRAIN {symbol}, {current_date_range} ...")
            current_date_range_indicators = (
                indicators_df[current_date_range]
                .dropna()
                .rename(lambda x: str(x), axis=1)
                .rename_axis("timestamp")
            )
            current_date_range_indicators = current_date_range_indicators.loc[
                relearn_entry:relearn_out
            ]

            df_for_train_models = np.round(
                pd.concat(
                    [
                        targeted_df[current_date_range].dropna().rename("target"),
                        current_date_range_indicators,
                    ],
                    axis=1,
                ),
                10,
            )

            trained_models = md.TRAIN(
                df_for_train_models,
                ticker_path,
                current_date_range_path,
                current_libraries,
                possible_function_to_call_TRAIN,
                cv_pct=0.2,
                label="target",
            )
            print(f">>> MODELS TRAIN {symbol}, {current_date_range} ... DONE")

        test_entry, test_out = trade_dates_grid.iloc[i]

        test_current_date_range_indicators = (
            indicators_df[current_date_range]
            .dropna()
            .rename(lambda x: str(x), axis=1)
            .rename_axis("timestamp")
        )
        test_current_date_range_indicators = np.round(
            test_current_date_range_indicators.loc[test_entry:test_out], 10
        )

        predicted_df_part = md.TEST(
            trained_models,
            possible_function_to_call_TEST,
            test_current_date_range_indicators,
        )
        predicted_df_parts.append(predicted_df_part)

    predicted_df = pd.concat(predicted_df_parts)

    predicted_df.to_csv(ticker_path + "predicted_df.csv", sep=";")

    print(f">>> DONE: MODELS BLOCK {symbol}")
    print("-" * 50)

    return predicted_df


def cook_mix_target(ohlcv):
    close = ohlcv["close"]
    width = 1
    distance = 1
    wlen = 24
    prominence = close.rolling(3).std().bfill().values
    prominence = np.where(prominence < 0.0005, 0.0005, prominence)

    peaks, _ = find_peaks(
        close, width=width, distance=distance, prominence=prominence, wlen=wlen
    )
    valley, _ = find_peaks(
        -close, width=width, distance=distance, prominence=prominence, wlen=wlen
    )
    long = pd.DataFrame(
        [1] * len(close[peaks]), columns=["long"], index=close[peaks].index
    )
    short = pd.DataFrame(
        [1] * len(close[valley]), columns=["short"], index=close[valley].index
    )

    mix_target = pd.concat([close, long, short], axis=1).fillna(0)
    return mix_target


def adjust_target(simple_target, target):
    adjusted_target = pd.DataFrame()
    for column in target.columns:
        new_column = simple_target.loc[target.loc[:, column].dropna().index]
        new_column.name = column
        adjusted_target = pd.concat([adjusted_target, new_column], axis=1)
    return adjusted_target

def main(symbol):

    ticker_path = TICKERS_PATH + symbol + "/"
    # fetching the data.
    ohlcv = DATA_BLOCK(
        symbol,
        FORCE,
        ticker_path,
        ohlcv_file,
        SAVE,
        start_date,
        freq,
        MAX_INDICATOR_WINDOW,
        TIMEFRAME,
        market_data_adress,
        query_id,
        drop_weekends=False,
    )

    # creating a grid of timestamps.
    (
        relearner_dates_grid,
        trade_dates_grid,  # Grids of dates for retraining and trading.
        backtest_IS_start_timestamp,
        backtest_IS_end_timestamp,  # Timestamps for the start and end of the first iteration of In-Sample (IS) phase.
        backtest_OOS_start_timestamp,  # The timestamp of the beginning of Out-of-Sample (OOS) phase.
    ) = DATES_GRID_BLOCK(
        symbol,
        ohlcv.iloc[MAX_INDICATOR_WINDOW:],
        relearn_duration,
        trade_duration,
        required_n_of_periods,
    )
    print(relearner_dates_grid)

    # creating labels from timestamps and mapping them to each other.
    (
        relearn_time_labels,
        trade_time_labels,
        relearn_to_trade_map,
        trade_to_relearn_map,
    ) = create_time_labels(relearner_dates_grid, trade_dates_grid, separetor=" ")

    for trend_side in ["long", "short"]:
        ticker_path = TICKERS_PATH + symbol + "/" + trend_side + "/"

        # labeling the target
        targeted_df = TARGET_BLOCK(
            ohlcv,
            symbol,
            FORCE,
            SAVE,
            ticker_path,
            target_file,
            relearner_dates_grid,
            relearn_time_labels,
            n_of_candles_to_add_to_target,
            trend_side,
        )

        indicators_df = INDICATORS_BLOCK(
            symbol,
            FORCE,
            SAVE,
            ticker_path,
            indicators_file,
            relearner_dates_grid,
            relearn_time_labels,
            trade_dates_grid,
            targeted_df,
            ohlcv,
            MAX_INDICATOR_WINDOW,
        )

        predicted_df = MODELS_BLOCK(
            symbol,
            relearner_dates_grid,
            trade_dates_grid,
            relearn_time_labels,
            current_libraries,
            ticker_path,
            FORCE,
            SAVE,
            possible_function_to_call_READ_MODELS,
            possible_function_to_call_TRAIN,
            indicators_df,
            targeted_df,
        )
        


# if __name__ == "__main__":
for symbol in symbols:
    main(symbol)