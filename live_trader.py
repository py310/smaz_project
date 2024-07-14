import datetime
import json
import logging
import multiprocessing
import os
import pickle
import time
import timeit
import urllib.parse
import warnings
from concurrent.futures import ProcessPoolExecutor as Pool
from urllib.request import URLError, urlopen

import numpy as np
import pandas as pd
import requests
from utils import data_acquisition as da
from utils import data_preparation as dp
from utils import features_engeneering as fe
from utils import loop_funcs as lf
from utils import modeling as md

pd.options.mode.chained_assignment = None

from pycaret import *
from pycaret.classification import *

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

ADRESS = config["java_ip"]["trading_ip"]
#WORK_PATH = config["directories"]["work_directory"]
#work_directory = config["directories"]["work_directory"]
my_id = config["java_ip"]["my_id"]

def only_close():
    requests.get(ADRESS + "onlyclose")


def send_portfolio():
    portfolios = PORTFOLIOS

    requests.post(ADRESS + "rebalance", json=portfolios)


def load_models(ticker_path, date_range):
    """
    Loading ML models

    Args:
        ticker_path(str): path to the ticker folder
        date_range(str): period of interest (last six months)

    Returns:
        pycaret_models(list): dictionary with trained PyCaret models
    """

    pycaret_models_names = [
        model_name
        for model_name in os.listdir(f"{ticker_path}/models/{date_range}/PC/models")
    ]

    pycaret_models = {}

    for model_name in pycaret_models_names:
        print(model_name)
        model = load_model(
            f"{ticker_path}/models/{date_range}/PC/models/{model_name}/model"
        )
        pycaret_models[model_name] = model

    return pycaret_models


def read_indis_params(ticker_path, date_range):
    """
    Args:
        ticker_path(str): path to the ticker folder
        date_range(str): period of interest (last six months)

    Returns:
        parameters_1arg(list) - list of parameters for indicators with 1 parameter
        parameters_2arg(list) - list of parameters for indicators with 2 parameters
    """
    parameters_1arg = (
        pd.read_csv(
            ticker_path + f"optimized_indicators_params/{date_range}/1_arg.csv",
            sep=";",
            index_col=0,
        )
        .sort_values(by="indicator_number")
        .loc[:, "Best_parameter_value"]
        .to_list()
    )

    parameters_2arg = (
        pd.read_csv(
            ticker_path + f"optimized_indicators_params/{date_range}/2_arg.csv",
            sep=";",
            index_col=0,
        )
        .sort_values(by="indicator_number")
        .loc[:, ["Best_parameter_value_1", "Best_parameter_value_2"]]
        .to_numpy()
        .astype(int)
        .tolist()
    )

    return parameters_1arg, parameters_2arg


def calculate_history_length(parameters_1arg, parameters_2arg):
    """
    Args:
        parameters_1arg(list) - list of parameters for indicators with 1 parameter
        parameters_2arg(list) - list of parameters for indicators with 2 parameters

    Returns:
        history_length(int) - maximum parameter value for these indicators + 14
        (for the number of candles requested to calculate indicators)
    """

    all_indis_params = parameters_1arg.copy()
    for indi in parameters_2arg:
        all_indis_params = all_indis_params + [
            sum(indi)
        ]  # только для stochf_k берем сумму параметров

    max_window = max(all_indis_params)
    history_length = int(max_window + 14)

    return history_length


def make_preparation(ticker_path):
    """
    Unpacking models, indicator parameters, and calculating the number of bars for operation

    Args:
        ticker_path(str): path to the ticker folder

    Returns:
        pycaret_models(list): dictionary of trained PyCaret models
        parameters_1arg(list) - list of parameters for indicators with 1 parameter
        parameters_2arg(list) - list of parameters for indicators with 2 parameters
        history_length(int) - maximum parameter value for these indicators + 14
        (for the number of candles requested to calculate indicators)
    """
    current_date_range = max(os.listdir(f"{ticker_path}/models/"))
    pycaret_all_models = load_models(ticker_path, current_date_range)
    parameters_1arg, parameters_2arg = read_indis_params(
        ticker_path, current_date_range
    )
    history_length = calculate_history_length(parameters_1arg, parameters_2arg)

    return (
        pycaret_all_models,
        parameters_1arg,
        parameters_2arg,
        history_length,
    )


def load_models(ticker_path, date_range):
    """
    Loading ML models

    Args:
        ticker_path(str): path to the ticker folder
        date_range(str): period of interest (six months)

    Returns:
        pycaret_models(list): dictionary of trained PyCaret models
    """

    pycaret_models_names = [
        model_name
        for model_name in os.listdir(f"{ticker_path}/models/{date_range}/PC/models")
    ]

    pycaret_models = {}

    for model_name in pycaret_models_names:
        print(model_name)
        model = load_model(
            f"{ticker_path}/models/{date_range}/PC/models/{model_name}/model"
        )
        pycaret_models[model_name] = model

    return pycaret_models


set_buffer = da.set_buffer
get_data = da.get_data
candles_to_df = da.candles_to_df


def get_ohlcv_dataset(ADRESS, history_length, request_tail):
    """
    The function receives a dataset with a number of candles equal to history_length up to the current moment.

    Args:
        history_length(int): the maximum parameter value for these indicators + 14
            (for the number of candles requested to calculate the indicators)
        request_tail(str): parameters for the server request, including symbol and timeframe
            (example - 'symbol=EURCHF&timeframe=M5')

    Returns:
        ohlcv_dataset(DataFrame): Bars prepared for processing in the pipeline in the required quantity
    """
    set_buffer(ADRESS, history_length, request_tail)
    ohlcv_dataset = pd.DataFrame()
    print(ohlcv_dataset)

    while ohlcv_dataset.shape[0] < history_length:
        time.sleep(5)
        data = get_data(ADRESS, request_tail)
        if data != []:
            if "High" in data[0]:
                print("Get")
                new_candles = candles_to_df(data)
                ohlcv_dataset = pd.concat([ohlcv_dataset, new_candles])
    return ohlcv_dataset


def get_proba_json(test_df, pycaret_models):
    """
    Args:
        test_df(DataFrame): dataset on which predictions are made
        ticker_path(str): path to the ticker folder
        pycaret_models(list): dictionary of trained PyCaret models

    Returns:
        proba_json(str): a JSON string with prediction probabilities for sending to the server
    """
    pycaret_predict_results = []
    pycaret_model_names = []
    print(f"{pycaret_models}")
    for model_name, model in pycaret_models.items():
        predicted_test = predict_model(model, data=test_df, encoded_labels=True)

        try:
            pycaret_predicted_test = predicted_test[
                ["prediction_label", "prediction_score"]
            ]
            pycaret_predict_results.append(pycaret_predicted_test)
            pycaret_model_names.append(model_name)

        except:
            print(f"Predict error at model - {model_name}")

    pycaret_predict_results = pd.concat(
        pycaret_predict_results, axis=1, keys=pycaret_model_names
    )
    pycaret_predict_results = pd.concat([pycaret_predict_results], axis=0)

    models = pycaret_predict_results.swaplevel(axis=1)["prediction_score"].columns

    together_df = pd.concat(
        [
            md.get_score_of_label_one(pycaret_predict_results[model], shifted=False)
            for model in models
        ],
        axis=1,
        keys=models,
    )

    json_obj = together_df.swaplevel(axis=1)["score_of_one"].to_json(
        orient="records", date_format="iso", lines=True
    )

    return json.loads(json_obj)


def get_signal_proba(SYMBOL, ohlcv_dataset, parameters_1arg, parameters_2arg, pycaret_models, TICKERS_PATH):

    indi_df = fe.generate_indis(ohlcv_dataset, parameters_1arg, parameters_2arg)
    indi_df.columns = indi_df.columns.astype("str")
    dp.write_log(f"indis -- {indi_df.tail(1).to_json()}", TICKERS_PATH)

    signal_proba = get_proba_json(np.round(indi_df.tail(1), 10), pycaret_models)
    return signal_proba


def calc_ultimate_proba(long_signal_proba, short_signal_proba):
    threshold = 0.3
    ultimate_proba_dict = {}
    ultimate_proba = 0
    for model, proba_long in long_signal_proba.items():
        proba_short = short_signal_proba[model]
        if proba_long > threshold:
            ultimate_proba -= 1
        if proba_short > threshold:
            ultimate_proba += 1

        if ultimate_proba == 0:
            ultimate_proba = 0.5
        if ultimate_proba == -1:
            ultimate_proba = 0

        ultimate_proba_dict[f"{model}~PC"] = ultimate_proba

        ultimate_proba = 0

    return ultimate_proba_dict

def previous_quarter(ref):
    if ref.month < 4:
        return datetime.date(ref.year - 1, 12, 31)
    elif ref.month < 7:
        return datetime.date(ref.year, 3, 31)
    elif ref.month < 10:
        return datetime.date(ref.year, 6, 30)
    return datetime.date(ref.year, 9, 30)

def main(SYMBOL, TIMEFRAME, WORK_PATH, my_id):

    request_tail = (
        "symbol=" + str(SYMBOL) + "&timeframe=" + str(TIMEFRAME) + "&id=" + my_id
    )
    TICKERS_PATH = f"{WORK_PATH}{SYMBOL}"
    TICKERS_PATH_LONG = f"{TICKERS_PATH}/long/"
    TICKERS_PATH_SHORT = f"{TICKERS_PATH}/short/"
    date_range_long = os.listdir(TICKERS_PATH_LONG + "optimized_indicators_params/")[-1]
    date_range_short = os.listdir(TICKERS_PATH_SHORT + "optimized_indicators_params/")[
        -1
    ]
    if date_range_long != date_range_short:
        print("Last interval for long != for short")
    else:
        date_range = date_range_long

    existed_last_range = date_range.split(" ")[-2][:-3]
    actual_last_range = f"{previous_quarter(datetime.datetime.today()).year}-{previous_quarter(datetime.datetime.today()).month}"
    if actual_last_range != existed_last_range:
        print(f"{actual_last_range} non equal {existed_last_range}")
    else:
        print(f"{actual_last_range} equal {existed_last_range}")

    # 1st launch
    portfolio_respond = requests.get(ADRESS + "getportfolio")
    portfolio_df = pd.DataFrame(json.loads(portfolio_respond.text))

    (  # Fetching new models, indicator parameters, and window length for indicator calculations.
        pycaret_models_long,
        parameters_1arg_long,
        parameters_2arg_long,
        history_length_long,
    ) = make_preparation(
        TICKERS_PATH_LONG
    )

    (  # Fetching new models, indicator parameters, and window length for indicator calculations.
        pycaret_models_short,
        parameters_1arg_short,
        parameters_2arg_short,
        history_length_short,
    ) = make_preparation(
        TICKERS_PATH_SHORT
    )

    history_length = max(history_length_short, history_length_short)
    # Retrieving models from JSON that will be used for trading in the current period.

    # autogluon_portfolio_models = []
    pycaret_portfolio_models = []
    for model_dict in portfolio_df[
        (portfolio_df.symbol == SYMBOL) & (portfolio_df.timeframe == TIMEFRAME)
    ][["negative_group", "positive_group"]].values.flat:
        if isinstance(model_dict, dict):
            for key in model_dict.keys():
                if "~" in key:
                    pycaret_portfolio_models.append(key.split("~")[0])
                elif "‾" in key:
                    pycaret_portfolio_models.append(key.split("‾")[0]) # For local machine where tilda reads as underscore
                else:
                    print(f"Unknow AutoML separator")
        elif model_dict is None:
            pass
        else:
            raise TypeError(f"{model_dict}")

    pycaret_portfolio_models = list(set(pycaret_portfolio_models))

    pycaret_models_long = {
        pycaret_portfolio_model: pycaret_models_long[pycaret_portfolio_model]
        for pycaret_portfolio_model in pycaret_portfolio_models
    }

    pycaret_models_short = {
        pycaret_portfolio_model: pycaret_models_short[pycaret_portfolio_model]
        for pycaret_portfolio_model in pycaret_portfolio_models
    }

    ohlcv_dataset = get_ohlcv_dataset(
        ADRESS, history_length, request_tail
    ) 
    print("Got Portfolio OHLCV")

    while True:
        try:
            data = get_data(ADRESS, request_tail)
            if data != []:
                if "High" in data[0]: 

                    print("Got Last 1H-bar", datetime.datetime.now())
                    new_candles = candles_to_df(data)
                    ohlcv_dataset = pd.concat([ohlcv_dataset, new_candles])
                    ohlcv_dataset = ohlcv_dataset[-history_length:]

                    long_signal_proba = get_signal_proba(
                        SYMBOL,
                        ohlcv_dataset,
                        parameters_1arg_long,
                        parameters_2arg_long,
                        pycaret_models_long,
                        TICKERS_PATH
                    )
                    short_signal_proba = get_signal_proba(
                        SYMBOL,
                        ohlcv_dataset,
                        parameters_1arg_short,
                        parameters_2arg_short,
                        pycaret_models_short,
                        TICKERS_PATH
                    )

                    dp.write_log(f"long_signal_proba -- {long_signal_proba}", TICKERS_PATH)
                    dp.write_log(f"short_signal_proba -- {short_signal_proba}", TICKERS_PATH)

                    ultimate_proba = calc_ultimate_proba(
                        long_signal_proba, short_signal_proba
                    )
                    dp.write_log(f"ultimate_proba -- {ultimate_proba}", TICKERS_PATH)

                    proba_json = json.dumps(
                        {
                            "exec_time": str(ohlcv_dataset.tail(1).index[0]),
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "probas": ultimate_proba,
                        }
                    )

                    print("Json Complete", datetime.datetime.now())
                    print(proba_json)
                    try:
                        respond = requests.post(ADRESS + "result", json=proba_json)
                        if respond.ok:
                            print(respond.text)
                        else:
                            print("Not Responded")

                    except:
                        time.sleep(0.03)
                        respond = requests.post(ADRESS + "result", json=str(proba_json))
                        if respond.ok:
                            print(respond.text)
                            print(datetime.datetime.now())
                        else:
                            print("Not Responded")

                elif "portfolio" in data[0]:
                    print("Get PORTFOLIO")

                    portfolio_df = pd.DataFrame(data)

                    (  # Fetching new models, indicator parameters, and window length for indicator calculations.
                        pycaret_models_long,
                        parameters_1arg_long,
                        parameters_2arg_long,
                        history_length_long,
                    ) = make_preparation(
                        TICKERS_PATH_LONG
                    )

                    (  # Fetching new models, indicator parameters, and window length for indicator calculations.
                        pycaret_models_short,
                        parameters_1arg_short,
                        parameters_2arg_short,
                        history_length_short,
                    ) = make_preparation(
                        TICKERS_PATH_SHORT
                    )

                    history_length = max(history_length_short, history_length_short)
                    # Retrieving models from JSON that will be used for trading in the current period.

                    # autogluon_portfolio_models = []
                    pycaret_portfolio_models = []
                    for model_dict in portfolio_df[
                        (portfolio_df.symbol == SYMBOL)
                        & (portfolio_df.timeframe == TIMEFRAME)
                    ][["negative_group", "positive_group"]].values.flat:
                        if isinstance(model_dict, dict):
                            for key in model_dict.keys():
                                if "~" in key:
                                    pycaret_portfolio_models.append(key.split("~")[0])
                                elif "‾" in key:
                                    pycaret_portfolio_models.append(key.split("‾")[0]) # For local machine where tilda reads as underscore
                                else:
                                    print(f"Unknow AutoML separator")
                        elif model_dict is None:
                            pass
                        else:
                            raise TypeError(f"{model_dict}")

                    pycaret_portfolio_models = list(set(pycaret_portfolio_models))

                    pycaret_models = {
                        pycaret_portfolio_model: pycaret_models[pycaret_portfolio_model]
                        for pycaret_portfolio_model in pycaret_portfolio_models
                    }

                    ohlcv_dataset = get_ohlcv_dataset(
                        ADRESS, history_length, request_tail
                    )
                    print("Got Portfolio OHLCV")

            time.sleep(5)

        except:
            pass