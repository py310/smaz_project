import configparser
import time

import requests

config = configparser.ConfigParser()
config.read("config.ini")

ADRESS = config["java_ip"]["trading_ip"]

def only_close():
    requests.get(ADRESS + "onlyclose")


def send_portfolio():
    portfolios = PORTFOLIOS
    requests.post(ADRESS + "rebalance", json=portfolios)
    
PORTFOLIOS = [
    {
        "negative_group": {"GradientBoostingClassifier~PC": 1},
        "positive_group": None,
        "portfolio": "('TOP_N', 'AG|PC', '0.5, 0.5', '1, 0')",
        "symbol": "EURCHF",
        "timeframe": "H1",
        "weight": 1 / 3,
    },
    {
        "negative_group": {"GradientBoostingClassifier~PC": 1},
        "positive_group": None,
        "portfolio": "('TOP_N', 'AG|PC', '0.5, 0.5', '1, 0')",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "weight": 1 / 3,
    },
    {
        "negative_group": {"GradientBoostingClassifier~PC": 1},
        "positive_group": None,
        "portfolio": "('TOP_N', 'AG|PC', '0.5, 0.5', '1, 0')",
        "symbol": "NZDCAD",
        "timeframe": "H1",
        "weight": 1 / 3,
    },
]

only_close()
time.sleep(2)
send_portfolio()