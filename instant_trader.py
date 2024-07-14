import json
import configparser
from concurrent.futures import ProcessPoolExecutor

import live_trader

config = configparser.ConfigParser()
config.read("config.ini")

ADRESS = config["java_ip"]["trading_ip"]
WORK_PATH = config["directories"]["work_directory"]
work_directory = config["directories"]["work_directory"]
my_id = config["java_ip"]["my_id"]

my_id = "trader"
symbols = [
    "EURCHF",
    "EURUSD",
    "NZDCAD"
]
TIMEFRAME = "H1"

max_workers = len(symbols)

params = [
    (
        symbol,
        TIMEFRAME,
        WORK_PATH,
        str(my_id),
    )
    for symbol in symbols
]


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for param in params:
            print(param)
            smth = executor.submit(
                live_trader.main, param[0], param[1], param[2], param[3]
            )