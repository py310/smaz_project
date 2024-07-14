from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import pymysql
import pytz
from sqlalchemy import create_engine

pd.set_option("display.max_columns", 500)  # number of columns to be displayed
pd.set_option("display.width", 1500)  # max table width to display
# import pytz module for working with time zone

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# set time zone
timezone = pytz.timezone("Etc/UTC")

today = datetime.today()
yesterday = today - timedelta(days=1)
yesterday_midnight = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
today_midnight = yesterday_midnight + timedelta(days=2)
date_from_midnight = yesterday_midnight - timedelta(days=100)
# # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
date_from = timezone.localize(date_from_midnight)
date_to = timezone.localize(today_midnight)
yesterday_midnight = timezone.localize(yesterday_midnight)

print(f"{date_from}")
print(f"{date_to}")

orders_columns = [
    "ticket",
    "time_setup",
    "time_setup_msc",
    "time_done",
    "time_done_msc",
    "time_expiration",
    "type",
    "type_time",
    "type_filling",
    "state",
    "magic",
    "position_id",
    "position_by_id",
    "reason",
    "volume_initial",
    "volume_current",
    "price_open",
    "sl",
    "tp",
    "price_current",
    "price_stoplimit",
    "symbol",
    "comment",
    "external_id",
]

deals_columns = [
    "ticket",
    "order",
    "time",
    "time_msc",
    "type",
    "entry",
    "magic",
    "position_id",
    "reason",
    "volume",
    "price",
    "commission",
    "swap",
    "profit",
    "fee",
    "symbol",
    "comment",
    "external_id",
]
positions_columns = [
    "ticket",
    "time",
    "time_msc",
    "time_update",
    "time_update_msc",
    "type",
    "magic",
    "identifier",
    "reason",
    "volume",
    "price_open",
    "sl",
    "tp",
    "price_current",
    "swap",
    "profit",
    "symbol",
    "comment",
    "external_id",
]

if not mt5.initialize(
    path=r"g:\Rublevskiy\alter\82\Alpari MT5\terminal64.exe", portable=True
):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

deals = mt5.history_deals_get(date_from, date_to)
deals_df = pd.DataFrame(deals, columns=deals_columns)
deals_df["timestamp"] = pd.to_datetime(deals_df.time, unit="s")
deals_df = deals_df.set_index("timestamp")
deals_df["balance"] = (
    (deals_df.profit + deals_df.commission + deals_df.swap).cumsum().round(2)
)

deals_df.to_csv("mt5_deals.csv")