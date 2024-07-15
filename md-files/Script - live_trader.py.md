# Purpose

The main Python script `live_trader.py` orchestrates the entire trading system. It includes functions for:

## Working Principle

### Server Communication
- `only_close()`: Sends a request to close trades.
- `send_portfolio()`: Sends updated portfolio information to the server.

### Model Management
- `load_models(ticker_path, date_range)`: Loads trained machine learning models from specified directories.
- `read_indis_params(ticker_path, date_range)`: Reads optimized parameters for technical indicators.

### Data Handling
- `get_ohlcv_dataset(ADRESS, history_length, request_tail)`: Retrieves and processes historical OHLCV data required for analysis.
- `get_proba_json(test_df, pycaret_models)`: Generates prediction probabilities using loaded models on a given dataset.
- `get_signal_proba(SYMBOL, ohlcv_dataset, parameters_1arg, parameters_2arg, pycaret_models, TICKERS_PATH)`: Calculates trading signals based on indicator data and model predictions.

### Main Trading Logic
- `main(SYMBOL, TIMEFRAME, WORK_PATH, my_id)`: Main function that initiates the trading process. It manages the flow of data acquisition, model loading, real-time data processing, and decision-making based on generated signals.
