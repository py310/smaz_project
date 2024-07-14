# Automated Trading System

## Overview

This project automates trading operations using machine learning models and technical analysis indicators. The main script handles data acquisition, model loading, prediction generation, and trading signal execution based on real-time market data.

## Project Structure

### Executable Script

The main Python script orchestrates the entire trading system. It includes functions for:

- **Server Communication**
  - `only_close()`: Sends a request to close trades.
  - `send_portfolio()`: Sends updated portfolio information to the server.

- **Model Management**
  - `load_models(ticker_path, date_range)`: Loads trained machine learning models from specified directories.
  - `read_indis_params(ticker_path, date_range)`: Reads optimized parameters for technical indicators.

- **Data Handling**
  - `get_ohlcv_dataset(ADRESS, history_length, request_tail)`: Retrieves and processes historical OHLCV data required for analysis.
  - `get_proba_json(test_df, pycaret_models)`: Generates prediction probabilities using loaded models on a given dataset.
  - `get_signal_proba(SYMBOL, ohlcv_dataset, parameters_1arg, parameters_2arg, pycaret_models, TICKERS_PATH)`: Calculates trading signals based on indicator data and model predictions.

- **Main Trading Logic**
  - `main(SYMBOL, TIMEFRAME, WORK_PATH, my_id)`: Main function that initiates the trading process. It manages the flow of data acquisition, model loading, real-time data processing, and decision-making based on generated signals.

### Usage

To run the script:
1. Ensure all necessary configurations are set in `config.ini`.
2. Execute the `main()` function with appropriate parameters (`SYMBOL`, `TIMEFRAME`, `WORK_PATH`, `my_id`).

### Dependencies

- **Python Libraries**
  - `numpy`
  - `pandas`
  - `requests`
  - `pycaret`
  - `configparser`
  - Other standard and custom utility libraries (`utils`)

### Configuration

The project utilizes `config.ini` for configuration settings such as server addresses (`trading_ip`), user IDs (`my_id`), and file directories (`WORK_PATH`).

## Future Enhancements

- Implement additional machine learning models for enhanced prediction accuracy.
- Expand technical indicators and optimize parameter selection.
- Incorporate real-time data visualization for monitoring trading performance.
