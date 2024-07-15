# Purpose
The script `instant_trader.py` launches the `main()` function from `live_trader.py` in multiple processes with the settings of the corresponding tickers.

The `main()` function fetches bars, generates signals based on already trained models, and sends them to the server.

The function takes `SYMBOL`, `TIMEFRAME`, `WORK_PATH`, and `my_id` (example: `'AUDUSD'`, `'M1'`, `'work_dir/'`, `'11'`).

## Working Principle

1. Request portfolio from the server.
2. Load trained models and indicator parameters from the disk:
   - If using long-short targets, load settings for both long and short models.
   - Model and indicator parameters are calculated using the `pycaret_pipeline.py` script.
3. Fetch bars for the working dataset `ohlcv_dataset`:
   - Request sufficient bars (`history_length`) to calculate indicators.
   - `history_length` is calculated from the indicator parameters loaded in the previous step.
4. Cyclical bar request:
   - Periodically query the server for a new completed candle.
5. Update the working dataset `ohlcv_dataset`:
   - Add the next bar to the end of `ohlcv_dataset`, discarding the earliest bar.
   - `ohlcv_dataset` always maintains a constant length.
6. Calculate indicators for `ohlcv_dataset`:
   - Calculate indicator values separately for long and short models as they may have different window parameters.
7. Generate signals for long and short:
   - Calculate probabilities for both models.
   - Convert probabilities to binary signals (0 and 1) based on the threshold.
8. Generate a unified signal:
   - Combine long and short model signals.
   - If both signals are equal, assign 0.5.
   - If `long == 1` and `short == 0`, final signal = 1; if `long == 0` and `short == 1`, final signal = -1.
9. Send the signal to the server.

The `main()` function accepts the following parameters:

- `ADRESS`: IP address of the server for fetching bars and sending signals.
- `SYMBOL`: ticker symbol, e.g., "AUDUSD".
- `TIMEFRAME`: ticker timeframe, e.g., "H1".
- `WORK_PATH`: path to the directory containing models (algorithm parameters and indicator parameters).
- `my_id`: unique identifier for GET request parameters, used to separate communication between different bots.

These parameters are used to configure the script and define the ticker for which signals will be generated based on the bars. `WORK_PATH` stores the necessary data for the script, such as models and indicator parameters, and `my_id` is used to identify bots and separate their communication on the server.

## Parameter Block

- `ADRESS`: IP of the server for fetching bars and sending signals.
- `SYMBOL`: ticker symbol (example: "AUDUSD").
- `TIMEFRAME`: ticker timeframe (example: "H1").
- `WORK_PATH`: directory containing models (algorithm parameters and indicator parameters).
- `my_id`: unique ID for GET request parameters (used to separate communication between different bots).
- `TICKERS_PATH`: common path to the working folder where models and associated information for the ticker are stored (example: `f"backtest/find_peaks_2folders/{SYMBOL}"`).
- `threshold`: 0.3.

## Restart on Rebalance

After rebalancing, you need to:

1. Stop the `instant_trader.py` script.
2. Run the `resetter.py` script if portfolio parameters have changed.
3. Restart the kernel in the notebook.
4. Restart the `instant_trader.py` script.
