# Purpose
The script `pycaret_pipeline.py` retrains the models.

## Working Principle

1. **Market Data Request**: Use the `market-data` request to get market price information in OHLCV format for the required period.
2. **Target Definition**: Use the `find_peaks` function to identify two targets: peaks for the short model and valleys for the long model.
3. **Indicator Calculation and Optimization**:
   - Calculate indicators for the given period with different parameters (typically windows).
   - Compute Spearman correlations between targets and indicators.
   - Select the best parameters for each indicator based on these correlations and save them to the working directory `work_dir/` (from which they will be read by the `instant_trader.py` bot).
4. **Model Training**:
   - Use the AutoML library PyCaret with previously optimized indicator parameters and the target for training.
   - Save algorithm parameters to the working directory `work_dir/` (from which they will be read by the `instant_trader.py` bot).

## Parameters for Execution

- `market_data_adress`: Address of the host providing bars.
- `symbols`: List of symbols to be used in the pipeline.
- `TIMEFRAME`: Time interval for data analysis (e.g., "1H" or "5M").
- `freq`: Data frequency (e.g., "D" for daily data or "H" for hourly data).
- `start_date`: Start date for the analysis period.
- `MAX_INDICATOR_WINDOW`: Number of bars added before the first bar of the period for analysis. Some indicators use windows for calculation, so some historical data is needed before the first candle of the required period. This parameter provides this data.
- `OPTIMIZE_INDICATORS`: Specifies whether to optimize indicators (`True` or `False`).
- `n_of_candles_to_add_to_target`: Number of bars added to the start and end of the OHLCV dataframe. This parameter is determined by the settings of the function that marks the target and "looks" into the future and past relative to the timestamp for which it is calculated.
- `relearn_duration`: Period for the train set (IS).
- `trade_duration`: Period for the test set (OOS).
- `required_n_of_periods`: Number of periods needed to create a portfolio.
- `TICKER_PATH`: Path to the project working folder where intermediate results (e.g., OHLCV, indicator parameters, trained models, etc.) will be saved.
- `ohlcv_file`: Name of the file to record OHLCV.
- `target_file`: Name of the file to record targets.
- `indicators_file`: Name of the file to record indicators.
- `possible_models`: AutoML systems models (e.g., AutoGluon or PyCaret) to be used in the pipeline. The current version uses only PyCaret.
- `FORCE`: Settings to enable/disable pipeline stages. Some stages can be skipped if their results were obtained earlier, imported from outside, or are not needed in the current run.
- `SAVE`: Save settings.
