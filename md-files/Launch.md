# Preliminary Requirements

- **Availability of Machine Learning Models**
  - To obtain models, run the `pycaret_pipeline.ipynb` notebook or have pre-trained models from prior research.
  - The model includes indicator parameters saved in files like `{symbol}/long|short/optimized_indicators_params/{last_range}/1|2_arg.csv` and ML algorithm parameters saved in `{symbol}/long|short/models/{last_range}/PC/models`.

- **Run the `resetter.ipynb` Notebook**
  - This will close all previous trades. Additionally, you can update portfolio positions in this notebook.

# Initial Run

1. Start the Java server and MetaTrader 5.
2. Obtain model parameters: load into appropriate folders if they exist, or run the `pycaret_pipeline.ipynb` script if not yet trained.
3. Run the `resetter.ipynb` file (make sure `PORTFOLIO` is prepared).
4. Launch all `instant_trader.ipynb` notebooks. In this implementation, each ticker has its own notebook, where `SYMBOL` and `TIMEFRAME` variables are set accordingly.

# Restart After Failure

1. Start the Java server and MetaTrader 5.
2. Restart all `instant_trader.ipynb` notebooks.

# Rebalancing Procedure

- All programs should be running and operational.

1. On the first day of the rebalancing period (currently the 1st of each new quarter), run the `pycaret_pipeline.ipynb` script.
2. In the `resetter.ipynb` notebook, run the `onlyclose()` function to switch the server to close-only mode based on signals.
3. The `instant_trader.ipynb` notebooks will continue sending signals with previous settings.
4. After retraining (step 1), obtain the weights for the portfolio.
5. Run the `resetter.ipynb` file with the updated `PORTFOLIO`.
6. Restart all `instant_trader.ipynb` notebooks (it is advisable to restart the notebook kernels).
