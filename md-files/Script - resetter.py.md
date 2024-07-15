# Purpose
The script `resetter.py`:

- Closes all open trades.
- Sends the portfolio to the Java server.
- Is used at the initial start and when changing portfolio parameters (e.g., after retraining).

The script reads the host from the `config.ini` file, where it sends the `onlyclose` command to close all open orders and the `rebalance` command to send the portfolio to the server.

The portfolio is sent in JSON format as follows:
```python
PORTFOLIOS = [
    {
        "negative_group": {
            "GradientBoostingClassifier~PC": 1
        },
        "positive_group": None,
        "portfolio": "('TOP_N', 'AG|PC', '0.5, 0.5', '1, 0')",
        "symbol": "EURCHF",
        "timeframe": "H1",
        "weight": 1/3,
    },
    {
        "negative_group": {
            "GradientBoostingClassifier~PC": 1
        },
        "positive_group": None,
        "portfolio": "('TOP_N', 'AG|PC', '0.5, 0.5', '1, 0')",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "weight": 1/3,
    },
    {
        "negative_group": {
            "GradientBoostingClassifier~PC": 1
        },
        "positive_group": None,
        "portfolio": "('TOP_N', 'AG|PC', '0.5, 0.5', '1, 0')",
        "symbol": "NZDCAD",
        "timeframe": "H1",
        "weight": 1/3,
    },
]
