## Overview

This project automates trading operations using machine learning models and technical analysis indicators. The main script handles data acquisition, model loading, prediction generation, and trading signal execution based on real-time market data.

## Challenges and Problem Statement

**The primary challenge of the project lies in:**

1. **Data Quality and Consistency:** Ensuring that historical data from MT5 is accurate, complete, and appropriately processed for model training and decision-making.

2. **Model Accuracy and Adaptability:** Optimizing machine learning models to accurately predict market movements under various market conditions and adapting to changing market dynamics.

3. **Integration and Automation:** Seamlessly integrating the predictive models with the trading platform (MT5) to automate trading decisions while maintaining reliability and security.

## Conclusion

By leveraging machine learning models trained with PyCaret and historical data from MetaTrader 5, the project aims to streamline and enhance trading strategies through automation and data-driven decision-making. Ensuring robustness in data handling, model training, and execution is crucial for achieving consistent and effective automated trading results.


## Project Structure
- [Installation](https://github.com/py310/smaz_project/blob/main/md-files/Installation.md)
- [Launch](https://github.com/py310/smaz_project/blob/main/md-files/Launch.md)
- [Launch params](https://github.com/py310/smaz_project/blob/main/md-files/Launch%20params.md)
- [Script - live_trader.py](https://github.com/py310/smaz_project/blob/main/md-files/Script%20-%20live_trader.py.md)
- [Script - instant_trader.py](https://github.com/py310/smaz_project/blob/main/md-files/Script%20-%20instant_trader.py.md)
- [Script - pycaret_pipeline.py](https://github.com/py310/smaz_project/blob/main/md-files/Script%20-%20pycaret_pipeline.py.md)
- [Script - resetter.py](https://github.com/py310/smaz_project/blob/main/md-files/Script%20-%20resetter.py.md)

## Project Process Diagram
![process_diagram](https://github.com/user-attachments/assets/4a3e7a4a-6417-4710-a6b6-1d6eab0a6ca2)

## Dependencies

[environment.yml](https://github.com/py310/smaz_project/blob/main/environment.yml)

## Configuration

The project utilizes `config.ini` for configuration settings such as server addresses (`trading_ip`), user IDs (`my_id`), and file directories (`WORK_PATH`).

## Future Enhancements

- Implement additional machine learning models for enhanced prediction accuracy.
- Expand technical indicators and optimize parameter selection.
- Incorporate real-time data visualization for monitoring trading performance.
