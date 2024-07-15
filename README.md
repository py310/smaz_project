# Automated Trading System

## Overview

This project automates trading operations using machine learning models and technical analysis indicators. The main script handles data acquisition, model loading, prediction generation, and trading signal execution based on real-time market data.

## Project Structure


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
