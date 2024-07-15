# Installation and Setup Guide for SMAZ Project Environment

## Installing Anaconda

1. Install the latest Anaconda distribution by following the instructions on the official Anaconda website [Anaconda Documentation](https://docs.anaconda.com/anaconda/).
2. During installation, do not select the options "Add to PATH" and "Register Anaconda as the default Python interpreter".

## Creating the Environment

1. In the `environment.yml` file, specify the correct prefix (path to Anaconda on your computer) and the environment name. For example, `name: smaz` [here guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
2. Create the environment from the `environment.yml` file using the command `conda env create -f environment.yml`.

## Project Deployment

1. Checkout the required branch from Git into your project's working directory.
2. Verify the presence and currency of project files, including `resetter.py`, `pycaret_pipeline.py`, `instant_trader.py`, `live_trader.py`, and files in the `utils/` folder.

## Loading Configuration File

1. Download the `config.ini` file into your project's working directory.

## Creating a Folder for Saving Model Parameters

1. Create an empty folder where model parameters will be saved for each ticker (e.g., `work_dir/`).

## Creating Notebook Kernel

1. Create a notebook kernel using the command `python -m ipykernel install --user --name yourenvname --display-name "display-name"`.
   - Replace `yourenvname` with the name of your created environment.
   - Replace `"display-name"` with the name of the kernel as it will appear in Jupyter Notebook.

## Post-Training Setup

After running the training (using `pycaret_pipeline.py`), the system will create folders for each ticker in the `work_dir` folder (e.g., `EURUSD`). These folders will contain the following:
- Automatically created `long` and `short` folders:
  - `models` (for saving models),
  - `optimized_indicators_params` (with parameters for indicators),
  - Files: `ohlcv.csv`, `target.csv`, and `predicted_df.csv`.

Additionally:
- The Pycaret library will create a `catboost_info` folder to save intermediate results for the CatBoost library, and a `logs.log` file for logging the training process.

## Running .bat Files

To simplify script execution, download the following [three .bat files](https://github.com/py310/smaz_project/tree/main/bat-files) at the same level where Anaconda is installed:

- `run_pycaret.bat`: Launches the trading script.
- `run_pycaret_resetter.bat`: Executes the script that closes trades and sends the portfolio to the server.
- `run_pycaret_pipeline.bat`: Initiates the script that re-trains the models.

These .bat files provide convenient shortcuts for running essential scripts in the ATASS project environment.
