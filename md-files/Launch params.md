All Parameters "Hardcoded into Notebooks"

## Target Parameters

Target parameters can be modified in the `cook_mix_target()` function in the `pycaret_pipeline.ipynb` notebook.

In the current version of `smaz_project`, a target is used that identifies peaks and valleys on the graph.

![peaks_valleys](https://github.com/user-attachments/assets/f6c246fc-b6e7-4108-8176-8cc6f7780672)

From peaks marked on the graph, the first part of the model is formed, called `short` (signals that we are at a peak and should sell), and from valleys, the second part - `long` model (signals that we are at a local low point and should sell).

This is implemented using the `find_peaks` function from the `scipy` library (documentation).

### Current Settings:

Parameters for the `find_peaks` function:

```python
width = 1
distance = 1
wlen = 24
prominence = close.rolling(3).std().bfill().values
prominence = np.where(prominence < 0.0005, 0.0005, prominence)
```

### Target Labeling Function:

```python
def cook_mix_target(ohlcv):
    close = ohlcv["close"]
    width = 1
    distance = 1
    wlen = 24
    prominence = close.rolling(3).std().bfill().values
    prominence = np.where(prominence < 0.0005, 0.0005, prominence)
 
    peaks, _ = find_peaks(
        close, width=width, distance=distance, prominence=prominence, wlen=wlen
    )
    valley, _ = find_peaks(
        -close, width=width, distance=distance, prominence=prominence, wlen=wlen
    )
    long = pd.DataFrame(
        [1] * len(close[peaks]), columns=["long"], index=close[peaks].index
    )
    short = pd.DataFrame(
        [1] * len(close[valley]), columns=["short"], index=close[valley].index
    )
 
    mix_target = pd.concat([close, long, short], axis=1).fillna(0)
    return mix_target
```

## Indicator Parameters

Indicator parameters are adjusted during retraining as part of optimization (running the `pycaret_pipeline.ipynb` notebook).

They are saved in the working directory `{symbol}/long|short/optimized_indicators_params/{last_range}/` in files `1_arg.csv` and `2_arg.csv`.

Parameters during optimization:
- 1-argument indicators are iterated with values in `range(2, 300, 10)`
- 2-argument indicators are iterated with values in `range(3, 200, 5)`

## Training Parameters

In the current version, the PyCaret library is used for training models (`estimators`).

Available estimators in PyCaret that can be used in the current implementation:
- ‘lr’ - Logistic Regression
- ‘knn’ - K Neighbors Classifier
- ‘nb’ - Naive Bayes
- ‘dt’ - Decision Tree Classifier
- ‘gpc’ - Gaussian Process Classifier
- ‘rf’ - Random Forest Classifier
- ‘qda’ - Quadratic Discriminant Analysis
- ‘ada’ - Ada Boost Classifier
- ‘gbc’ - Gradient Boosting Classifier
- ‘lda’ - Linear Discriminant Analysis
- ‘et’ - Extra Trees Classifier
- ‘xgboost’ - Extreme Gradient Boosting
- ‘lightgbm’ - Light Gradient Boosting Machine
- ‘catboost’ - CatBoost Classifier

Currently, the decision is made to use only the `Gradient Boosting Classifier`.

PyCaret automatically calculates parameters for each estimator, and these parameters are then written to the folder `{symbol}/long|short/models/{last_range}/PC/models`.

Setup parameters for the PyCaret library can be modified in the file `utils/modeling.py` in the `PC_TRAIN()` function.

### Current Settings:

```python
setup(
    data=train_df,
    target=label,
    fold_strategy="timeseries",
    use_gpu=False,
    normalize=True,
    transformation=True,
    transformation_method="quantile",
    low_variance_threshold=0,
    # ignore_low_variance=True,
    session_id=909,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.90,
    verbose=False,
    fix_imbalance=True,
    fold=2
)
```

### Function PC_TRAIN:
```python
def PC_TRAIN(train_df, ticker_path, current_date_range, cv_pct, label="target"):
    setup(
        data=train_df,
        target=label,
        fold_strategy="timeseries",
        use_gpu=False,
        normalize=True,
        transformation=True,
        transformation_method="quantile",
        low_variance_threshold=0,
        # ignore_low_variance=True,
        session_id=909,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.90,
        verbose=False,
        fix_imbalance=True,
        fold=2
    )
    trained_models = compare_models(n_select=16)
    models = {}
    for model in trained_models:
        model_name = str(model.__class__).split(".")[-1].replace("'>","")
        os.makedirs(f"{ticker_path}/models/{current_date_range}/PC/models/{model_name}", exist_ok=True)
        save_model(model, f"{ticker_path}/models/{current_date_range}/PC/models/{model_name}/model")
        models[model_name] = model
 
    return models
```

### Global Settings (in pycaret_pipeline.ipynb)
```python
MAX_INDICATOR_WINDOW = 1000
OPTIMIZE_INDICATORS = True
n_of_candles_to_add_to_target = 24
relearn_duration, trade_duration, required_n_of_periods = "36M", 3, 1
```
