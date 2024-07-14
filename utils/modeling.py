'''
Here will be functions related to models (Linear regression, kneighboors,...) or their libraries (AG, PyCaret, ...) or metrics.
'''
import os
# from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import numpy as np
# from autogluon.tabular import TabularDataset, TabularPredictor, models
from pycaret.classification import *

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


def create_Label_Score(pred):
    """
    Input: AutoGluon predictor.predict_proba output columns
    Output: PyCaret-type predict output with "Label" and "Score"(probability of Label)
    """
    pred.loc[pred[0] > pred[1], "Label"] = 0
    pred.loc[pred[0] > pred[1], "Score"] = pred[0]
    pred.loc[pred[0] < pred[1], "Label"] = 1
    pred.loc[pred[0] < pred[1], "Score"] = pred[1]
    
    # pred.max(axis=1).ffill()    # <--- max_value_in_row
    # pred.idxmax(axis=1).ffill() # <--- Label
    
    pred = pred.ffill()
    return pred

def predict_ag(train_df, test_df, ticker_path, cv_pct=0.2, label="target"):
    """

    """
    
    path=f"{ticker_path}/agModels"

    cv_size = int(len(train_df) * cv_pct)
    train_data = TabularDataset(train_df[:-cv_size])
    tuning_data = TabularDataset(train_df[-cv_size:])

    predictor = TabularPredictor(
        label=label, path=path, verbosity=0  # eval_metric="precision_macro",
    ).fit(
        train_data,
        tuning_data=tuning_data,
        eval_metric="roc_auc",
        # presets="best_quality",
        # auto_stack=False,
        ag_args_fit={"num_gpus": 1},
    )

    test_data = TabularDataset(test_df)
    y_test = test_data[label]
    predictor = TabularPredictor.load(path)


#     model_names = predictor.leaderboard(test_data)["model"]
    model_names = pd.Series(predictor.get_model_names())

    model_test_results = []

    for model_name in model_names:  # prediction on test set
        predicted_proba = predictor.predict_proba(test_data, model=model_name)
        predicted_test = create_Label_Score(predicted_proba)
        predicted_test.drop(columns=[0, 1], inplace=True)
        model_test_results.append(predicted_test)
        
    model_names = model_names.apply(lambda x: f"{x}~AG")  ###  
    model_test_results = pd.concat(model_test_results, axis=1, keys=model_names)

    return model_test_results

def predict_pycaret(train_df, test_df, p, ticker_path):

    initialization = setup(
        data=train_df,
        target=p.target,
        test_data=test_df,
        fold_strategy="timeseries",
        fold = 2,
        use_gpu=False,
        normalize=True,
        transformation=True,
        ignore_low_variance=True,
        session_id=909,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.90,
        silent=True,
        verbose=False,
        fix_imbalance=True,
        fix_imbalance_method=smote
    )
    top15 = compare_models(n_select=16)
    
    pycaret_prdict_results = []
    pycaret_model_names = []
    for model in top15:  # prediction on test set
#         model_name = model.__str__().split("(")[0]
        model_name = str(model.__class__).split(".")[-1].replace("'>","")
        predicted_test = predict_model(model, data=test_df, encoded_labels=True)
        save_model(model, f"{ticker_path}/pycaret/{model_name}") ##+++++++++++++
        try:
            pycaret_predicted_test = predicted_test[["Label", "Score"]]
            pycaret_prdict_results.append(pycaret_predicted_test)
            pycaret_model_names.append(f"{model_name}~PC")

        except:
            print(f"Predict error at model - {model_name}")

    pycaret_prdict_results = pd.concat(
        pycaret_prdict_results, axis=1, keys=pycaret_model_names
    )
    return pycaret_prdict_results
    
def train_and_get_prediction(dataset, reb_idx, p, ticker_path):

    ag_models_results = []
    pycaret_models_results = []
    
    for i in tqdm(range(p.TRAIN_PERIOD, reb_idx.shape[0], p.TEST_PERIOD)):

        train_start_date, train_end_date = (
            reb_idx[i - p.TRAIN_PERIOD],
            reb_idx.index[i - 1],
        )  # method first is required in get_idx
        test_start_date, test_end_date = reb_idx[i], reb_idx.index[i + p.TEST_PERIOD - 1]


        train_df = dataset.loc[train_start_date : str(train_end_date.date())]
        test_df = dataset.loc[test_start_date : str(test_end_date.date())]
        
        ag_predicted = predict_ag(train_df, test_df, ticker_path,cv_pct=0.2 )
        ag_models_results.append(ag_predicted)

        pycaret_prdicted = predict_pycaret(train_df, test_df, p, ticker_path)
        pycaret_models_results.append(pycaret_prdicted) 
        
    ag_models_results = pd.concat(ag_models_results, axis=0)
    pycaret_models_results = pd.concat(pycaret_models_results, axis=0)
    
    all_models_results = ag_models_results.join(pycaret_models_results)
    all_models_results.to_csv(f"{ticker_path}/predicted_df.csv")
    
    return all_models_results

def AG_TRAIN(train_df, ticker_path, current_date_range, cv_pct=0.2, label="target"):
    path=f"{ticker_path}/models/{current_date_range}/AG/"
    cv_size = int(len(train_df) * cv_pct)
    train_data = TabularDataset(train_df[:-cv_size])
    tuning_data = TabularDataset(train_df[-cv_size:])

    predictor = TabularPredictor(
        label=label, path=path, verbosity=0,  eval_metric="balanced_accuracy"
    ).fit(
        train_data,
        tuning_data=tuning_data,
        # presets="best_quality",
        # auto_stack=False,
        ag_args_fit={"num_gpus": 1},
    )

    return predictor

def PC_TRAIN(train_df, ticker_path, current_date_range, cv_pct, label="target"):
    
    # Retrieve the number of logical kernels (CPU cores)
    n_logical_kernels = os.cpu_count()   
    print(f"{n_logical_kernels}")
    
    # Check if the number of logical kernels is greater than 60
    if n_logical_kernels > 60:
        n_jobs = 60
    else: n_jobs = -1
    print(f"{n_jobs=}")

    setup(
        data=train_df,
        target=label,
        fold_strategy="timeseries",
        use_gpu=False,
        normalize=True,
        transformation=True,
        transformation_method="quantile",
        low_variance_threshold = 0,
        # ignore_low_variance=True,
        session_id=909,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.90,
        verbose=False,
        fix_imbalance=True,
        fold=2,
        n_jobs=n_jobs
    )
    trained_models = compare_models(n_select=16)
    models = {}
    for model in trained_models:
        model_name = str(model.__class__).split(".")[-1].replace("'>","")
        os.makedirs(f"{ticker_path}/models/{current_date_range}/PC/models/{model_name}", exist_ok=True)
        save_model(model, f"{ticker_path}/models/{current_date_range}/PC/models/{model_name}/model")
        models[model_name] = model

    return models

def TRAIN(dataset, ticker_path, current_date_range, current_libraries, possible_function_to_call, cv_pct=0.2, label='target'):
    trained_models = {
        library: possible_function_to_call[library](dataset, ticker_path, current_date_range, cv_pct, label)
        for library in current_libraries
    }
    
    return trained_models

def AG_READ_MODELS(ticker_path, current_date_range):
    path_to_read_predictor = ticker_path+f'/models/{current_date_range}/AG'
    predictor = TabularPredictor.load(path_to_read_predictor)

    return predictor

def PC_READ_MODELS(ticker_path, current_date_range):
    path_to_folders_with_models = ticker_path+f'/models/{current_date_range}/PC/models/'
    model_names = os.listdir(path_to_folders_with_models)

    models = {}

    for model_name in model_names:
        model_file = path_to_folders_with_models+f'{model_name}/model'
        model = load_model(model_file)
        models[model_name] = model

    return models

def READ_MODELS(ticker_path, current_date_range, current_libraries, possible_function_to_call):
    trained_models = {
        library: possible_function_to_call[library](ticker_path, current_date_range)
        for library in current_libraries
    }

    return trained_models

def AG_TEST(predictor, dataset):
    model_names = pd.Series(predictor.get_model_names())

    model_test_results = []

    for model_name in model_names:  # prediction on test set
        predicted_proba = predictor.predict_proba(dataset, model=model_name)
        predicted_test = create_Label_Score(predicted_proba)
        predicted_test.drop(columns=[0, 1], inplace=True)
        model_test_results.append(predicted_test)
        
    model_names = model_names.apply(lambda x: f"{x}~AG")  ###  
    model_test_results = pd.concat(model_test_results, axis=1, keys=model_names)

    return model_test_results

def PC_TEST(models: dict, dataset):
    pycaret_predict_results, pycaret_model_names = [], []
    for model_name, model in models.items():  

        predicted_test = predict_model(model, data=dataset, encoded_labels=True)
        predicted_test.index = dataset.index

        try:
            pycaret_predicted_test = predicted_test[["prediction_label", "prediction_score"]]
            pycaret_predict_results.append(pycaret_predicted_test)
            pycaret_model_names.append(f"{model_name}~PC")

        except:
            print(f"Predict error at model - {model_name}")

    pycaret_predict_results = pd.concat(
        pycaret_predict_results, axis=1, keys=pycaret_model_names
    )
    return pycaret_predict_results

def TEST(trained_models, possible_function_to_call_TEST, dataset):
    dfs = []
    for library, models in trained_models.items():
        dfs.append(
            possible_function_to_call_TEST[library](models, dataset)
        )
    
    predicted_df = pd.concat(dfs, axis=1)

    return predicted_df

def get_score_of_label_one(model_df, shifted=True):
    if shifted:
        model_df = pd.concat(
            [model_df, 
             pd.DataFrame(np.where(model_df['prediction_label']==0, 1-model_df['prediction_score'], model_df['prediction_score']), index=model_df.index).shift()
            ], axis=1).bfill()
        model_df.columns = ['prediction_label', 'prediction_score', 'shifted']
    else:
        model_df = pd.concat(
            [model_df, 
             pd.DataFrame(np.where(model_df['prediction_label']==0, 1-model_df['prediction_score'], model_df['prediction_score']), index=model_df.index)
            ], axis=1).dropna()
        model_df.columns = ['prediction_label', 'prediction_score', 'score_of_one']       
    
    return model_df

def load_models(ticker_path, date_range):
    """
    Loading ML models

    Args:
    ticker_path (str): path to the ticker folder
    date_range (str): the range of interest (half a year)

    Returns:
    predictor (autogluon.tabular.TabularPredictor): AutoGluon object with trained models
    pycaret_models (list): dictionary with trained PyCaret models
        
        
    """
    predictor = TabularPredictor.load(f"{ticker_path}/models/{date_range}/AG/")

    pycaret_models_names = [
        model_name
        for model_name in os.listdir(f"{ticker_path}/models/{date_range}/PC/models")
    ]

    pycaret_models = {}

    for model_name in pycaret_models_names:
        print(model_name)
        model = load_model(
            f"{ticker_path}/models/{date_range}/PC/models/{model_name}/model"
        )
        pycaret_models[model_name] = model

    return predictor, pycaret_models


def read_indis_params(ticker_path, date_range):
    """
    Args:
        ticker_path (str): path to the ticker folder
        date_range (str): the range of interest (half-year)

    Returns:
        parameters_1arg (list): a list of parameters for indicators with 1 parameter
        parameters_2arg (list): a list of parameters for indicators with 2 parameters
    """
    parameters_1arg = (
        pd.read_csv(
            ticker_path + "/optimized_indicators_params/1_arg.csv", sep=";", index_col=0
        )
        .sort_values(by="indicator_number")
        .loc[[date_range], "Best_parameter_value"]
        .to_list()
    )

    parameters_2arg = (
        pd.read_csv(
            ticker_path + "/optimized_indicators_params/2_arg.csv", sep=";", index_col=0
        )
        .sort_values(by="indicator_number")
        .loc[[date_range], ["Best_parameter_value_1", "Best_parameter_value_2"]]
        .to_numpy()
        .astype(int)
        .tolist()
    )

    return parameters_1arg, parameters_2arg


def calculate_history_length(parameters_1arg, parameters_2arg):
    """
    Args:
        parameters_1arg (list): list of parameters for indicators with 1 parameter
        parameters_2arg (list): list of parameters for indicators with 2 parameters

    Returns:
        history_length (int): maximum parameter value for these indicators + 14 
            (number of candles requested for indicator calculation)
    """

    all_indis_params = parameters_1arg.copy()
    for indi in parameters_2arg:
        all_indis_params = all_indis_params + [
            sum(indi)
        ]  # только для stochf_k берем сумму параметров

    max_window = max(all_indis_params)
    history_length = int(max_window + 14)

    return history_length


def make_preparation(ticker_path):
    """
    Unpacks models, indicator parameters, and calculates the number of bars for processing.

    Args:
        ticker_path (str): path to the ticker folder

    Returns:
        predictor (autogluon.tabular.TabularPredictor): AutoGluon object with trained models
        pycaret_models (list): dictionary with trained PyCaret models
        parameters_1arg (list): list of parameters for indicators with 1 parameter
        parameters_2arg (list): list of parameters for indicators with 2 parameters
        history_length (int): maximum parameter value for these indicators + 14
            (number of candles requested for indicator calculation)
    """
    current_date_range = max(os.listdir(f"{ticker_path}/models/"))
    predictor, pycaret_all_models = load_models(ticker_path, current_date_range)
    parameters_1arg, parameters_2arg = read_indis_params(
        ticker_path, current_date_range
    )
    history_length = calculate_history_length(parameters_1arg, parameters_2arg)

    return (
        predictor,
        pycaret_all_models,
        parameters_1arg,
        parameters_2arg,
        history_length,
    )


def get_ohlcv_dataset(history_length, request_tail):
    """
    Function retrieves a dataset with `history_length` number of candles up to the current moment.

    Args:
        history_length (int): maximum parameter value for these indicators + 14
            (number of candles requested for indicator calculation)
        request_tail (str): parameters for server request, including symbol and timeframe
            (example - 'symbol=EURCHF&timeframe=M5')

    Returns:
        ohlcv_dataset (DataFrame): Bars prepared for pipeline work in the required quantity
    """
    set_buffer(history_length, request_tail)
    ohlcv_dataset = pd.DataFrame()
    print(ohlcv_dataset)

    while ohlcv_dataset.shape[0] < history_length:
        time.sleep(5)
        data = get_data(request_tail)
        if data != []:
            if "High" in data[0]:  # Если в данных есть бары
                print("Get")
                new_candles = candles_to_df(data)
                ohlcv_dataset = pd.concat([ohlcv_dataset, new_candles])

    return ohlcv_dataset


def get_proba_json(test_df, ticker_path, predictor, pycaret_models, autogluon_models):
    """
    Args:
        test_df (DataFrame): dataset to predict
        ticker_path (str): path to the ticker folder
        predictor (autogluon.tabular.TabularPredictor): AutoGluon object with trained models
        pycaret_models (list): dictionary of trained PyCaret models
        autogluon_models (list): AutoGluon models to be used in trading

    Returns:
        proba_json (str): a JSON string with prediction probabilities for server submission
    """
    all_models_results = pd.DataFrame()

    if autogluon_models != []:
        print("Start AutoGluon")
        autogluon_models = pd.Series(autogluon_models)

        ag_predict_results = []
        for model in autogluon_models:
            print(model)

            predicted_proba = predictor.predict_proba(test_df, model=model)
            predicted_candle = create_Label_Score(predicted_proba)
            predicted_candle.drop(columns=[0, 1], inplace=True)
            ag_predict_results.append(predicted_candle)

        autogluon_models = autogluon_models.apply(lambda x: f"{x}~AG")  ###
        ag_predict_results = pd.concat(
            ag_predict_results, axis=1, keys=autogluon_models
        )
        ag_predict_results = pd.concat([ag_predict_results], axis=0)
        all_models_results = ag_predict_results

    if pycaret_models != {}:
        print("Start PyCaret")

        pycaret_predict_results = []
        pycaret_model_names = []

        for model_name, model in pycaret_models.items():
            print(model_name)
            predicted_test = predict_model(model, data=test_df, encoded_labels=True)
            try:
                pycaret_predicted_test = predicted_test[["Label", "Score"]]
                pycaret_predict_results.append(pycaret_predicted_test)
                pycaret_model_names.append(f"{model_name}~PC")

            except:
                print(f"Predict error at model - {model_name}")

        pycaret_predict_results = pd.concat(
            pycaret_predict_results, axis=1, keys=pycaret_model_names
        )
        pycaret_predict_results = pd.concat([pycaret_predict_results], axis=0)

        if all_models_results.shape[0] == 0:
            all_models_results = pycaret_predict_results
        else:
            all_models_results = all_models_results.join(pycaret_predict_results)


    models = all_models_results.swaplevel(axis=1)["Score"].columns

    print("models", models)

    together_df = pd.concat(
        [
            get_score_of_label_one(all_models_results[model], shifted=False)
            for model in models
        ],
        axis=1,
        keys=models,
    )

    return together_df
