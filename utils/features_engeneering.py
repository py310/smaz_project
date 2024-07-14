"""
Functions for feature generation will be placed here.
"""
import datetime
import logging
import os
import pickle

from utils import data_preparation as dp
# from utils import modeling as md
#from utils import portfolio_construction as pc
#from utils import stats as st
#from utils import trading as td

from .utils import *

# import billiard as multiprocessing

# logging.basicConfig(filename="loggi-3.log", format="%(asctime)s %(name)s: %(message)s")
# logger = logging.getLogger("ATASS_FE")
# logger.setLevel(logging.INFO)


pd.options.mode.chained_assignment = None  # default='warn'


def mark_target(df, ticker_path, adjClose=False, **target_params):
    """
    Signature:
    mark_target(df, adjClose=False, **target_params)

    Docstring:
    Generates a DataFrame with 'close' and returns it with a 'target' column.

    Parameters
    ----------
    df: DataFrame

    adjClose: bool, default False
        Indicates if there is a column named "Adjusted Close" in the data.

    **target_params - arguments related to the target marking function

    Returns
    -------
    DataFrame
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.
    """

    if adjClose:
        real_close = "adj_close"
    else:
        real_close = "close"

    if target_params["target_type"] == "digital_filter":

        FS3 = target_params["FS3"]
        CUTOFF_LOW = target_params["CUTOFF_LOW"]
        CUTOFF_HIGH = target_params["CUTOFF_HIGH"]
        ORDER = target_params["ORDER"]

        target = digital_filter(df[real_close], FS3, CUTOFF_LOW, CUTOFF_HIGH, ORDER)
        df = df[np.where(target == 0.5)[0][-1] + 1 :]
        target = target[np.where(target == 0.5)[0][-1] + 1 :]

    elif target_params["target_type"] == "kalman":
        NOISE = target_params["NOISE"]
        DEGREE = target_params["DEGREE"]
        target = kalman_filter_inflection_point(df[real_close], NOISE, DEGREE)

    elif target_params["target_type"] == "renko_target":
        RENKO_LEVEL = target_params["RENKO_LEVEL"]
        target = renko_target(df[real_close], RENKO_LEVEL)
        df = df[: len(target)]

    elif target_params["target_type"] == "horizon_target":
        X = target_params["X"]
        target = horizon_target(df[real_close], X)
        df = df[: len(target)]

    df["target"] = target
    df.to_csv(f"{ticker_path}/targeted.csv")

    return df

def mark_target_new(df, ticker_path, adjClose=False, **target_params):
    """
    Signature:
    mark_target(df, adjClose=False, **target_params)

    Docstring:
    Generates a DataFrame with 'close' and returns it with a 'target' column.

    Parameters
    ----------
    df: DataFrame
        The input dataframe containing 'close' data.

    adjClose: bool, default False
        Indicates whether there is a column named "Adjusted Close" in the data.

    **target_params - arguments related to the function that marks the target

    Returns
    -------
    DataFrame
        A comma-separated values (csv) file is returned as a two-dimensional
        data structure with labeled axes.
    """

    if adjClose:
        real_close = "adj_close"
    else:
        real_close = "close"

    if target_params["target_type"] == "digital_filter":

        FS3 = target_params["FS3"]
        CUTOFF_LOW = target_params["CUTOFF_LOW"]
        CUTOFF_HIGH = target_params["CUTOFF_HIGH"]
        ORDER = target_params["ORDER"]

        target = digital_filter(df[real_close], FS3, CUTOFF_LOW, CUTOFF_HIGH, ORDER)
        df = df[np.where(target == 0.5)[0][-1] + 1 :]
        target = target[np.where(target == 0.5)[0][-1] + 1 :]

    elif target_params["target_type"] == "kalman":
        NOISE = target_params["NOISE"]
        DEGREE = target_params["DEGREE"]
        target = kalman_filter_inflection_point(df[real_close], NOISE, DEGREE)

    elif target_params["target_type"] == "renko_target":
        RENKO_LEVEL = target_params["RENKO_LEVEL"]
        target = renko_target(df[real_close], RENKO_LEVEL)
        df = df[: len(target)]

    elif target_params["target_type"] == "horizon_target":
        X = target_params["X"]
        target = horizon_target(df[real_close], X)
        df = df[: len(target)]


    return pd.Series(data=target, index=df.index, name='target')


def optimize_indicator(n_arg, n, ticker_path):

    #     logger = dp.init_logger(ticker_path)
    #     logger.info(f"Start 'optimize_indicator' - {n_arg}-{n}")

    fs = "indicators"
    df = pd.read_csv(f"{ticker_path}/targeted.csv")
    o, h, l, c, v, target = (
        df["open"],
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        df["target"],
    )
    returns = pd.Series(np.concatenate(([np.nan], np.diff(c))) / c)
    vwap = (
        c + o
    ) / 2 

    builtins.o, builtins.h, builtins.l, builtins.c, builtins.v = o, h, l, c, v
    builtins.returns = returns
    builtins.vwap = vwap

    best = None
    bestq = 0
    hist = []

    if n_arg == 1:
        
        indicators_1arg = [
    #         obvm,
    #         cmf,
    #         fi,
            eom,
    #         atr,
            bhi,
            bli,
            dchi,
            dcli,
    #         adx_1arg,
            vip,
            vin,
    #         trix_1arg,
    #         cci_1arg,
    #         dpo_1arg,
            mfi,
    #         rsi_1arg,
            stoch_1arg,
    #         bbands_upper,
    #         bbands_lower,
    #         dema,
    #         ema,
    #         kama,
    #         ma,
            midpoint,
            midprice,
    #         tema,
    #         trima,
    #         wma,
    #         adxr,
            aroondown,
            aroonup,
            aroonosc,
    #         cmo,
    #         dx,
    #         minus_di,
    #         minus_dm,
            mom,
    #         plus_di,
    #         plus_dm,
            roc,
            new_indicator1,
    #         #         # new_indicator3,
            new_indicator4,
    #         new_indicator5,
            new_indicator6,
            new_indicator7,
            new_indicator8,
        ]

        indicator = indicators_1arg[n]

        # If the boundaries of the parameter are unknown, optimize it over a wide range.

        if str(n) + "1arg.txt" not in [
            f
            for f in os.listdir(f"{ticker_path}/optimization_borders/")
            if os.path.isfile(os.path.join(f"{ticker_path}/optimization_borders/", f))
        ]:
            start_time = datetime.datetime.now()

            for i in range(2, 300, 10):
                res = target.corr(indicator(i), "spearman")
                hist.append(np.abs(res))
                if np.abs(res) > bestq:
                    bestq = np.abs(res)
                    best = i
            try:
                mn = list(range(2, 300, 10))[
                    np.min(np.where(np.array(hist) > 0.7 * bestq)[0])
                ]
                mx = list(range(2, 300, 10))[
                    np.max(np.where(np.array(hist) > 0.7 * bestq)[0])
                ]
            except:
                mn = 2
                mx = 1000
            #                 logger.info(f"mn-mx for {n}-1 not counted")
            f = open(f"{ticker_path}/optimization_borders/" + str(n) + "1arg.txt", "w")
            f.write(str(mn) + " " + str(mx))
            f.close()

            f = open(f"{ticker_path}/trash/" + str(n) + "1arg.txt", "w")
            f.write(str(best) + " " + str(bestq))
            f.close()
            end_time = datetime.datetime.now()
            work_time = end_time - start_time

        # If this feature has been optimized before, we will optimize it in the vicinity of the previous maximum. The characteristic range is already available in the 'optimization borders' folder.
        else:
            f = open(f"{ticker_path}/optimization_borders/" + str(n) + "1arg.txt", "r")
            s = f.read()
            f.close()
            mn = int(s.split(" ")[0])
            mx = int(s.split(" ")[1])

            if mx - mn < 10:
                rng = range(max(2, mn - 10), mx + 10)
            elif mx - mn < 30:
                rng = range(mn, mx)
            else:
                rng = range(mn, mx, int(np.round((mx - mn) / 30)))
            for i in rng:
                res = target.corr(indicator(i), "spearman")
                if np.abs(res) > bestq:
                    bestq = np.abs(res)
                    best = i

            f = open(f"{ticker_path}/trash/" + str(n) + "1arg.txt", "w")
            f.write(str(best) + " " + str(bestq))
            f.close()

    elif n_arg == 2:
        indicators_2arg = [
#             macd_2arg,
#             mi,
#             tsi_2arg,
#             ao_2arg,
#             ppo,
            stochf_k,
#             stochf_d,
#             adosc,
#             mama,
#             fama,
#             apo,
#             new_indicator2,
        ]
        indicator = indicators_2arg[n]

        if str(n) + "2arg.txt" not in [
            f
            for f in os.listdir(f"{ticker_path}/optimization_borders/")
            if os.path.isfile(os.path.join(f"{ticker_path}/optimization_borders/", f))
        ]:

            start_time = datetime.datetime.now()

            ij = []
            if n == 11 and fs in ["indicators", "both"]:
                atrs = [average_true_range(h, l, c, i) for i in range(3, 200, 5)]
                for i in range(3, 200, 5):
                    for j in range(3, 200, 5):
                        res = target.corr(
                            np.log(c / talib.MA(c, j))
                            / atrs[list(range(3, 200, 5)).index(i)],
                            "spearman",
                        )
                        hist.append(np.abs(res))
                        ij.append([i, j])
                        if np.abs(res) > bestq:
                            bestq = np.abs(res)
                            best = [i, j]
            else:
                for i in range(3, 200, 5):
                    for j in range(3, 200, 5):
                        res = target.corr(indicator(i, j), "spearman")
                        hist.append(np.abs(res))
                        ij.append([i, j])
                        if np.abs(res) > bestq:
                            bestq = np.abs(res)
                            best = [i, j]
            if n == 11:
                inds = np.where(np.array(hist) > 0.9 * bestq)[0]
            else:
                inds = np.where(np.array(hist) > 0.7 * bestq)[0]
                
            f = open(f"{ticker_path}/optimization_borders/"+str(n)+'2arg.txt','w')
            f.write(str((np.array(ij)[inds]).tolist()))
            f.close()
            
            print(str(n))
            print(best)
            f = open(f"{ticker_path}/trash/" + str(n) + "2arg.txt", "w")
            f.write(str(best[0]) + " " + str(best[1]) + " " + str(bestq))
            f.close()

            end_time = (
                datetime.datetime.now()
            )
            work_time = end_time - start_time

        else:
            f = open(f"{ticker_path}/optimization_borders/" + str(n) + "2arg.txt", "r")
            s = f.read()
            rng = eval(s)
            f.close()

            if n == 11 and fs in ["indicators", "both"]:
                atrs = [average_true_range(h, l, c, i) for i in range(3, 200, 5)]
                for i in range(3, 200, 5):
                    for j in range(3, 200, 5):
                        res = target.corr(
                            np.log(c / talib.MA(c, j))
                            / atrs[list(range(3, 200, 5)).index(i)],
                            "spearman",
                        )
                        if np.abs(res) > bestq:
                            bestq = np.abs(res)
                            best = [i, j]

            else:
                for i, j in rng:
                    res = target.corr(indicator(i, j), "spearman")
                    if np.abs(res) > bestq:
                        bestq = np.abs(res)
                        best = [i, j]

            f = open(f"{ticker_path}/trash/" + str(n) + "2arg.txt", "w")
            f.write(str(best[0]) + " " + str(best[1]) + " " + str(bestq))
            f.close()

    work_time = work_time - datetime.timedelta(microseconds=work_time.microseconds)

    return f"{n_arg}_{n}", str(work_time)

def optimize_indicator_new(n_arg, n, ticker_path, df):

    fs = "indicators"
    o, h, l, c, v, target = (
        df["open"],
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        df["target"].dropna(),
    )
    returns = pd.Series(np.concatenate(([np.nan], np.diff(c))) / c)
    vwap = (
        c + o
    ) / 2

    builtins.o, builtins.h, builtins.l, builtins.c, builtins.v = o, h, l, c, v
    builtins.returns = returns
    builtins.vwap = vwap

    best = None
    bestq = 0
    hist = []

    if n_arg == 1:
        
        indicators_1arg = [
    #         obvm,
    #         cmf,
    #         fi,
            eom,
    #         atr,
            bhi,
            bli,
            dchi,
            dcli,
    #         adx_1arg,
            vip,
            vin,
    #         trix_1arg,
    #         cci_1arg,
    #         dpo_1arg,
            mfi,
    #         rsi_1arg,
            stoch_1arg,
    #         bbands_upper,
    #         bbands_lower,
    #         dema,
    #         ema,
    #         kama,
    #         ma,
            midpoint,
            midprice,
    #         tema,
    #         trima,
    #         wma,
    #         adxr,
            aroondown,
            aroonup,
            aroonosc,
    #         cmo,
    #         dx,
    #         minus_di,
    #         minus_dm,
            mom,
    #         plus_di,
    #         plus_dm,
            roc,
            new_indicator1,
    #         #         # new_indicator3,
            new_indicator4,
    #         new_indicator5,
            new_indicator6,
            new_indicator7,
            new_indicator8,
        ]

        indicator = indicators_1arg[n]

        # If we don't know the boundaries of the parameter where it should be optimized, we will optimize it over a wide range.

        for i in range(2, 300, 10):
            res = target.corr(indicator(i), "spearman")
            hist.append(np.abs(res))
            if np.abs(res) > bestq:
                bestq = np.abs(res)
                best = i


        # f = open(f"{ticker_path}/trash/" + str(n) + "1arg.txt", "w")
        # f.write(str(best) + " " + str(bestq))
        # f.close()
    
        return pd.Series(data=[best, bestq, n, n_arg], index=['Best_parameter_value', 'Best_corr_with_target_value', 'indicator_number', 'number_of_arguments'])
    


    elif n_arg == 2:
        indicators_2arg = [
#             macd_2arg,
#             mi,
#             tsi_2arg,
#             ao_2arg,
#             ppo,
            stochf_k,
#             stochf_d,
#             adosc,
#             mama,
#             fama,
#             apo,
#             new_indicator2,
        ]
        indicator = indicators_2arg[n]

        ij = []
        if n == 11 and fs in ["indicators", "both"]:
            atrs = [average_true_range(h, l, c, i) for i in range(3, 200, 5)]
            for i in range(3, 200, 5):
                for j in range(3, 200, 5):
                    res = target.corr(
                        np.log(c / talib.MA(c, j))
                        / atrs[list(range(3, 200, 5)).index(i)],
                        "spearman",
                    )
                    hist.append(np.abs(res))
                    ij.append([i, j])
                    if np.abs(res) > bestq:
                        bestq = np.abs(res)
                        best = [i, j]
        else:
            for i in range(3, 200, 5):
                for j in range(3, 200, 5):
                    res = target.corr(indicator(i, j), "spearman")
                    hist.append(np.abs(res))
                    ij.append([i, j])
                    if np.abs(res) > bestq:
                        bestq = np.abs(res)
                        best = [i, j]
        
        # f = open(f"{ticker_path}/trash/" + str(n) + "2arg.txt", "w")
        # f.write(str(best[0]) + " " + str(best[1]) + " " + str(bestq))
        # f.close()


        return pd.Series(data=best+[bestq, n, n_arg], index=['Best_parameter_value_1', 'Best_parameter_value_2', 'Best_corr_with_target_value', 'indicator_number', 'number_of_arguments'])

def multioptimize(indicators_1arg, indicators_2arg, ticker_path):  # add , logger

    len_1arg = len(indicators_1arg)
    len_2arg = len(indicators_2arg)
    PROCESSES = len_1arg + len_2arg

    params = [(1, x, ticker_path) for x in range(len(indicators_1arg))] + [
        (2, x, ticker_path) for x in range(len(indicators_2arg))
    ]

    with multiprocessing.Pool(PROCESSES) as pool:
        results = [pool.apply_async(optimize_indicator, p) for p in params]
        optimize_work_time = {}
        for r in results:
            n_indicatora, work_time = r.get()
            optimize_work_time[n_indicatora] = [work_time]
    # pd.DataFrame(optimize_work_time).T.to_csv(f"{ticker_path}/optimize_work_time.csv")

def multioptimize_new(indicators_1arg, indicators_2arg, ticker_path, df, optimize_results_folder='/optimized_indicators_params/'):  # add , logger

    len_1arg = len(indicators_1arg)
    len_2arg = len(indicators_2arg)
    PROCESSES = len_1arg + len_2arg

    params = [(1, x, ticker_path) for x in range(len(indicators_1arg))] + [
        (2, x, ticker_path) for x in range(len(indicators_2arg))
    ]

    best_params_info_1_args = []
    best_params_info_2_args = []

    with multiprocessing.Pool(PROCESSES) as pool:
        results = [pool.apply_async(optimize_indicator_new, list(p)+[df]) for p in params]
        for r in results:
            res = r.get()
            if res['number_of_arguments'] == 1:
                best_params_info_1_args.append(res)
            elif res['number_of_arguments'] == 2:
                best_params_info_2_args.append(res)

    target_dropped = df['target'].dropna()
    date_entry_out = target_dropped.index[0], target_dropped.index[-1]
    key = f"{date_entry_out[0].strftime('%Y-%m-%d')} - {date_entry_out[1].strftime('%Y-%m-%d')}"

    best_params_info_1_args = pd.concat(best_params_info_1_args, axis=1, keys=[key for _ in range(len(best_params_info_1_args))]).T
    best_params_info_2_args = pd.concat(best_params_info_2_args, axis=1, keys=[key for _ in range(len(best_params_info_2_args))]).T
    
    best_params_info_1_args.to_csv(ticker_path+optimize_results_folder+'1_arg.csv', sep=';')
    best_params_info_2_args.to_csv(ticker_path+optimize_results_folder+'2_arg.csv', sep=';')

#     if os.path.exists(ticker_path+optimize_results_folder+'1_arg.csv'):
#         result_df = pd.read_csv(ticker_path+optimize_results_folder+'1_arg.csv', sep=';', index_col=0)
#         result_df = pd.concat([result_df, best_params_info_1_args])
#         result_df.to_csv(ticker_path+optimize_results_folder+'1_arg.csv', sep=';')
#     else:
#         best_params_info_1_args.to_csv(ticker_path+optimize_results_folder+'1_arg.csv', sep=';')

#     if os.path.exists(ticker_path+optimize_results_folder+'2_arg.csv'):
#         result_df = pd.read_csv(ticker_path+optimize_results_folder+'2_arg.csv', sep=';', index_col=0)
#         result_df = pd.concat([result_df, best_params_info_2_args])
#         result_df.to_csv(ticker_path+optimize_results_folder+'2_arg.csv', sep=';')
#     else:
#         best_params_info_2_args.to_csv(ticker_path+optimize_results_folder+'2_arg.csv', sep=';')
    

def generate_indis(df, parameters_1arg, parameters_2arg):  # add logger
    """
    Signature:
    generate_indis(df, n_process=25)

    Docstring:
    Obtain a DataFrame with OHLCV and target, and return it with indicator columns.

    Parameters
    ----------
    df: DataFrame
        DataFrame with OHLCV and target.

    n_process: int, default 25
        Number of simultaneously launched scripts for optimizing indicators.

    Returns
    -------
    DataFrame
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.
    """

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    returns = pd.Series(np.concatenate(([np.nan], np.diff(c))) / c)
    vwap = (c + o) / 2

    builtins.o, builtins.h, builtins.l, builtins.c, builtins.v = o, h, l, c, v
    builtins.returns = returns
    builtins.vwap = vwap
    
    indicators_1arg = [
#         obvm,
#         cmf,
#         fi,
        eom,
#         atr,
        bhi,
        bli,
        dchi,
        dcli,
#         adx_1arg,
        vip,
        vin,
#         trix_1arg,
#         cci_1arg,
#         dpo_1arg,
        mfi,
#         rsi_1arg,
        stoch_1arg,
#         bbands_upper,
#         bbands_lower,
#         dema,
#         ema,
#         kama,
#         ma,
        midpoint,
        midprice,
#         tema,
#         trima,
#         wma,
#         adxr,
        aroondown,
        aroonup,
        aroonosc,
#         cmo,
#         dx,
#         minus_di,
#         minus_dm,
        mom,
#         plus_di,
#         plus_dm,
        roc,
        new_indicator1,
#         #         # new_indicator3,
        new_indicator4,
#         new_indicator5,
        new_indicator6,
        new_indicator7,
        new_indicator8,
    ]

    indicators_2arg = [
#         macd_2arg,
#         mi,
#         tsi_2arg,
#         ao_2arg,
#         ppo,
        stochf_k,
#         stochf_d,
#         adosc,
#         mama,
#         fama,
#         apo,
#         new_indicator2,
    ]

#     if optimization:
#         dp.write_log("Start indis optimization", ticker_path)
#         multioptimize(indicators_1arg, indicators_2arg, ticker_path)  # , logger   
#         dp.write_log("End indis optimization", ticker_path)


    def get_nonparametric_indicators():  # 19 indis
        global h, l, c, v
        return np.hstack(
            (
#                 np.array(acc_dist_index(h, l, c, v, True)).reshape(-1, 1),
#                 np.array(on_balance_volume(c, v, True)).reshape(-1, 1),
#                 np.array(volume_price_trend(c, v, True)).reshape(-1, 1),
#                 np.array(kst(c, 10, 15, 20, 30, 10, 10, 10, 15, True)).reshape(-1, 1),
#                 np.array(dcperiod(c)).reshape(-1, 1),
#                 np.array(dcphase(c)).reshape(-1, 1),
#                 np.array(phasor_ph(c)).reshape(-1, 1),
#                 np.array(phasor_quad(c)).reshape(-1, 1),
#                 np.array(sine(c)).reshape(-1, 1),
#                 np.array(leadsine(c)).reshape(-1, 1),
#                 np.array(trendmode(c)).reshape(-1, 1),
                np.array(avgprice(o, h, l, c)).reshape(-1, 1),
                np.array(medprice(h, l)).reshape(-1, 1),
                np.array(typprice(h, l, c)).reshape(-1, 1),
                np.array(wclprice(h, l, c)).reshape(-1, 1),
                np.array(tr(h, l, c)).reshape(-1, 1),
#                 np.array(ht_trendline(c)).reshape(-1, 1),
                np.array(bop(o, h, l, c)).reshape(-1, 1),
#                 np.array(ad(h, l, c, v)).reshape(-1, 1),
            )
        )

    X_train = get_nonparametric_indicators()

    # dp.write_log("End get_nonparametric_indicators", ticker_path)

#     # Читаем оптимизированные параметры
#     parameters_1arg = []
#     for i in range(len(indicators_1arg)):
#         f = open(f"{ticker_path}/trash/" + str(i) + "1arg.txt")
#         s = f.read()
#         parameters_1arg.append(int(s.split()[0]))
#         f.close()

#     parameters_2arg = []
#     for i in range(len(indicators_2arg)):
#         f = open(f"{ticker_path}/trash/" + str(i) + "2arg.txt")
#         s = f.read()
#         parameters_2arg.append([int(s.split()[0]), int(s.split()[1])])
#         f.close()
    
#     if write_params:
#         with open(f"{ticker_path}/parameters_1arg.p","wb") as f:
#             pickle.dump(parameters_1arg, f)

#         with open(f"{ticker_path}/parameters_2arg.p","wb") as f:
#             pickle.dump(parameters_2arg, f)
        

    # Enhancing training data with technical indicators optimized parameters
    for i in range(len(indicators_1arg)):
        try:
            X_train = np.hstack(
                (
                    X_train,
                    np.array(indicators_1arg[i](int(parameters_1arg[i]))).reshape(-1, 1),
                )
            )
        except Exception as e:
            print(f"Error 1arg - {i}\n")
            print(e)
            
            # dp.write_log(f"Error 1arg - {i}\n{e}", ticker_path)

    for i in range(len(indicators_2arg)):
        try:
            X_train = np.hstack(
                (
                    X_train,
                    np.array(
                        indicators_2arg[i](int(parameters_2arg[i][0]), int(parameters_2arg[i][1]))
                    ).reshape(-1, 1),
                )
            )
        except Exception as e: 
            print(f"Error 2arg - {i}\n")
            print(e)
            # dp.write_log(f"Error 2arg - {i}\n{e}", ticker_path)

    #     logger.info("Stack 1-2 args indicators")

    maxnanind = 0
    for i in range(len(X_train[0])):
        try:
            if np.where(np.isnan(X_train[:, i]))[0][-1] > maxnanind:
                maxnanind = np.where(np.isnan(X_train[:, i]))[0][-1]
        
        except IndexError:
            pass

    indi_df = pd.DataFrame(X_train[maxnanind:], index=df.index[maxnanind:])


#     indi_df = pd.DataFrame(X_train, index=df.index)
#     df = df.join(indi_df)

#     df = df[maxnanind:]
    
#     if write_csv:
#         df.to_csv(f"{ticker_path}/indi_target.csv")

#     end_generate_indis_time = datetime.datetime.now()

    return indi_df

def generate_indis_new(df, ticker_path, df_OOS, optimization=True, write_csv=True, optimize_results_folder='/optimized_indicators_params/', MAX_INDICATOR_WINDOW=1000):  # add logger
    """
    Signature:
    generate_indis(df)

    Docstring:
    Obtain a DataFrame with OHLCV and target, and return it with indicator columns.

    Parameters
    ----------
    df: DataFrame
        DataFrame with OHLCV and target.

    Returns
    -------
    DataFrame
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.
    """

    target_dropped = df['target'].dropna()
    date_entry_out = target_dropped.index[0], target_dropped.index[-1]
    current_dates_entry_out = f"{date_entry_out[0].strftime('%Y-%m-%d')} - {date_entry_out[1].strftime('%Y-%m-%d')}"

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    
    returns = pd.Series(np.concatenate(([np.nan], np.diff(c))) / c)
    vwap = (c + o) / 2

    builtins.o, builtins.h, builtins.l, builtins.c, builtins.v = o, h, l, c, v
    builtins.returns = returns
    builtins.vwap = vwap
    
    indicators_1arg = [
#         obvm,
#         cmf,
#         fi,
        eom,
#         atr,
        bhi,
        bli,
        dchi,
        dcli,
#         adx_1arg,
        vip,
        vin,
#         trix_1arg,
#         cci_1arg,
#         dpo_1arg,
        mfi,
#         rsi_1arg,
        stoch_1arg,
#         bbands_upper,
#         bbands_lower,
#         dema,
#         ema,
#         kama,
#         ma,
        midpoint,
        midprice,
#         tema,
#         trima,
#         wma,
#         adxr,
        aroondown,
        aroonup,
        aroonosc,
#         cmo,
#         dx,
#         minus_di,
#         minus_dm,
        mom,
#         plus_di,
#         plus_dm,
        roc,
        new_indicator1,
#         #         # new_indicator3,
        new_indicator4,
#         new_indicator5,
        new_indicator6,
        new_indicator7,
        new_indicator8,
    ]

    indicators_2arg = [
#         macd_2arg,
#         mi,
#         tsi_2arg,
#         ao_2arg,
#         ppo,
        stochf_k,
#         stochf_d,
#         adosc,
#         mama,
#         fama,
#         apo,
#         new_indicator2,
    ]

    if optimization:
        # dp.write_log("Start indis optimization", ticker_path)
        multioptimize_new(indicators_1arg, indicators_2arg, ticker_path, df, optimize_results_folder=optimize_results_folder)  # , logger   
        # dp.write_log("End indis optimization", ticker_path)


    def get_nonparametric_indicators():  # 19 indis
        global o, h, l, c, v
        return np.hstack(
            (
#                 np.array(acc_dist_index(h, l, c, v, True)).reshape(-1, 1),
#                 np.array(on_balance_volume(c, v, True)).reshape(-1, 1),
#                 np.array(volume_price_trend(c, v, True)).reshape(-1, 1),
#                 np.array(kst(c, 10, 15, 20, 30, 10, 10, 10, 15, True)).reshape(-1, 1),
#                 np.array(dcperiod(c)).reshape(-1, 1),
#                 np.array(dcphase(c)).reshape(-1, 1),
#                 np.array(phasor_ph(c)).reshape(-1, 1),
#                 np.array(phasor_quad(c)).reshape(-1, 1),
#                 np.array(sine(c)).reshape(-1, 1),
#                 np.array(leadsine(c)).reshape(-1, 1),
#                 np.array(trendmode(c)).reshape(-1, 1),
                np.array(avgprice(o, h, l, c)).reshape(-1, 1),
                np.array(medprice(h, l)).reshape(-1, 1),
                np.array(typprice(h, l, c)).reshape(-1, 1),
                np.array(wclprice(h, l, c)).reshape(-1, 1),
                np.array(tr(h, l, c)).reshape(-1, 1),
#                 np.array(ht_trendline(c)).reshape(-1, 1),
                np.array(bop(o, h, l, c)).reshape(-1, 1),
#                 np.array(ad(h, l, c, v)).reshape(-1, 1),
            )
        )

    X_train = get_nonparametric_indicators()

    # dp.write_log("End get_nonparametric_indicators", ticker_path)

    # Читаем оптимизированные параметры
    # parameters_1arg = []
    # for i in range(len(indicators_1arg)):
    #     f = open(f"{ticker_path}/trash/" + str(i) + "1arg.txt")
    #     s = f.read()
    #     parameters_1arg.append(int(s.split()[0]))
    #     f.close()

    # parameters_2arg = []
    # for i in range(len(indicators_2arg)):
    #     f = open(f"{ticker_path}/trash/" + str(i) + "2arg.txt")
    #     s = f.read()
    #     parameters_2arg.append([int(s.split()[0]), int(s.split()[1])])
    #     f.close()
    
    parameters_1arg = (
        pd.read_csv(ticker_path+optimize_results_folder+'1_arg.csv', sep=';', index_col=0)
        .sort_values(by='indicator_number')
        .loc[[current_dates_entry_out], 'Best_parameter_value']
        .to_numpy()
        .astype(int)
    )

    parameters_2arg = (
        pd.read_csv(ticker_path+optimize_results_folder+'2_arg.csv', sep=';', index_col=0)
        .sort_values(by='indicator_number')
        .loc[[current_dates_entry_out], ['Best_parameter_value_1', 'Best_parameter_value_2']]
        .to_numpy()
        .astype(int)
    )

    # print(parameters_2arg)
    # print(parameters_2arg[0])
        
    # Augment training data with technical indicators using optimized parameters.
    for i in range(len(indicators_1arg)):
        try:
            X_train = np.hstack(
                (
                    X_train,
                    np.array(indicators_1arg[i](parameters_1arg[i])).reshape(-1, 1),
                )
            )
        except Exception as e:
            print(f"Error 1arg - {i}\n")
            print(e)
            
            # dp.write_log(f"Error 1arg - {i}\n{e}", ticker_path)

    for i in range(len(indicators_2arg)):
        try:
            X_train = np.hstack(
                (
                    X_train,
                    np.array(
                        indicators_2arg[i](parameters_2arg[i][0], parameters_2arg[i][1])
                    ).reshape(-1, 1),
                )
            )
        except Exception as e: 
            print(f"Error 2arg - {i}\n")
            print(e)
            # dp.write_log(f"Error 2arg - {i}\n{e}", ticker_path)

    #     logger.info("Stack 1-2 args indicators")

#     maxnanind = 0
#     for i in range(len(X_train[0])):
#         try:
#             if np.where(np.isnan(X_train[:, i]))[0][-1] > maxnanind:
#                 maxnanind = np.where(np.isnan(X_train[:, i]))[0][-1]
# #                                                             print(i,maxnanind)
#         except IndexError:
#             pass

#     indi_df = pd.DataFrame(X_train[maxnanind:], index=df.index[maxnanind:])

    o, h, l, c, v = df_OOS["open"], df_OOS["high"], df_OOS["low"], df_OOS["close"], df_OOS["volume"]
    returns = pd.Series(np.concatenate(([np.nan], np.diff(c))) / c)
    vwap = (c + o) / 2

    builtins.o, builtins.h, builtins.l, builtins.c, builtins.v = o, h, l, c, v
    builtins.returns = returns
    builtins.vwap = vwap
    
    X_test = get_nonparametric_indicators()

    for i in range(len(indicators_1arg)):
        try:
            X_test = np.hstack(
                (
                    X_test,
                    np.array(indicators_1arg[i](parameters_1arg[i])).reshape(-1, 1),
                )
            )
        except Exception as e:
            print(f"Error 1arg - {i}\n")
            print(e)
            
            # dp.write_log(f"Error 1arg - {i}\n{e}", ticker_path)

    for i in range(len(indicators_2arg)):
        try:
            X_test = np.hstack(
                (
                    X_test,
                    np.array(
                        indicators_2arg[i](parameters_2arg[i][0], parameters_2arg[i][1])
                    ).reshape(-1, 1),
                )
            )
        except Exception as e: 
            print(f"Error 2arg - {i}\n")
            print(e)
    
    
    indi_df_IS = pd.DataFrame(X_train[MAX_INDICATOR_WINDOW:], index=df.index[MAX_INDICATOR_WINDOW:]).ffill()
    indi_df_OOS = pd.DataFrame(X_test[MAX_INDICATOR_WINDOW:], index=df_OOS.index[MAX_INDICATOR_WINDOW:]).ffill()
    indi_df = pd.concat([indi_df_IS, indi_df_OOS])
    return indi_df
    
def INDICATORS_BLOCK(
    symbol,
    FORCE,
    SAVE,
    ticker_path,
    indicators_file,
    relearn_dates_grid,
    relearn_time_labels,
    trade_dates_grid,
    targeted_df,
    ohlcv,
    MAX_INDICATOR_WINDOW
):
    print(f">>> START: INDICATORS BLOCK {symbol}")

    os.makedirs(f"{ticker_path}/optimized_indicators_params/", exist_ok=True)

    parts_indicators_df = []

    for i in range(relearn_dates_grid.shape[0]):

        indicators_optimized_params_folder = f"optimized_indicators_params/{relearn_time_labels.iloc[i].replace(':', '_')}/"
        # We need to create a folder just in case.
        os.makedirs(
            rf"{ticker_path}/{indicators_optimized_params_folder}", exist_ok=True
        )

        file_exists = check_existance_of_file(
            path_to_file=f"{ticker_path}/{indicators_optimized_params_folder}",
            file_name=indicators_file,
        )

        # ? If the target is already calculated and doesn't need to be recalculated.
        if file_exists and not FORCE["INDICATORS"]:
            print(
                f"The data exists in {FORCE['INDICATORS']}, so the data will simply be read from the file {ticker_path}/{indicators_optimized_params_folder}{indicators_file}"
            )
            indicators_part_df = pd.read_csv(
                ticker_path + indicators_optimized_params_folder + indicators_file,
                sep=";",
                index_col=0,
                parse_dates=[0],
                dayfirst=True,
            )
            parts_indicators_df.append(indicators_part_df)

        else:
            # ? It needs to be recalculated.
            print(f">>> CALCULATING INDICATORS {symbol}...")

            # ? We determine the dates on which it is necessary to focus.
            relearn_entry, relearn_out = relearn_dates_grid.iloc[i]
            trade_entry, trade_out = trade_dates_grid.iloc[i]
            indicators_entry_IS, indicators_out_IS = (
                ohlcv.index.to_series()
                .shift(MAX_INDICATOR_WINDOW)
                .loc[relearn_entry],  #! <--- comma
                relearn_out,
            )

            indicators_entry_OOS, indicators_out_OOS = (
                ohlcv.index.to_series()
                .shift(MAX_INDICATOR_WINDOW)
                .loc[trade_entry],  #! <--- comma
                trade_out,
            )

            if (
                str(relearn_entry)
                + " "
                + str(relearn_out)
                != relearn_time_labels.iloc[i]
            ):
                input(
                    "The date labels do not match; most likely there's an error. Submit at your own risk."
                )

            print(f"INDICATORS {relearn_entry, trade_out} CALCULATING...")
            # Calculating indicators
            # Added 1000 extra candles, but trimmed the unnecessary ones at the end.

            ohlcv_for_indicators_IS = ohlcv.loc[indicators_entry_IS:indicators_out_IS]
            target_for_indicators_IS = targeted_df[relearn_time_labels.iloc[i]].dropna()

            df_for_indicators_IS = pd.concat(
                [ohlcv_for_indicators_IS, target_for_indicators_IS], axis=1
            )
            df_for_indicators_IS.columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "target",
            ]

            indicators_part_df = fe.generate_indis_new(
                df_for_indicators_IS,
                ticker_path,
                df_OOS=ohlcv.loc[indicators_entry_OOS:indicators_out_OOS],
                optimize_results_folder=indicators_optimized_params_folder,
                optimization=OPTIMIZE_INDICATORS,
            ).loc[
                relearn_entry:
            ]

            parts_indicators_df.append(indicators_part_df)

            if SAVE["INDICATORS"]:
                indicators_part_df.to_csv(
                    f"{ticker_path}/{indicators_optimized_params_folder}{indicators_file}",
                    sep=";",
                )
            print(f"INDICATORS {relearn_entry, trade_out} ... DONE")

    indicators_df = pd.concat(parts_indicators_df, axis=1, keys=relearn_time_labels)

    return indicators_df
