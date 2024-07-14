from ta import *
from scipy.signal import butter, filtfilt
import numpy as np
import talib
import tsfresh
import matplotlib.pyplot as plt
import os
import shutil
from filterpy.kalman import KalmanFilter
import talib
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
# import xgboost as xgb
import warnings
from ta import *
import os.path
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.volume import *

# from utils import *
# from cutters import *
import datetime
import matplotlib.pyplot as plt
import shutil
import platform
import collections
import operator

def norm(x,N):
    return (x-x.rolling(N,min_periods=1).mean())/x.rolling(N,min_periods=1).std()

def first_der(x):
    return x.diff()

def second_der(x):
    return x.diff(2)

def third_der(x):
    return x.diff(3)

def first_relat_der(x):
    return first_der(x)/x.shift(1)

def second_relat_der(x):
    return second_der(x)/x.shift(2)

def third_relat_der(x):
    return third_der(x)/x.shift(3)

def get_autocorr(x,N):
    return pd.Series(x).rolling(N).apply(lambda a: pd.Series(a).autocorr(N//2))

# def optimize_1arg(indicator,target,n):
#     best = None
#     bestq = 0
#     for i in range(10,1000,20):
#         res = target.corr(indicator(i))
#         if np.abs(res) > bestq:
#             bestq = np.abs(res)
#             best = i
#     f = open('./trash/'+str(n)+'1arg.txt','w')
#     f.write(str(best)+" "+str(bestq))
#     f.close()
    
    
# def optimize_2arg(indicator,target,n):
#     best = None
#     bestq = 0
#     for i in range(2,150,3):
#         for j in range(2,150,3):
#             res = target.corr(indicator(i,j))
#             if np.abs(res) > bestq:
#                 bestq = np.abs(res)
#                 best = [i,j]
#     f = open('./trash/'+str(n)+'2arg.txt','w')
#     f.write(str(best[0])+" "+str(best[1])+" "+str(bestq))
#     f.close()

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def clear_trash(path):
    shutil.rmtree(f'{path}/trash')
    os.mkdir(f'{path}/trash')
    shutil.rmtree(f'{path}/optimization_borders')
    os.mkdir(f'{path}/optimization_borders')
    
def detrend(data, degree):
    detrended = [0] * degree
    for i in range(degree, len(data)):
        chunk = data[i - degree: i]
        detrended.append(data[i] - sum(chunk) / len(chunk))
    return detrended

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def digital_filter(data,fs_3,cutoff_low,cutoff_high,order): 
    g = np.array(detrend(data, int(fs_3 / 100.)))
    signal = g

    conditioned_signal = butter_lowpass_filter(signal, cutoff_low, fs_3, order)
    c = np.roll(conditioned_signal, 0)

    conditioned_signal_high = butter_lowpass_filter(signal, cutoff_high, fs_3, order)
    C = np.roll(conditioned_signal_high, 0)


    cl = np.array([1. if c[i] >= C[i] else 0. for i in range(len(c))])
    cl[:int(fs_3/100)] = 0.5

    return cl

def kalman_filter_inflection_point(data, noise, degree, show_velocity=False):
    Q = 0.001
    fk = KalmanFilter(dim_x=2, dim_z=1)
    fk.x = np.array([0., 1.])      # state (x and dx)
    fk.F = np.array([[1., 1.],[0., 1.]])    # state transition matrix
    fk.H = np.array([[1., 0.]])    # Measurement function
    fk.P = 10000.                  # covariance matrix
    fk.R = noise                   # state uncertainty
    fk.Q = Q                       # process uncertainty

    # create noisy data
    zs = np.array(detrend(data, degree))

    # filter data with Kalman filter, than run smoother on it
    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, C, _ = fk.rts_smoother(mu, cov)
    
    d = [0] * (len(M[:, 0]))
    C = M[:, 0]
    for i in range(1, len(C)-1):
        if (C[i-1] < C[i] > C[i+1]):
            d[i] = 1
        elif (C[i-1] > C[i] < C[i+1]):
            d[i] = 0
        else:
            d[i]=d[i-1]
    
    return d

def renko_target(close, RENKO_LEVEL):
    target = np.empty_like(close)
    level = close[0]
    pointer = 0
    for ind, c in enumerate(close):
        if c - level >= RENKO_LEVEL:
            target[pointer:ind+1] = 1
            pointer = ind+1
            level = c
        elif level - c >= RENKO_LEVEL:
            target[pointer:ind+1] = 0
            pointer = ind+1
            level = c
    return pd.Series(target[np.where((target == 0) | (target == 1))])  

def horizon_target(close, X):
    target = []
    for i in range(len(close)-X):
        if close[i+X] > close[i]:
            target.append(1)
        else:
            target.append(0)
    return pd.Series(target)

def maxmin_target(c, w):
    loc_max = []
    loc_min = []
    c = np.array(c)

    for i in range(w//2,len(c)-w//2):
        if c[i] == np.max(c[i-w//2:i+w//2]):
            loc_max.append(i)
            continue
        if c[i] == np.min(c[i-w//2:i+w//2]):
            loc_min.append(i)

    loc_max = np.array(loc_max)
    loc_min = np.array(loc_min)
    
    i = 0
    j = 0
    all_points = []
    while True:
        if loc_max[i] < loc_min[j]:
            all_points.append([loc_max[i], 1])
            i += 1
        else:
            all_points.append([loc_min[j], 0])
            j += 1
        if i == len(loc_max):
            for k in loc_min[j:]:
                all_points.append([k, 0])
            break
        if j == len(loc_min):
            for k in loc_max[i:]:
                all_points.append([k, 1])
            break

    i = 1
    while i < len(all_points):

        if all_points[i][1] == all_points[i-1][1] == 1:
            if c[all_points[i][0]] > c[all_points[i-1][0]]:
                del all_points[i-1]
            else:
                del all_points[i]
        elif all_points[i][1] == all_points[i-1][1] == 0:
            if c[all_points[i][0]] < c[all_points[i-1][0]]:
                del all_points[i-1]
            else:
                del all_points[i]
        else:
            i += 1

    all_points = np.array(all_points)

    prev_pos = 0
    target = []
    for i in all_points:
        if i[1] == 1:
            target += [1]*(i[0]-prev_pos)
            prev_pos = i[0]
        elif i[1] == 0:
            target += [0]*(i[0]-prev_pos)
            prev_pos = i[0]
    target += [1-i[1]]*(len(c)-prev_pos)
    return np.array(target)

def ma_target(c, w):
    arr = [1 if i > 0 else 0 for i in np.diff(pd.Series(c).rolling(w,min_periods=1,center=True).mean())]
    arr = [arr[0]] + arr
    return arr

def correct_renko_target(c, L):
    level = L
    target = []
    for i in range(len(c)-1):
        for j in range(i+1, len(c)):
            if (c[j]-c[i])/c[i] >= level:
                target.append(1)
                break
            elif (c[j]-c[i])/c[i] <= -level:
                target.append(0)
                break
    target += [target[-1]]*(len(c)-len(target))
    return target

def updown_level_target(c, LUP, LDOWN):
    target = []
    for i in range(len(c)-1):
        for j in range(i+1, len(c)):
            if (c[j]-c[i])/c[i] >= LUP:
                target.append(1)
                break
            elif (c[j]-c[i])/c[i] <= -LDOWN:
                target.append(0)
                break
    target += [target[-1]]*(len(c)-len(target))
    return target

def obvm(x):
    global c,v
    return on_balance_volume(c,v,False).rolling(x, min_periods=0).mean()
    
def cmf(x):
    global h, l, c, v
    return chaikin_money_flow(h,l,c,v,x,False)

def fi(x):
    global c, v
    return force_index(c,v,x,False)

def eom(x):
    global h, l, v
    return ease_of_movement(h,l,v,x,False)

def atr(x):
    global h, l, c
    return average_true_range(h,l,c,x,False)

def bhi(x):
    global c
    return bollinger_hband_indicator(c,x,2,False)

def bli(x):
    global c
    return bollinger_lband_indicator(c,x,2,False)

def shift_elements(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def dchi(x):
    global h, l, c
    return pd.Series(
        np.where(
            # donchian_channel_hband(h, l, c, x, False) <= shift_elements(h, -1, 0),
            donchian_channel_hband(h, l, c, x, False).shift(1) <= h,
            1,
            0,
        ),
        index=c.index,
)

def dcli(x):
    global h, l, c
    return pd.Series(
        np.where(
            # donchian_channel_lband(h, l, c, x, False) >= shift_elements(l, -1, 0),
            donchian_channel_lband(h, l, c, x, False).shift(1) >= l,
            1,
            0,
        ),
        index=c.index,
    )
def adx_1arg(x):
    global h, l, c
    return adx(h,l,c,x,False)

def vip(x):
    global h, l, c
    return vortex_indicator_pos(h,l,c,x,False)

def vin(x):
    global h, l, c
    return vortex_indicator_neg(h,l,c,x,False)

def trix_1arg(x):
    global c
    return trix(c,x,False)

def cci_1arg(x):
    global h, l, c
    return cci(h,l,c,x,0.015,False)

def dpo_1arg(x):
    global c
    return dpo(c,x,False)

def mfi(x):
    global h, l, c, v
    return money_flow_index(h,l,c,v,x,False)

def rsi_1arg(x):
    global c
    return rsi(c,x,False)

def stoch_1arg(x):
    global h, l, c
    return stoch(h,l,c,x,False)

# def wr_1arg(x):
#     global h, l, c
#     return wr(h,l,c,x,True)

def macd_2arg(x,y):
    global c
    return macd(c,x,y,False)

def mi(x,y):
    global h, l
    return mass_index(h,l,x,y,False)

def tsi_2arg(x,y):
    global c
    return tsi(c,x,y,False)

def ao_2arg(x,y):
    global h, l
    return ao(h,l,x,y,False)

def bbands_upper(x):
    global c
    ind = (talib.BBANDS(c,x)[0]-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

# def bbands_middle(x):
#     global c
#     ind = (talib.BBANDS(c,x)[1]-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
#     return ind

def bbands_lower(x):
    global c
    ind = (talib.BBANDS(c,x)[2]-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def dema(x):
    global c
    ind = (talib.DEMA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def ema(x):
    global c
    ind = (talib.EMA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def kama(x):
    global c
    ind = (talib.KAMA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def ma(x):
    global c
    ind = (talib.MA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def midpoint(x):
    global c
    ind = (talib.MIDPOINT(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def midprice(x):
    global h, l
    ind = (talib.MIDPRICE(h,l,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def tema(x):
    global c
    ind = (talib.TEMA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def trima(x):
    global c
    ind = (talib.TRIMA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def wma(x):
    global c
    ind = (talib.WMA(c,x)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def adxr(x):
    global h, l, c
    ind = talib.ADXR(h,l,c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def aroondown(x):
    global h, l
    ind = talib.AROON(h,l,x)[0]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def aroonup(x):
    global h, l
    ind = talib.AROON(h,l,x)[1]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def aroonosc(x):
    global h, l
    ind = talib.AROONOSC(h, l, x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def cmo(x):
    global c
    ind = talib.CMO(c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def dx(x):
    global h, l, c
    ind = talib.DX(h,l,c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def minus_di(x):
    global h, l, c
    ind = talib.MINUS_DI(h,l,c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def minus_dm(x):
    global h, l
    ind = talib.MINUS_DM(h,l,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def mom(x):
    global c
    ind = talib.MOM(c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def plus_di(x):
    global h, l, c
    ind = talib.PLUS_DI(h,l,c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def plus_dm(x):
    global h, l
    ind = talib.PLUS_DM(h,l,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def roc(x):
    global c
    ind = talib.ROC(c,x)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def ppo(x,y):
    global c
    ind = talib.PPO(c,x,y)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def stochf_k(x,y):
    global h, l, c
    ind = talib.STOCHF(h,l,c,x,y)[0]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def stochf_d(x,y):
    global h, l, c
    ind = talib.STOCHF(h,l,c,x,y)[1]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def adosc(x,y):
    global h, l, c, v
    ind = talib.ADOSC(h,l,c,v,x,y)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def mama(x,y):
    global c
    x = np.linspace(0.01,0.99,200)[x]
    y = np.linspace(0.01,0.99,200)[y]
    ind = (talib.MAMA(c,fastlimit=x,slowlimit=y)[0]-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def fama(x,y):
    global c
    x = np.linspace(0.01,0.99,200)[x]
    y = np.linspace(0.01,0.99,200)[y]
    ind = (talib.MAMA(c,fastlimit=x,slowlimit=y)[1]-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def apo(x,y):
    global c
    ind = talib.APO(c,x,y)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def dcperiod(c):
    ind = talib.HT_DCPERIOD(c)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def dcphase(c):
    ind = talib.HT_DCPHASE(c)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def phasor_ph(c):
    ind = talib.HT_PHASOR(c)[0]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def phasor_quad(c):
    ind = talib.HT_PHASOR(c)[1]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def sine(c):
    ind = talib.HT_SINE(c)[0]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def leadsine(c):
    ind = talib.HT_SINE(c)[1]
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def trendmode(c):
    return talib.HT_TRENDMODE(c)

def avgprice(o,h,l,c):
    return (talib.AVGPRICE(o,h,l,c)-c)/c

def medprice(h,l):
    return (talib.MEDPRICE(h,l)-c)/c

def typprice(h,l,c):
    return (talib.TYPPRICE(h,l,c)-c)/c

def wclprice(h,l,c):
    return (talib.WCLPRICE(h,l,c)-c)/c

def tr(h,l,c):
    ind = talib.TRANGE(h,l,c)
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def ht_trendline(c):
    ind = (talib.HT_TRENDLINE(c)-c)/c
#     nan_ind = np.where(np.isnan(ind))[0][-1]
#     ind[:nan_ind+1] = ind[nan_ind+1]
    return ind

def bop(o,h,l,c):
    return talib.BOP(o,h,l,c)

def ad(h,l,c,v):
    return talib.AD(h,l,c,v)

def new_indicator1(x):
    global c,o
    return (c-o.shift(x))/o.shift(x)

def new_indicator2(x,y):
    global h,l,c
    return np.log(c/talib.MA(c,x))/average_true_range(h,l,c,y)

from sklearn.linear_model import LinearRegression
def my_linreg_new(x,n, reg):
    ans = [np.nan]*n
    shit = np.array(range(n)).reshape(-1,1)
    shit_2 = np.array([n]).reshape(-1,1)

    for i in range(len(x)-n):
        reg.fit(shit, x[i:i+n])
        ans.append(reg.predict(shit_2)[0])

    return ans

def new_indicator3(x):
    global c
    reg = LinearRegression()
    return np.log(c) - my_linreg_new(np.log(c.values), x, reg)

def linreg(x,n):
    from sklearn.linear_model import LinearRegression

    ans = [np.nan]*n
    reg = LinearRegression()
    for i in range(len(x)-n):
        reg.fit(np.array(range(n)).reshape(-1,1),x[i:i+n])
        ans.append(reg.predict(np.array([n]).reshape(-1,1))[0])
    return ans

# def new_indicator3(x):
#     global c
#     return np.log(c)-pd.Series(linreg(np.log(c),x))

def new_indicator4(x):
    global c
    return np.log(c/c.shift(x))

def new_indicator5(x):
    global c
    return np.log(c/talib.MA(c,x))

def my_energy_ratio_by_chunks(np_closes, window, num_segments=2, segment_focus=1):
    energy_result = []
    for i in range(1,len(np_closes)+1):
        time_series = np_closes[max(0,i-window):i]
        full_energy = np.sum(time_series ** 2)
        energy_result.append(np.sum(np.array_split(time_series, num_segments)[segment_focus] ** 2.0) / full_energy)
    
    return energy_result

def new_indicator6(x):
    global c
    
    return pd.Series(my_energy_ratio_by_chunks(c.values, x, num_segments=2, segment_focus=1), index = c.index)

# def new_indicator6(x):
#     global c
#     return pd.Series([tsfresh.feature_extraction.feature_calculators.energy_ratio_by_chunks(c[max(0,i-x):i],[{"num_segments":2,"segment_focus":1}])[0][1] for i in range(1,len(c)+1)])

def my_index_mass_quantile(closes, x):
    result = []
    for i in range(1,len(closes)+1):
        tmp = closes[max(0,i-x):i]

        total_sum = np.sum(tmp)
        mass_centralized = np.cumsum(tmp) / total_sum
        result.append((np.argmax(mass_centralized >= 0.5) + 1) / tmp.shape[0])
        
    return result

def new_indicator7(x):
    global c
    return pd.Series(my_index_mass_quantile(c.values, x), index = c.index)

# def new_indicator7(x):
#     global c
#     return pd.Series([tsfresh.feature_extraction.feature_calculators.index_mass_quantile(c[max(0,i-x):i],[{"q":0.5}])[0][1] for i in range(1,len(c)+1)])

def new_indicator8(x):
    global c
    np_c = c.values
    return pd.Series([tsfresh.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(np_c[max(0,i-x):i],2) for i in range(1,len(c)+1)], index = c.index)

# def new_indicator8(x):
#     global c
#     return pd.Series([tsfresh.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(c[max(0,i-x):i],2) for i in range(1,len(c)+1)])

def correlation(x,y,d,method='pearson'):
    x = rolling_window(x,d)
    y = rolling_window(y,d)
    return np.concatenate(([np.nan]*(d-1),np.array([pd.Series(i).corr(pd.Series(j),method=method) for i,j in zip(x,y)])))

def delay(x,d):
    return np.array(pd.Series(c).shift(d))

def delta(x,d):
    return x-delay(x,d)

def covariance(x,y,d):
    x = rolling_window(x,d)
    y = rolling_window(y,d)
    return np.concatenate(([np.nan]*(d-1),np.array([np.cov(np.vstack((i,j)))[0,1] for i,j in zip(x,y)])))

def scale(x,a=1):
    try:
        lastnanind = np.where(np.isnan(x))[0][-1]
    except IndexError:
        lastnanind = -1
    x = x[lastnanind+1:]
    return np.concatenate(([np.nan]*(lastnanind+1),x*a/np.sum(np.abs(x))))

def signedpower(x,a):
    return np.power(x,a)

def decay_linear(x,d):
    weights = np.arange(1,d+1,1)
    weights = scale(weights,1)
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.dot(x,weights.reshape(-1,1)).ravel()))

def ts_min(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.min(x,axis=1)))

def ts_max(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.max(x,axis=1)))

def ts_argmax(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.argmax(x,axis=1)))

def ts_argmin(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.argmin(x,axis=1)))

def ts_sum(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.sum(x,axis=1)))

def product(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.product(x,axis=1)))
                          
def stddev(x,d):
    x = rolling_window(x,d)
    return np.concatenate(([np.nan]*(d-1),np.std(x,axis=1)))      

def adv(v,d):
    v = rolling_window(v,d)
    return np.concatenate(([np.nan]*(d-1),np.mean(v,axis=1)))   

def alpha1(w1=20,w2=5):
    global c, returns
    ans = stddev(returns,w1)
    inds = np.where(returns >= 0)[0]
    ans[inds] = c[inds]
    return pd.Series(ts_argmax(signedpower(ans,2),w2))

def alpha2(w1=2,w2=6):
    global o, c, v
    return pd.Series(correlation(delta(np.log(v),w1),(c-o)/o,w2))

def alpha3(w=10):
    global o, v
    return pd.Series(correlation(o,v,w))

def alpha5(w=10):
    global o, c, vwap
    return pd.Series((o - ts_sum(vwap, w) / w) * np.abs(c - vwap))

def alpha7(w1=20,w2=7):
    global c, v
    inds = np.where(adv(v,w1) >= v)[0]
    ans = -1*np.abs(delta(c,w2))*np.sign(delta(c,w2))
    ans[inds] = -1
    return pd.Series(ans)

def alpha8(w1=5,w2=10):
    global o, returns
    return pd.Series(ts_sum(o,w1)*ts_sum(returns,w1)-delay(ts_sum(o,w1)*ts_sum(returns,w1),w2))

def alpha9(w=5):
    global c
    d = delta(c,1)
    inds = np.where(ts_max(d,w) >= 0)[0]
    ans = d
    ans[inds] = -1*d[inds]
    inds = np.where(0 >= ts_min(d,w))[0]
    ans2 = d
    ans2[inds] = ans[inds]
    return pd.Series(ans2)

def alpha10(w=4):
    global c
    return pd.Series(alpha9(w))

def alpha11(w=3):
    global c, v, vwap
    return pd.Series(ts_max(vwap - c, w) + ts_min(vwap - c, w) * delta(v, w))

def alpha12(v):
    return pd.Series(np.sign(delta(v,1))*-1*delta(c,1))

def alpha13(w=5):
    global c, v
    return pd.Series(covariance(c,v,w))

def alpha14(w1=3,w2=10):
    global o, v, returns
    return pd.Series(delta(returns,w1)*correlation(o,v,w2))

def alpha15(w1=3,w2=3):
    global h, v
    return pd.Series(ts_sum(correlation(h,v,w1),w2))

def alpha16(w=5):
    global h, v
    return pd.Series(covariance(h,v,w))

def alpha17(w=20):
    global c, v
    return pd.Series(c*delta(delta(c,1),1)*v/adv(v,w))

def alpha18(w1=5,w2=10):
    global o, c
    return pd.Series(stddev(np.abs(c-o),w1)+c-o+correlation(c,o,w2))

def alpha19(w1=7,w2=250):
    global c, returns
    return pd.Series(np.sign(c-delay(c,w1)+delta(c,w1))*(2+ts_sum(returns,w2)))

def alpha20(o,h,l,c):
    return pd.Series((o-delay(h,1))*(o-delay(c,1))*(o-delay(l,1)))

def alpha21(w2=8,w3=20):
    global c, v
    inds = np.where((1 < v/adv(v,w3)) | (v/adv(v,w3) == 1))[0]
    ans = -1*np.ones(len(c))
    ans[inds] = 1
    inds = np.where(ts_sum(c,2)/2 < ts_sum(c,w2)/w2-stddev(c,w2))[0]
    ans[inds] = 1
    inds = np.where(ts_sum(c,w2)/w2+stddev(c,w2) < ts_sum(c,2)/2)[0]
    ans[inds] = -1
    return pd.Series(ans)

def alpha22(w1=5,w2=20):
    global h, v
    return pd.Series(delta(correlation(h,v,w1),w1)*stddev(c,w2))

def alpha23(w=20):
    global h
    ans = np.zeros(len(h))
    inds = np.where(ts_sum(h,w)/w < h)[0]
    ans[inds] = -1*delta(h,2)[inds]
    return pd.Series(ans)

def alpha24(w=100):
    global c
    ans = delta(c,3)
    inds = np.where((delta(ts_sum(c,w)/w,w)/delay(c,w)<0.05) | (delta(ts_sum(c,w)/w,w)/delay(c,w)==0.05))[0]
    ans[inds] = (c-ts_min(c,w))[inds]
    return pd.Series(ans)

def alpha25(w=20):
    global h, c, v, returns, vwap
    return returns * adv(v,w) * vwap * (h - c)

def alpha26(w1=5,w2=3):
    global h, v
    return pd.Series(ts_max(correlation(v,h,w1),w2))

def alpha27(w1=6,w2=2):
    global v, vwap
    ans = np.ones(len(v))
    inds = np.where(0.5 < ts_sum(correlation(v, vwap, w1), w2) / w2)[0]
    ans[inds] = -1
    return pd.Series(ans)

def alpha28(w1=20,w2=5):
    global h, l, c, v
    return pd.Series(scale(correlation(adv(v,w1),l,w2)+(h+l)/2-c))

def alpha29(c,returns):
    return pd.Series((ts_min(product(scale(np.log(ts_sum(ts_min(-1*delta(c-1,5),2),1))),1),5)+delay(-1*returns,6)-c)/c)

def alpha30(c,v):
    return pd.Series((1-np.sign(c-delay(c,1))-np.sign(delay(c,1)-delay(c,2))-np.sign(delay(c,2)-delay(c,3)))*ts_sum(v,5)/ts_sum(v,20))

def alpha31(w=10):
    global l, c, v
    return pd.Series(decay_linear(-delta(c,w),w)-delta(c,3)+np.sign(scale(correlation(adv(v,20),l,12))))

def alpha32(w1=7,w2=5):
    global c, vwap
    return pd.Series(scale(ts_sum(c, w1) / w1 - c) + 20 * scale(correlation(vwap, delay(c, w2), 230)))

def alpha33(o,c):
    return pd.Series(o/c-1)

def alpha34(w1=2,w2=5):
    global c, returns
    return pd.Series(2-stddev(returns,w1)/stddev(returns,w2)-delta(c,1))

def alpha35(h,l,c,v,returns):
    return pd.Series(v*(1-c+h-l)*(1-returns))

def alpha36(o,c,v,vwap):
    return pd.Series(2.21*correlation(c-o,delay(v,1),15)+0.7*(o - c)+0.73*delay(-1*returns,6) 
+ np.abs(correlation(vwap, adv(v,20), 6)) + 0.6 * (ts_sum(c, 200) / 200 - o) * (c - o))

def alpha37(w=200):
    global o, c
    return pd.Series(correlation(delay(o-c,1),c,w)+o-c)

def alpha39(c,v,returns):
    return pd.Series(-delta(c,7)*(1-decay_linear(v/adv(v,20),9))*(1+ts_sum(returns,250)))

def alpha40(w=10):
    global h, v
    return pd.Series(-stddev(h,w)*correlation(h,v,w))

def alpha41(h,l,vwap):
    return pd.Series((h * l)**0.5 - vwap)

def alpha42(c,vwap):
    return pd.Series((vwap - c) / (vwap + c))

def alpha43(w1=7,w2=20):
    global c, v
    return pd.Series(-v*delta(c,w1)/adv(v,w2))

def alpha44(w=5):
    global h, v
    return pd.Series(correlation(h,v,w))

def alpha45(w1=5,w2=20):
    global c, v
    return pd.Series(ts_sum(delay(c,w1),w2)*correlation(c,v,2)*correlation(ts_sum(c,w1),ts_sum(c,w2),2)/w2)

def alpha46(w1=10,w2=20):
    global c
    ans = -c + delay(c, 1)
    inds = np.where((delay(c, w2) - delay(c, w1)) / w1 - (delay(c, w1) - c) / w1 < 0)[0]
    ans[inds] = 1
    inds = np.where(0.25 < (delay(c, w2) - delay(c, w1)) / w1 - (delay(c, w1) - c) / w1)[0]
    ans[inds] = -1
    return pd.Series(ans)

def alpha47(w1=5,w2=20):
    global h, c, v, vwap
    return pd.Series((v / (adv(v,w2)*c)) * (h * (h - c) / (ts_sum(h, w1) / w1)) - vwap + delay(vwap, w1))

def alpha49(w1=10,w2=20):
    global c
    ans = -c + delay(c, 1)
    inds = np.where((delay(c, w2) - delay(c, w1)) / w1 - (delay(c, w1) - c) / w1 < -0.1)[0]
    ans[inds] = 1
    return pd.Series(ans)

def alpha50(w=5):
    global v, vwap
    return pd.Series(ts_max(correlation(v, vwap, w), w))

def alpha51(w1=10,w2=20):
    global c
    ans = -c + delay(c, 1)
    inds = np.where((delay(c, w2) - delay(c, w1)) / w1 - (delay(c, w1) - c) / w1 < -0.05)[0]
    ans[inds] = 1
    return pd.Series(ans)

def alpha52(w=5):
    global l, v, returns
    return pd.Series((-ts_min(l, w)+delay(ts_min(l, w),w))*(ts_sum(returns, 240) - ts_sum(returns, 20))*v/220)

def alpha53(w=9):
    global h, l, c
    return pd.Series(-delta((2*c - l - h) / (c - l), w))

def alpha54(p=5):
    global o, h, l, c
    return pd.Series((l - c) * (o**p)/ ((l - h) * (c**p)))

def alpha55(w=12):
    global h, l, c, v
    return pd.Series(-correlation((c - ts_min(l, w)) / (ts_max(h, w) - ts_min(l, w)), v, w//2))

def alpha57(w1=2,w2=30):
    global c, vwap
    res = pd.Series((vwap-c) / decay_linear(ts_argmax(c, w2), w1)).tolist()[w1+w2-2:]
    return pd.Series([np.nan]*(w1+w2-2)+list(np.nan_to_num(res)))

def alpha60(w=10):
    global h, l, c, v
    return pd.Series(-2*scale((2*c - l - h)* v / (h - l)) + scale(ts_argmax(c, w)))

def alpha61(v,vwap):
    return pd.Series(vwap - ts_min(vwap, 16) < correlation(vwap, adv(v,180), 18))

def alpha62(h,l,v,vwap):
    return pd.Series(correlation(vwap, ts_sum(adv(v,20), 22), 10) < (2*o < (h + l) / 2 + h))

def alpha64(o,l,v,vwap):
    return pd.Series(correlation(ts_sum(o * 0.178404 + l * (1 - 0.178404), 13), ts_sum(adv(v,120), 13), 17) < delta((h + l)* 0.178404 / 2  + vwap * (1 - 0.178404), 4))

def alpha65(o,v,vwap):
    return pd.Series(correlation(o * 0.00817205 + vwap * (1 - 0.00817205), ts_sum(adv(v,60), 9), 6) < o - ts_min(o, 14))
  
def alpha66(o,h,l,vwap):
    return pd.Series(decay_linear(delta(vwap, 4), 7) + decay_linear((l* 0.96633 + l * (1 - 0.96633) - vwap) / (o - (h + l) / 2),12))
    
def alpha68(w=15):
    global h, l, c, v
    return pd.Series(correlation(h,adv(v,w), 9) < delta(c * 0.518371 + l * (1 - 0.518371), 1))

def alpha72(h,l,v,vwap):
    return pd.Series(decay_linear(correlation((h + l) / 2, adv(v,40), 9), 10) / decay_linear(correlation(vwap, v, 7),3))

def alpha74(h,v,c,vwap):
    return pd.Series(correlation(c, ts_sum(adv(v,30), 37), 15) < correlation(h * 0.0261661 + vwap * (1 - 0.0261661), v, 11))

def alpha75(l,v,vwap):
    return pd.Series(correlation(vwap, v, 4) < correlation(l, adv(v,50), 12))

def alpha83(w1=5,w2=2):
    global h, l, c, v, vwap
    return pd.Series((delay((h - l) / (ts_sum(c, w1) / w1), w2) * v) / (((h - l) / (ts_sum(c, w1) / w1)) / (vwap - c)))

# def alpha85(h,l,c,v,w=30):
#     return np.power(correlation(h * 0.876703 + c * (1 - 0.876703), adv(v,w),10),correlation((h + l) / 2, v, 7))

def alpha86(c,v,vwap):
    return pd.Series(correlation(c, ts_sum(adv(v,20), 15), 6) < c - vwap)

# def alpha88(o,h,l,c,v):
#     return ts_min(decay_linear(o + l - h - c, 8), decay_linear(correlation(c, adv(v,60), 8), 7))

# def alpha92(o,h,l,c,v):
#     return ts_min(decay_linear((((h + l) / 2) + c < l + o), 15), decay_linear(correlation(l, adv(v,30), 8), 7))

def alpha95(o,h,l,v):
    return pd.Series(o - ts_min(o, 12) < correlation(ts_sum((h + l)/ 2, 19), ts_sum(adv(v,40), 19), 13)**5)

def alpha98(o,v,vwap):
    return pd.Series(decay_linear(correlation(vwap, ts_sum(adv(v,5), 26), 5), 7) - decay_linear(correlation(o, adv(v,15), 21), 8))

def alpha99(h,l,v):
    return pd.Series(correlation(ts_sum(((h + l) / 2), 20), ts_sum(adv(v,60), 20), 9) < correlation(l, v, 6))

def alpha101(o,h,l,c):
    return pd.Series((c - o) / (h - l + .001))

def l1_ind1(x):
    global c
    return norm(c,x)

def l1_ind2(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).mean()

def l1_ind3(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).std().fillna(0)

def l1_ind4(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).median()

def l1_ind5(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).max()

def l1_ind6(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).min()

def l1_ind7(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).quantile(.1)

def l1_ind8(x,y):
    global c
    return norm(c,x).rolling(y,min_periods=1).quantile(.9)

def l1_ind9(c):
    return first_der(c)

def l1_ind10(x):
    global c
    return first_der(c).rolling(x,min_periods=1).mean()

def l1_ind11(x):
    global c
    return first_der(c).rolling(x,min_periods=1).std().fillna(0)

def l1_ind12(x):
    global c
    return first_der(c).rolling(x,min_periods=1).median()

def l1_ind13(x):
    global c
    return first_der(c).rolling(x,min_periods=1).max()

def l1_ind14(x):
    global c
    return first_der(c).rolling(x,min_periods=1).min()

def l1_ind15(x):
    global c
    return first_der(c).rolling(x,min_periods=1).quantile(.1)

def l1_ind16(x):
    global c
    return first_der(c).rolling(x,min_periods=1).quantile(.9)

def l1_ind17(x):
    global c
    return first_der(norm(c,x))

def l1_ind18(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).mean()

def l1_ind19(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind20(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).median()

def l1_ind21(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).max()

def l1_ind22(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).min()

def l1_ind23(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).quantile(.1)

def l1_ind24(x,y):
    global c
    return first_der(norm(c,x)).rolling(y,min_periods=1).quantile(.9)

def l1_ind25(c):
    return second_der(c)

def l1_ind26(x):
    global c
    return second_der(c).rolling(x,min_periods=1).mean()

def l1_ind27(x):
    global c
    return second_der(c).rolling(x,min_periods=1).std().fillna(0)

def l1_ind28(x):
    global c
    return second_der(c).rolling(x,min_periods=1).median()

def l1_ind29(x):
    global c
    return second_der(c).rolling(x,min_periods=1).max()

def l1_ind30(x):
    global c
    return second_der(c).rolling(x,min_periods=1).min()

def l1_ind31(x):
    global c
    return second_der(c).rolling(x,min_periods=1).quantile(.1)

def l1_ind32(x):
    global c
    return second_der(c).rolling(x,min_periods=1).quantile(.9)

def l1_ind33(x):
    global c
    return second_der(norm(c,x))

def l1_ind34(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).mean()

def l1_ind35(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind36(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).median()

def l1_ind37(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).max()

def l1_ind38(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).min()

def l1_ind39(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).quantile(.1)

def l1_ind40(x,y):
    global c
    return second_der(norm(c,x)).rolling(y,min_periods=1).quantile(.9)

def l1_ind41(c):
    return third_der(c)

def l1_ind42(x):
    global c
    return third_der(c).rolling(x,min_periods=1).mean()

def l1_ind43(x):
    global c
    return third_der(c).rolling(x,min_periods=1).std().fillna(0)

def l1_ind44(x):
    global c
    return third_der(c).rolling(x,min_periods=1).median()

def l1_ind45(x):
    global c
    return third_der(c).rolling(x,min_periods=1).max()

def l1_ind46(x):
    global c
    return third_der(c).rolling(x,min_periods=1).min()

def l1_ind47(x):
    global c
    return third_der(c).rolling(x,min_periods=1).quantile(.1)

def l1_ind48(x):
    global c
    return third_der(c).rolling(x,min_periods=1).quantile(.9)

def l1_ind49(x):
    global c
    return third_der(norm(c,x))

def l1_ind50(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).mean()

def l1_ind51(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind52(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).median()

def l1_ind53(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).max()

def l1_ind54(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).min()

def l1_ind55(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).quantile(.1)

def l1_ind56(x,y):
    global c
    return third_der(norm(c,x)).rolling(y,min_periods=1).quantile(.9)

def l1_ind57(c):
    return first_relat_der(c)

def l1_ind58(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).mean()

def l1_ind59(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).std().fillna(0)

def l1_ind60(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).median()

def l1_ind61(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).max()

def l1_ind62(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).min()

def l1_ind63(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).quantile(.1)

def l1_ind64(x):
    global c
    return first_relat_der(c).rolling(x,min_periods=1).quantile(.9)

def l1_ind65(x):
    global c
    return first_relat_der(norm(c,x))

def l1_ind66(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).mean()

def l1_ind67(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind68(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).median()

def l1_ind69(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).max()

def l1_ind70(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).min()

def l1_ind71(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).quantile(.1)

def l1_ind72(x,y):
    global c
    return first_relat_der(norm(c,x)).rolling(y,min_periods=1).quantile(.9)

def l1_ind73(c):
    return second_relat_der(c)

def l1_ind74(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).mean()

def l1_ind75(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).std().fillna(0)

def l1_ind76(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).median()

def l1_ind77(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).max()

def l1_ind78(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).min()

def l1_ind79(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).quantile(.1)

def l1_ind80(x):
    global c
    return second_relat_der(c).rolling(x,min_periods=1).quantile(.9)

def l1_ind81(x):
    global c
    return second_relat_der(norm(c,x))

def l1_ind82(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).mean()

def l1_ind83(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind84(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).median()

def l1_ind85(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).max()

def l1_ind86(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).min()

def l1_ind87(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).quantile(.1)

def l1_ind88(x,y):
    global c
    return second_relat_der(norm(c,x)).rolling(y,min_periods=1).quantile(.9)

def l1_ind89(c):
    return third_relat_der(c)

def l1_ind90(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).mean()

def l1_ind91(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).std().fillna(0)

def l1_ind92(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).median()

def l1_ind93(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).max()

def l1_ind94(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).min()

def l1_ind95(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).quantile(.1)

def l1_ind96(x):
    global c
    return third_relat_der(c).rolling(x,min_periods=1).quantile(.9)

def l1_ind97(x):
    global c
    return third_relat_der(norm(c,x))

def l1_ind98(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).mean()

def l1_ind99(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind100(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).median()

def l1_ind101(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).max()

def l1_ind102(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).min()

def l1_ind103(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).quantile(.1)

def l1_ind104(x,y):
    global c
    return third_relat_der(norm(c,x)).rolling(y,min_periods=1).quantile(.9)

def l1_ind105(x):
    global c
    return get_autocorr(c,x)

def l1_ind106(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).mean()

def l1_ind107(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).std().fillna(0)

def l1_ind108(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).median()

def l1_ind109(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).max()

def l1_ind110(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).min()

def l1_ind111(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).quantile(.1)

def l1_ind112(x,y):
    global c
    return get_autocorr(c,x).rolling(y,min_periods=1).quantile(.9)

def l1_ind113(x):
    global c
    return get_autocorr(norm(c,x),x)

def l1_ind114(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).mean()

def l1_ind115(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).std().fillna(0)

def l1_ind116(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).median()

def l1_ind117(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).max()

def l1_ind118(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).min()

def l1_ind119(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).quantile(.1)

def l1_ind120(x,y):
    global c
    return get_autocorr(norm(c,x),x).rolling(y,min_periods=1).quantile(.9)

def l1_ind121(x):
    global c
    return (pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)

def l1_ind122(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).mean()

def l1_ind123(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).std().fillna(0)

def l1_ind124(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).median()

def l1_ind125(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).max()

def l1_ind126(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).min()

def l1_ind127(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).quantile(.1)

def l1_ind128(x,y):
    global c
    return ((pd.Series(c)-pd.Series([sorted(dict(collections.Counter(c[max(i-x,0):i])).items(), key=operator.itemgetter(1))[-1][0] for i in range(1,len(c)+1)]))/pd.Series(c)).rolling(y,min_periods=1).quantile(.9)

def backtester():
    # Results will add to the folder results
    dir_results = 'results/'
    data_filename = 'EURUSD5.csv'

    trading_logics = [1,2,3,4,5,6]  #1,2,3,4,5,6
    levels_numbers = [1,2,3]        #1,2,3
    time_filters = [1,0]            #1,0
    vola_filters = [1,0]            #1,0

    need_files = []
    for trading_logic in trading_logics:
        for levels_number in levels_numbers:
            for time_filter in time_filters:
                for vola_filter in vola_filters:
                    if platform.system() == 'Linux':
                        os.system("nohup python v11_main.py "+str([trading_logic,levels_number,time_filter,vola_filter]).replace(' ','')+" &")
                    else:
                        os.system("START /B python v11_main.py "+str([trading_logic,levels_number,time_filter,vola_filter]).replace(' ',''))
                    need_files.append(data_filename+'_'+ str(trading_logic)+'_'+str(levels_number)+'_'+str(time_filter)+'_'+str(vola_filter)+'_all_results_statistic.csv')

    while True:
        files = os.listdir(dir_results)
        if len(set(need_files) - set(files)) == 0:
            df = pd.DataFrame()
            for trading_logic in trading_logics:
                for levels_number in levels_numbers:
                    for time_filter in time_filters:
                        for vola_filter in vola_filters:            
                            df_stats = pd.read_csv(dir_results+data_filename+'_'+ str(trading_logic)+'_'+str(levels_number)+'_'+str(time_filter)+'_'+str(vola_filter)+'_all_results_statistic.csv', sep=';')
                            df_stats['levels_number'] = levels_number
                            df = df.append(df_stats)
            df = df[['code', 'trading_logic', 'levels_number', 'reverse_logic', 'pf', 'pf_buy', 'pf_sell', 'total_profit', 'average_profit', 'average_loss', 'probability_profit', 'probability_loss',\
                     'quatity_of_trades', 'quatity_of_trades_buy', 'quatity_of_trades_sell', 'std', 'max_dd', 'level_open_buy', 'level_open_sell', 'level_close_buy', 'level_close_sell',\
                     'time_filter', 'vola_filter', 'end_time', 'start_time', 'atr_window', 'decil_window', 'vola_filter_side', 'vola_filter_value', 'take_profit', 'stop_loss',\
                     'middle_line_window', 'new_volume', 'step_size', 'step_quantity', 'hard_stop_loss',' zero_line']]
            df.to_csv('best_'+dir_results+data_filename+'_all_results_statistic.csv', sep=';', index=False)
            break
        time.sleep(1)

    need_best_results = [x.split('_')[1]+'_'+x.split('_')[2] for x in os.listdir('best_'+dir_results) if 'png' in x]
    df = df[df.time_filter==1]
    df = df[df.code.isin(need_best_results)]

    df_need_results = pd.DataFrame(columns=df.columns)
    for trading_logic in trading_logics:
        if trading_logic in [3,5]:
            for pf_side in ['pf']:
                df_temp = df[(df.trading_logic==trading_logic)&(df[pf_side] > 1.8)].copy()
                df_temp.sort_values(by=['reverse_logic', 'start_time', pf_side], ascending=[False, True, False], inplace=True)
                for reverse_logic in set(df_temp.reverse_logic):
                    for start_time in set(df_temp.start_time):
                        df_to_add = df_temp[(df_temp.reverse_logic==reverse_logic)&(df_temp.start_time==start_time)]
                        df_to_add['pf_side'] = pf_side
                        df_need_results = df_need_results.append(df_to_add[:1])
        elif trading_logic in [1,2,4,6]:
            for pf_side in ['pf_buy', 'pf_sell']:
                df_temp = df[(df.trading_logic==trading_logic)&(df[pf_side] > 1.8)].copy()
                df_temp.sort_values(by=['reverse_logic', 'start_time', pf_side], ascending=[False, True, False], inplace=True)
                for reverse_logic in set(df_temp.reverse_logic):
                    for start_time in set(df_temp.start_time):
                        df_to_add = df_temp[(df_temp.reverse_logic==reverse_logic)&(df_temp.start_time==start_time)]
                        df_to_add['pf_side'] = pf_side
                        df_need_results = df_need_results.append(df_to_add[:1])

    df_need_results.set_index('code', inplace=True)
    txt_results_files = [x for x in os.listdir(dir_results) if 'png' not in x and '_all_results_statistic' not in x and x.split('_')[1]+'_'+x.split('_')[2] in df_need_results.index]
    if len(txt_results_files)>0:
        for file_name in txt_results_files:
            code = file_name.split('_')[1]+'_'+file_name.split('_')[2]
            df_results = pd.read_csv(dir_results+file_name, sep=';')
            if df_need_results.loc[code, 'pf_side'] == 'pf_buy': df_results = df_results[df_results.Side=='Buy']
            if df_need_results.loc[code, 'pf_side'] == 'pf_sell': df_results = df_results[df_results.Side=='Sell']
            std = df_results['Result'].std()
            df_need_results.loc[code, 'std'] = std
        df_need_results['koef'] = min(df_need_results['std']) / df_need_results['std']
        df_need_results.to_csv('best_'+dir_results+data_filename+'_need_results_statistic.csv', sep=';')

        chosen_results = pd.DataFrame()
        for file_name in txt_results_files:
            code = file_name.split('_')[1]+'_'+file_name.split('_')[2]
            koef = df_need_results.loc[code, 'koef']
            df_results = pd.read_csv(dir_results+file_name, sep=';')
            if df_need_results.loc[code, 'pf_side'] == 'pf_buy': df_results = df_results[df_results.Side=='Buy']
            if df_need_results.loc[code, 'pf_side'] == 'pf_sell': df_results = df_results[df_results.Side=='Sell']
            df_results['koef'] = koef
            df_results['Result'] = df_results['Result'] * koef
            df_results['trading_logic'] = int(file_name.split('_')[1][0])
            chosen_results = chosen_results.append(df_results)
        chosen_results = chosen_results.sort_values(by=['OpenTime'])
        chosen_results = chosen_results[['OpenTime','OpenPrice','Side','CloseTime','ClosePrice','Result','koef','trading_logic']]
        chosen_results.to_csv('best_'+dir_results+data_filename+'_all_chosen_results.csv', sep=';')
        for trading_logic in trading_logics:
            chosen_results_logic = chosen_results[chosen_results.trading_logic == trading_logic].copy()
            chosen_results_logic.to_csv('best_'+dir_results+data_filename+'_'+str(trading_logic)+'_chosen_results.csv', sep=';')

        chosen_results['OpenTime'] = pd.to_datetime(chosen_results['OpenTime'], format = '%Y-%m-%d %H:%M:%S')
        chosen_results['CloseTime'] = pd.to_datetime(chosen_results['CloseTime'], format = '%Y-%m-%d %H:%M:%S')
        chosen_results = chosen_results.sort_values(by=['CloseTime'])
        #chosen_results['Equity'] = np.cumsum(chosen_results['Result'])
        df_buy = chosen_results[chosen_results.Side == 'Buy'].copy()
        df_sell = chosen_results[chosen_results.Side == 'Sell'].copy()
        df_chosen_results = chosen_results.groupby(['CloseTime'])['Result'].sum().to_frame()
        df_buy = df_buy.groupby(['CloseTime'])['Result'].sum().to_frame()
        df_sell = df_sell.groupby(['CloseTime'])['Result'].sum().to_frame()

        plt.title('Manual Choice')
        plt.plot(df_chosen_results.index, np.cumsum(df_chosen_results['Result']), label = 'Equity')
        plt.plot(df_buy.index, np.cumsum(df_buy['Result']), label = 'Buy')
        plt.plot(df_sell.index, np.cumsum(df_sell['Result']), label = 'Sell')
        plt.xticks(rotation=45)
        plt.xlabel('Dates')
        plt.ylabel('$')
        plt.legend()
        plt.savefig('best_'+dir_results+data_filename+'_all.jpg')
        plt.close()

        for trading_logic in trading_logics:
            df_chosen_results_logic = chosen_results[chosen_results.trading_logic == trading_logic].copy()
            df_buy = df_chosen_results_logic[df_chosen_results_logic.Side == 'Buy'].copy()
            df_sell = df_chosen_results_logic[df_chosen_results_logic.Side == 'Sell'].copy()
            df_chosen_results_logic = df_chosen_results_logic.groupby(['CloseTime'])['Result'].sum().to_frame()
            df_buy = df_buy.groupby(['CloseTime'])['Result'].sum().to_frame()
            df_sell = df_sell.groupby(['CloseTime'])['Result'].sum().to_frame()

            plt.title('Manual Choice')
            plt.plot(df_chosen_results_logic.index, np.cumsum(df_chosen_results_logic['Result']), label = 'Equity')
            plt.plot(df_buy.index, np.cumsum(df_buy['Result']), label = 'Buy')
            plt.plot(df_sell.index, np.cumsum(df_sell['Result']), label = 'Sell')
            plt.xticks(rotation=45)
            plt.xlabel('Dates')
            plt.ylabel('$')
            plt.legend()
            plt.savefig('best_'+dir_results+data_filename+'_'+str(trading_logic)+'.jpg')
            plt.close()
    
def feature_selector(X,y,mode='corr',top=5):
    import pandas as pd
    import numpy as np
    
    X = np.nan_to_num(X)
    X[X > 1e10] = 1e10
    X[X < -1e10] = -1e10
    if mode == 'corr':
        target = pd.Series(y)
        corr = []
        for i in range(len(X[0])):
            series = pd.Series(X[:,i])
            corr.append(np.abs(target.corr(series)))
        corr = np.nan_to_num(np.array(corr))
        selected_features = np.argsort(corr)[-top:]
    elif mode == 'xgb':
        clf = xgb.XGBClassifier(n_jobs=8)
        clf.fit(X,y)
        selected_features = np.argsort(clf.feature_importances_)[-top:]
    elif mode == 'tree':
        qualities = []
        for i in range(len(X[0])):
            clf = DecisionTreeClassifier()
            qualities.append(cross_val_score(clf,X[:,i].reshape(-1,1),y,cv=3,n_jobs=3).mean())
        qualities = np.array(qualities)
        selected_features = np.argsort(qualities)[-top:]
#     elif mode == 'xgb_brute':
    elif mode == 'advanced':
        target = pd.Series(y)
        corr = []
        for i in range(len(X[0])):
            series = pd.Series(X[:,i])
            corr.append(np.abs(target.corr(series)))
        corr = np.nan_to_num(np.array(corr))
        selected_features = np.argsort(corr)[-15:]
        
        umax = 0
        for i in range(2**15-1):
            mask = np.array(list("0"*(15-len(bin(i))+2)+bin(i)[2:]),dtype=int)
            candidates = selected_features[np.where(mask == 1)[0]]
            if len(candidates) not in [2,3,4]:
                continue
            n = len(candidates)
            rys = np.mean(np.abs([pd.Series(X[:,k]).corr(pd.Series(y)) for k in candidates]))
            m = np.mean(np.abs(np.corrcoef(np.matrix(X[:,candidates]).T)))
            rss = (m*n**2-n)/(n**2-n)
            u = n*rys/np.sqrt(n+n*(n-1)*rss)
            if u > umax:
                umax = u
                best_config = candidates
        selected_features = best_config
    elif mode == 'advanced2':
        target = pd.Series(y)
        corr = []
        for i in range(len(X[0])):
            series = pd.Series(X[:,i])
            corr.append(np.abs(target.corr(series)))
        corr = np.nan_to_num(np.array(corr))
        selected_features = np.argsort(corr)[-15:]
        
        
        umax = 0
        for i in range(2**15-1):
            mask = np.array(list("0"*(15-len(bin(i))+2)+bin(i)[2:]),dtype=int)
            candidates = selected_features[np.where(mask == 1)[0]]
            if len(candidates) > 7:
                continue
            rsy = corr[candidates]
            rss_inv = np.linalg.inv(np.matrix(np.corrcoef(np.matrix(X[:,candidates]).T)))
            u = np.sqrt(rsy.reshape(1,-1).dot(rss_inv).dot(rsy.reshape(-1,1))[0,0])
            if u > umax:
                umax = u
                best_config = candidates
        selected_features = best_config
        
    return X[:,selected_features], selected_features