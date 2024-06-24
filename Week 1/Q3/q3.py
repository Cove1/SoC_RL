from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    S = v * (t - (1 - np.exp((-k) * t)) / k)
    return S
    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
# def find_constants(df: pd.DataFrame):

    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    print(df)
    parameters, covariance = curve_fit(func, df['t'] , df['S'])
    parameters = np.round(parameters, 4)
    v = parameters[0]
    k = parameters[1]
    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
#   v, k = find_constants(df)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    S_fit = func(df['t'], v, k)
    plt.plot(df['t'], df['S'], marker='o', linestyle='', markersize=8, label = 'data')
    plt.plot(df['t'], S_fit, color = 'red', label = 'fit: v = '+ str(v) + " k = "+ str(k))
    plt.xlabel('t')
    plt.ylabel('S')
    plt.legend()
    plt.savefig('fit_curve.png')
    plt.show()
    # END TODO
