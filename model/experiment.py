import pandas as pd
import numpy as np
from model.optimizer import Optimizer
from tools.utils import get_ret
from tools.utils import get_alc


def evaluation(date, num_coms, ms, tran_type):
    eff_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(num_coms), dtype=np.float64)
    sol_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(num_coms), dtype=np.float64)
    ret_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(num_coms), dtype=np.float64)
    wgt_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(num_coms))
    alc_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(num_coms))

    for (m, s) in ms:
        print('\n')
        print('#######################')
        print(f'#### {m} & {s} ####')
        print('#######################')

        for num_com in num_coms:
            print('\n')
            print(f'### Company Number = {num_com} ###')

            # time and solve
            weights_i = np.zeros(num_com)
            optimizer = Optimizer(date, num_com, weights_i=weights_i, tran_type=tran_type)
            lapse, solution, weights = timer(load_opt(optimizer, m))(s)
            if weights is not None:
                ret = get_ret(date, num_com, weights)
                alc = get_alc(date, num_com, weights_i, weights, tran_type)
                eff_df.loc[f'({m}, {s})', num_com] = lapse
                sol_df.loc[f'({m}, {s})', num_com], wgt_df.loc[f'({m}, {s})', num_com] = solution, weights
                ret_df.loc[f'({m}, {s})', num_com], alc_df.loc[f'({m}, {s})', num_com] = ret, alc

    return eff_df, sol_df, ret_df, wgt_df, alc_df


def backtest(dates, num_com, ms, tran_type):
    eff_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(dates), dtype=np.float64)
    sol_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(dates), dtype=np.float64)
    ret_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(dates), dtype=np.float64)
    wgt_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(dates))
    alc_df = pd.DataFrame(index=[f'({m}, {s})' for (m, s) in ms], columns=list(dates))

    for (m, s) in ms:
        print('\n')
        print('#######################')
        print(f'#### {m} & {s} ####')
        print('#######################')
        weights_i = np.ones(num_com) / num_com

        for date in dates:
            print('\n')
            print(f'### Date = {date} ###')

            # time and solve
            optimizer = Optimizer(date, num_com, weights_i=weights_i, tran_type=tran_type)
            lapse, solution, weights = timer(load_opt(optimizer, m))(s)
            if weights is not None:
                ret = get_ret(date, num_com, weights)
                alc = get_alc(date, num_com, weights_i, weights, tran_type)
                eff_df.loc[f'({m}, {s})', date] = lapse
                sol_df.loc[f'({m}, {s})', date], wgt_df.loc[f'({m}, {s})', date] = solution, weights
                ret_df.loc[f'({m}, {s})', date], alc_df.loc[f'({m}, {s})', date] = ret, alc
                weights_i = weights / weights.sum()

    return eff_df, sol_df, ret_df, wgt_df, alc_df


def load_opt(optimizer, m):
    if m == 'Scipy':
        opt = optimizer.scipy_opt
    elif m == 'NLopt':
        opt = optimizer.nlopt_opt
    elif m == 'CVXPY':
        opt = optimizer.cvxpy_opt
    elif m == 'GEKKO':
        opt = optimizer.gekko_opt
    elif m == 'Pyomo':
        opt = optimizer.pyomo_opt
    elif m == 'GUROBI':
        opt = optimizer.gurobi_opt
    else:
        raise AssertionError('Invalid modelling tool')

    return opt


def timer(func):
    def wrapper(*args, **kwargs):
        import time
        beg = time.time()
        outputs = func(*args, **kwargs)
        end = time.time()
        success, solution, weights = outputs
        lapse = round(end - beg, 5) if success else np.nan
        print("Success:", success)
        print("Time lapsed:", lapse)
        print("Solution:", solution)
        print(f"Weights:", sum(weights)) if weights is not None else print(f"Weights: None")
        return lapse, solution, weights
    return wrapper
