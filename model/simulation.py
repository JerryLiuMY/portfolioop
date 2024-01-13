import os
import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm_notebook
from global_settings import DATA_FOLDER
from tran_obje.sci import SciTran, SciObje


def simulation(date, tran_type):
    samples = 6000
    np.random.seed(42)
    num_col = len(pd.read_pickle(os.path.join(DATA_FOLDER, 'data', 'columns.pkl')))
    sci_tran = SciTran(date, num_com=-1, tran_type=tran_type).get_cost
    sci_obje = SciObje(date, num_com=-1)

    wgt_arr, opt_arr = np.zeros((samples, num_col)), np.zeros((samples, ))
    ret_arr, vol_arr = np.zeros((samples, )), np.zeros((samples, ))

    print(f'\n Simulating {samples} samples')
    for _ in tqdm_notebook(range(samples)):
        weights = np.array(np.random.random(num_col))
        result = optimize.root(
            lambda foo: 1 - sci_tran(np.zeros(num_col), weights/foo) - (weights/foo).sum(),
            x0=np.array([1])
        )
        weights = weights / result.x[0]
        ret, vol = sci_obje.get_ret(weights), sci_obje.get_var(weights)
        opt = sci_obje.get_obje(weights)
        wgt_arr[_], opt_arr[_] = weights, opt
        ret_arr[_], vol_arr[_] = ret, vol

    max_arg = opt_arr.argmax()
    opt, weights = np.amax(opt_arr), wgt_arr[max_arg]
    ret, vol = ret_arr[max_arg], vol_arr[max_arg]
    print(f'\n Optimal = {round(opt, 3)} (ret = {round(ret, 3)} & vol = {round(vol, 3)})')

    return opt, ret, vol


def frontier(date, tran_type):
    num_col = len(pd.read_pickle(os.path.join(DATA_FOLDER, 'data', 'columns.pkl')))
    sci_tran = SciTran(date, num_com=-1, tran_type=tran_type).get_cost
    sci_obje = SciObje(date, num_com=-1)

    init, bounds = np.ones(num_col) * (1 / num_col), [(0, 1) for _ in range(num_col)]
    ret_front, vol_front = np.linspace(0.2, 0.375, 200), np.zeros(200)
    print(f'\n Simulating frontier of {len(ret_front)} samples')
    for idx, front in enumerate(tqdm_notebook(ret_front)):
        constraints = (
            {'type': 'eq', 'fun': lambda weights: 1 - sci_tran(np.zeros(num_col), weights) - np.sum(weights)},
            {'type': 'eq', 'fun': lambda weights: sci_obje.get_ret(weights) - front}
        )
        result = optimize.minimize(sci_obje.get_var, x0=init, bounds=bounds, method='SLSQP', constraints=constraints)
        vol_front[idx] = result['fun']

    return ret_front, vol_front
