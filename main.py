from config.ms_config import ms
from global_settings import DATA_FOLDER
from tools.utils import create_path
from model.experiment import evaluation, backtest
import pandas as pd
import numpy as np
import math
import os


def run_experiment(exp_type):
    date, num_com = '2015-06-24', 500
    (lim, sep), (year_beg, year_end) = (500, 1.5), (2015, 2017)
    num_coms = [int(i ** sep) for i in range(2, math.ceil(math.pow(lim, 1 / sep)) + 1)]
    dates = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'))['Date']
    dates = list(dates[dates.apply(lambda _: int(_[:4]) in np.arange(year_beg, year_end))])
    exp_path = create_path(os.path.join(DATA_FOLDER, exp_type))

    for tran_type in ['none', 'line', 'quad', 'gene']:
        if exp_type == 'evaluation':
            dfs = evaluation(date=date, num_coms=num_coms, ms=ms, tran_type=tran_type)
        elif exp_type == 'backtest':
            dfs = backtest(dates=dates, num_com=num_com, ms=ms, tran_type=tran_type)
        else:
            raise AssertionError('Invalid experiment type')
        eff_df, sol_df, ret_df, wgt_df, alc_df = dfs
        eff_df.to_pickle(os.path.join(exp_path, f'{tran_type}_eff.pkl'))
        sol_df.to_pickle(os.path.join(exp_path, f'{tran_type}_sol.pkl'))
        ret_df.to_pickle(os.path.join(exp_path, f'{tran_type}_ret.pkl'))
        wgt_df.to_pickle(os.path.join(exp_path, f'{tran_type}_wgt.pkl'))
        alc_df.to_pickle(os.path.join(exp_path, f'{tran_type}_alc.pkl'))


if __name__ == '__main__':
    run_experiment('backtest')
    run_experiment('evaluation')
