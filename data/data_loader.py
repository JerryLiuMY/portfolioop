import os
import pickle
import numpy as np
from global_settings import DATA_FOLDER


def load_param(date, num_com):
    param_dir = os.path.join(DATA_FOLDER, 'param')
    with open(os.path.join(param_dir, f'{date}.pkl'), 'rb') as handle:
        param = pickle.load(handle)

    year = param['year']
    clos = np.array(param['clos'])[:num_com]
    vlms = np.array(param['vlms'])[:num_com]
    caps = np.array(param['caps'])[:num_com]
    stds = np.array(param['stds'])[:num_com]
    vix = param['vix']

    return year, clos, vlms, caps, stds, vix


def load_summary(date, num_com):
    summary_dir = os.path.join(DATA_FOLDER, 'summary')
    with open(os.path.join(summary_dir, f'{date}.pkl'), 'rb') as handle:
        summary = pickle.load(handle)

    betas, ivols = np.array(summary['betas'])[:num_com], np.array(summary['ivols'])[:num_com]
    rp, rf, rp_var, rf_var = summary['rp'], summary['rf'], summary['rp_var'], summary['rf_var']

    ave_vec = betas * rp + rf
    cov_mtx = np.outer(betas, betas) * rp_var + rf_var + np.diag(ivols ** 2)

    return ave_vec, cov_mtx
