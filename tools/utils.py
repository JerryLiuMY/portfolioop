import os
import numpy as np
import seaborn as sns
from matplotlib import colors as plc
from data.data_loader import load_summary
from tran_obje.base import BaseTran


def create_path(log_dir):
    old_dirs = next(os.walk(log_dir))[1]
    new_num = 0 if len(old_dirs) == 0 else np.max([int(past_dir.split('_')[-1]) for past_dir in old_dirs]) + 1
    exp_path = os.path.join(log_dir, '_'.join(['experiment', str(new_num)]))
    os.mkdir(exp_path)

    return exp_path


def get_ret(date, num_com, weights):
    assert len(weights) == num_com, 'Invalid weights'
    ave_vec = load_summary(date, num_com)[0]
    ret = np.dot(weights.T, np.ones(len(ave_vec)) + ave_vec)

    return ret


def get_alc(date, num_com, weights_i, weights_f, tran_type):
    base_tran = BaseTran(date, num_com, tran_type)
    diffs = weights_f - weights_i
    buy_diff = np.array([max(diff, 0.0) for diff in diffs])
    sel_diff = np.array([min(diff, 0.0) for diff in diffs])
    assert len(buy_diff) == len(sel_diff) and np.dot(buy_diff, sel_diff) == 0.0

    if tran_type == 'none':
        weights_cost = np.zeros(num_com)
    elif tran_type == 'line':
        weights_buy_cost = buy_diff * base_tran.buy_b
        weights_sel_cost = sel_diff * base_tran.sel_b
        weights_cost = weights_buy_cost + weights_sel_cost
    elif tran_type == 'quad':
        weights_buy_cost = buy_diff * base_tran.buy_b + buy_diff**2 * base_tran.buy_k
        weights_sel_cost = sel_diff * base_tran.sel_b + sel_diff**2 * base_tran.sel_k
        weights_cost = weights_buy_cost + weights_sel_cost
    elif tran_type == 'gene':
        factors = np.abs(base_tran.expo / (base_tran.clos * base_tran.vlms) * 100)
        weights_cost_abs = np.multiply(
            abs(base_tran.p0) +
            abs(base_tran.p2) * (base_tran.year - 1925) + abs(base_tran.p7) * base_tran.vix * 100 +
            abs(base_tran.p3) * np.log(1 + base_tran.caps / 10 ** 9) +
            abs(base_tran.p6) * base_tran.stds * 100,
            np.abs(diffs)
        )
        weights_cost_quad = abs(base_tran.p4) * np.multiply(factors, np.square(diffs))
        weights_cost_pow = abs(base_tran.p5) * np.multiply(factors, np.power(np.abs(diffs), 2))
        weights_cost = (weights_cost_abs + weights_cost_quad + weights_cost_pow) / 10000
    else:
        raise AssertionError('Invalid cost function')

    return weights_f + weights_cost


ms = [
    ('Scipy', 'SLSQP'), ('CVXPY', 'CVXOPT'), ('CVXPY', 'ECOS'),
    ('CVXPY', 'XPRESS'), ('CVXPY', 'GUROBI'), ('CVXPY', 'MOSEK'),
    ('Pyomo', 'IPOPT'), ('Pyomo', 'SCIP'), ('Pyomo', 'BONMIN'), ('GUROBI', 'GUROBI')
]

tt = ['none', 'line', 'quad', 'gene']
ms_pal = {f'({_ms_[0]}, {_ms_[1]})': pal for _ms_, pal in zip(ms, sns.color_palette(n_colors=len(ms))[::-1])}
tt_pal = {_tt_: plc.to_hex(pal, keep_alpha=True) for _tt_, pal in zip(tt, sns.color_palette(n_colors=len(tt)))}
