import numpy as np
from config.tran_config import expo
from config.tran_config import tran_dict
from data.data_loader import load_summary
from data.data_loader import load_param


class BaseTran:
    def __init__(self, date, num_com, tran_type):
        self.date = date
        self.num_com = num_com
        self.tran_type = tran_type
        self._load_params()

    def _load_params(self):
        gene_param, line_param, quad_param = tran_dict["gene"], tran_dict["line"], tran_dict["quad"]
        self.p0, self.p1, self.p2, self.p3 = gene_param["p0"], gene_param["p1"], gene_param["p2"], gene_param["p3"]
        self.p4, self.p5, self.p6, self.p7 = gene_param["p4"], gene_param["p5"], gene_param["p6"], gene_param["p7"]
        self.buy_b, self.sel_b = line_param["buy_b"], line_param["sel_b"]
        self.buy_k, self.sel_k = quad_param["buy_k"], quad_param["sel_k"]
        self.year, self.clos, self.vlms, self.caps, self.stds, self.vix = load_param(self.date, self.num_com)
        self.expo = expo

    # diffs_pos, diffs_neg for grb and pym
    # weights_i, weights for all others
    def get_cost(self, weights_i, weights_f):
        if self.tran_type == 'none':
            cost_func = self.none_cost
        elif self.tran_type == 'line':
            cost_func = self.line_cost
        elif self.tran_type == 'quad':
            cost_func = self.quad_cost
        elif self.tran_type == 'gene':
            cost_func = self.gene_cost
        else:
            raise AssertionError('Invalid cost function')

        cost = cost_func(weights_i, weights_f)

        return cost

    @staticmethod
    def none_cost(weights_i, weights_f):

        return None

    def line_cost(self, weights_i, weights_f):

        return None

    def quad_cost(self, weights_i, weights_f):

        return None

    def gene_cost(self, weights_i, weights_f):

        return None


class BaseObje:
    def __init__(self, date, num_com):
        summary = load_summary(date, num_com)
        self.date, self.num_com = date, num_com
        self.ave_vec = summary[0]
        self.cov_mtx = summary[1]

    def get_obje(self, weights):
        ret = self.get_ret(weights)
        var = self.get_var(weights)
        obje = ret - 0.5 * var

        return -obje

    def get_ret(self, weights):

        return np.nan

    def get_var(self, weights):

        return np.nan
