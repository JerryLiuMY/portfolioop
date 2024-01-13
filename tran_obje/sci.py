import numpy as np
from tran_obje.base import BaseTran, BaseObje


class SciTran(BaseTran):
    def __init__(self, date, num_com, tran_type):
        super().__init__(date, num_com, tran_type)

    @staticmethod
    def none_cost(weights_i, weights_f):
        cost = 0.0

        return cost

    def line_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        buy_diff = diffs[diffs > 0]
        sel_diff = -diffs[diffs < 0]
        buy_cost = np.sum(buy_diff) * self.buy_b
        sel_cost = np.sum(sel_diff) * self.sel_b
        cost = buy_cost + sel_cost

        return cost

    def quad_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        buy_diff = diffs[diffs > 0]
        sel_diff = -diffs[diffs < 0]
        buy_cost = (buy_diff * self.buy_b + buy_diff**2 * self.buy_k).sum()
        sel_cost = (sel_diff * self.sel_b + sel_diff**2 * self.sel_k).sum()
        cost = buy_cost + sel_cost

        return cost

    def gene_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        factors = np.abs(self.expo / (self.clos * self.vlms) * 100)
        cost_abs = np.sum(np.multiply(
            abs(self.p0) +
            abs(self.p2) * (self.year - 1925) + abs(self.p7) * self.vix * 100 +
            abs(self.p3) * np.log(1 + self.caps / 10 ** 9) +
            abs(self.p6) * self.stds * 100,
            np.abs(diffs)
        ))
        cost_quad = np.sum(abs(self.p4) * np.multiply(factors, np.square(diffs)))
        cost_pow = np.sum(abs(self.p5) * np.multiply(factors, np.power(np.abs(diffs), 2)))
        cost = (cost_abs + cost_quad + cost_pow) / 10000

        return cost


class SciObje(BaseObje):
    def __init__(self, date, num_com):
        super().__init__(date, num_com)

    def get_ret(self, weights):
        ret = np.dot(weights.T, np.ones(len(self.ave_vec)) + self.ave_vec)

        return ret

    def get_var(self, weights):
        var = np.dot(weights.T, np.dot(self.cov_mtx, weights))

        return var
