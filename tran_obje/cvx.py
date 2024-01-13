import numpy as np
import cvxpy as cp
from tran_obje.base import BaseTran, BaseObje


class CVXTran(BaseTran):
    def __init__(self, date, num_com, tran_type):
        super().__init__(date, num_com, tran_type)

    @staticmethod
    def none_cost(weights_i, weights_f):
        cost = 0.0

        return cost

    def line_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        buy_diffs = cp.maximum(diffs, 0)
        sel_diffs = -cp.minimum(diffs, 0)
        buy_cost = cp.sum(buy_diffs) * self.buy_b
        sel_cost = cp.sum(sel_diffs) * self.sel_b
        cost = buy_cost + sel_cost

        return cost

    def quad_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        buy_diffs = cp.maximum(diffs, 0)
        sel_diffs = -cp.minimum(diffs, 0)
        buy_cost = cp.sum(buy_diffs * self.buy_b + cp.square(buy_diffs) * self.buy_k)
        sel_cost = cp.sum(sel_diffs * self.sel_b + cp.square(sel_diffs) * self.sel_k)
        cost = cp.sum(buy_cost + sel_cost)

        return cost

    def gene_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        factors = cp.abs(self.expo / (self.clos * self.vlms) * 100)
        cost_abs = cp.sum(cp.multiply(
            abs(self.p0) +
            abs(self.p2) * (self.year - 1925) + abs(self.p7) * self.vix * 100 +
            abs(self.p3) * np.log(1 + self.caps / 10**9) +
            abs(self.p6) * self.stds * 100,
            cp.abs(diffs)
        ))
        cost_quad = cp.sum(cp.multiply(abs(self.p4) * factors, cp.square(diffs)))
        cost_pow = cp.sum(cp.multiply(abs(self.p5) * factors, cp.power(cp.abs(diffs), 2)))
        cost = (cost_abs + cost_quad + cost_pow) / 10000

        return cost


class CVXObje(BaseObje):
    def __init__(self, date, num_com):
        super().__init__(date, num_com)

    def get_ret(self, weights):
        ret = weights.T @ (np.ones(len(self.ave_vec)) + self.ave_vec)

        return ret

    def get_var(self, weights):
        var = cp.quad_form(weights, self.cov_mtx)

        return var
