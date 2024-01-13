import numpy as np
from tran_obje.base import BaseTran, BaseObje


class PymTran(BaseTran):
    def __init__(self, date, num_com, tran_type):
        super().__init__(date, num_com, tran_type)
        self.idx = range(self.num_com)
        self.sum_func = sum

    @staticmethod
    def none_cost(diffs_pos, diffs_neg):
        cost = 0.0

        return cost

    def line_cost(self, diffs_pos, diffs_neg):
        buy_cost = self.sum_func(diffs_pos[i] for i in self.idx) * self.buy_b
        sel_cost = self.sum_func(-diffs_neg[i] for i in self.idx) * self.sel_b
        cost = buy_cost + sel_cost

        return cost

    def quad_cost(self, diffs_pos, diffs_neg):
        buy_cost = self.sum_func(diffs_pos[i] * self.buy_b + diffs_pos[i] * diffs_pos[i] * self.buy_k
                                 for i in self.idx)
        sel_cost = self.sum_func(-diffs_neg[i] * self.sel_b + diffs_neg[i] * diffs_neg[i] * self.sel_k
                                 for i in self.idx)
        cost = buy_cost + sel_cost

        return cost

    def gene_cost(self, diffs_pos, diffs_neg):
        factors = np.abs(self.expo / (self.clos * self.vlms) * 100)
        diff = [(diffs_pos[i] - diffs_neg[i]) for i in self.idx]
        cost_abs = self.sum_func((
            abs(self.p0) +
            abs(self.p2) * (self.year - 1925) + abs(self.p7) * self.vix * 100 +
            abs(self.p3) * np.log(1 + self.caps[i] / 10**9) +
            abs(self.p6) * self.stds[i] * 100)
            * diff[i] for i in self.idx
        )
        cost_quad = self.sum_func(abs(self.p4) * factors[i] * (diff[i] * diff[i]) for i in self.idx)
        cost_pow = self.sum_func(abs(self.p5) * factors[i] * (diff[i] * diff[i]) for i in self.idx)
        cost = (cost_abs + cost_quad + cost_pow) / 10000

        return cost


class PymObje(BaseObje):
    def __init__(self, date, num_com):
        super().__init__(date, num_com)
        self.idx = range(self.num_com)
        self.sum_func = sum

    def get_ret(self, weights):
        ret = self.sum_func(weights[i] * (1 + self.ave_vec[i]) for i in self.idx)

        return ret

    def get_var(self, weights):
        var = self.sum_func(
            self.sum_func(self.cov_mtx[i, j] * weights[i] for i in self.idx) * weights[j] for j in self.idx
        )

        return var
