from tran_obje.base import BaseTran
import numpy as np
from tran_obje.base import BaseObje


class GekTran(BaseTran):
    def __init__(self, date, num_com, tran_type, model):
        super().__init__(date, num_com, tran_type)
        self.idx = range(self.num_com)
        self.model = model

    @staticmethod
    def none_cost(weights_i, weights_f):
        cost = 0.0

        return cost

    def line_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        buy_cost = self.model.sum([self.model.max2(diffs[i], 0) for i in self.idx]) * self.buy_b
        sel_cost = self.model.sum([-self.model.min2(diffs[i], 0) for i in self.idx]) * self.sel_b
        cost = buy_cost + sel_cost

        return cost

    def quad_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        buy_cost = self.model.sum([
            self.buy_b * (self.model.max2(diffs[i], 0)) +
            self.buy_k * (self.model.max2(diffs[i], 0))**2
            for i in range(self.num_com)
        ])

        sel_cost = self.model.sum([
            self.sel_b * -(self.model.min2(diffs[i], 0)) +
            self.sel_k * (self.model.min2(diffs[i], 0))**2
            for i in range(self.num_com)
        ])

        cost = buy_cost + sel_cost

        return cost

    def gene_cost(self, weights_i, weights_f):
        diffs = weights_f - weights_i
        factors = np.abs(self.expo / (self.clos * self.vlms) * 100)
        cost_abs = self.model.sum((
            abs(self.p0) +
            abs(self.p2) * (self.year - 1925) + abs(self.p7) * self.vix * 100 +
            abs(self.p6) * self.stds[i] * 100 +
            abs(self.p3) * np.log(1 + self.caps[i] / 10**9))
            * self.model.abs2(diffs[i]) for i in self.idx
        )
        cost_quad = self.model.sum(abs(self.p4) * factors[i] * (diffs[i])**2 for i in self.idx)
        cost_pow = self.model.sum(abs(self.p5) * factors[i] * self.model.abs2(diffs[i])**2 for i in self.idx)
        cost = (cost_abs + cost_quad + cost_pow) / 10000

        return cost


class GekObje(BaseObje):
    def __init__(self, date, num_com, model):
        super().__init__(date, num_com)
        self.idx = range(self.num_com)
        self.model = model

    def get_ret(self, weights):
        # use self.model.sum to avoid maximum length error
        ret = self.model.sum([weights[i] * (1 + self.ave_vec[i]) for i in self.idx])

        return ret

    def get_var(self, weights):
        var = self.model.sum(
            [self.model.sum([self.cov_mtx[i, j] * weights[i] for i in self.idx]) * weights[j] for j in self.idx]
        )

        return var
