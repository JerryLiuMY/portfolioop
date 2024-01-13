from tran_obje.pym import PymTran
from tran_obje.pym import PymObje
from gurobipy import quicksum


class GRBTran(PymTran):
    def __init__(self, date, num_com, tran_type):
        super().__init__(date, num_com, tran_type)
        self.sum_func = quicksum


class GRBObje(PymObje):
    def __init__(self, date, num_com):
        super().__init__(date, num_com)
        self.sum_func = quicksum
