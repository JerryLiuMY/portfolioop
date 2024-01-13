import numpy as np
from gekko import GEKKO
from tran_obje.cvx import CVXTran
from tran_obje.gek import GekTran
from tran_obje.grb import GRBTran
from tran_obje.nlo import NloTran
from tran_obje.pym import PymTran
from tran_obje.sci import SciTran


def test_tran(tran_type):
    assert tran_type in ['line', 'quad', 'gene']
    model = GEKKO(remote=False)
    num_com = 100
    cvx_tran = CVXTran(date='2010-03-02', num_com=num_com, tran_type=tran_type)
    gek_tran = GekTran(date='2010-03-02', num_com=num_com, tran_type=tran_type, model=model)
    grb_tran = GRBTran(date='2010-03-02', num_com=num_com, tran_type=tran_type)
    nlo_tran = NloTran(date='2010-03-02', num_com=num_com, tran_type=tran_type)
    pym_tran = PymTran(date='2010-03-02', num_com=num_com, tran_type=tran_type)
    sci_tran = SciTran(date='2010-03-02', num_com=num_com, tran_type=tran_type)

    cvx_cost = cvx_tran.get_cost(np.zeros(num_com), np.ones(num_com)/num_com)
    gek_cost = gek_tran.get_cost(np.zeros(num_com), np.ones(num_com)/num_com)
    grb_cost = grb_tran.get_cost(np.ones(num_com)/num_com, np.zeros(num_com))
    nlo_cost = nlo_tran.get_cost(np.zeros(num_com), np.ones(num_com)/num_com)
    pym_cost = pym_tran.get_cost(np.zeros(num_com), np.ones(num_com)/num_com)
    sci_cost = sci_tran.get_cost(np.zeros(num_com), np.ones(num_com)/num_com)

    print(f'cvxpy: {cvx_cost.value}')
    print(f'gekko: {gek_cost.value}')
    print(f'gurobi: {grb_cost.getValue()}')
    print(f'nlopt: {nlo_cost}')
    print(f'pyomo: {pym_cost}')
    print(f'scipy: {sci_cost}')

