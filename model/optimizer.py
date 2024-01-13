import numpy as np
import warnings
import dccp
import nlopt
import cvxpy as cp
import xpress
import gurobipy
from gekko import GEKKO
from gurobipy import GRB
from scipy import optimize
import pyomo.environ as pyomo
from pyomo.opt import SolverStatus
from pyomo.opt import TerminationCondition

from tran_obje.sci import SciTran, SciObje
from tran_obje.nlo import NloTran, NloObje
from tran_obje.cvx import CVXTran, CVXObje
from tran_obje.pym import PymTran, PymObje
from tran_obje.gek import GekTran, GekObje
from tran_obje.grb import GRBTran, GRBObje
patience = float(120)


class Optimizer:
    def __init__(self, date, num_com, weights_i, tran_type):
        self.date = date
        self.num_com = num_com
        self.weights_i = weights_i
        self.tran_type = tran_type
        self._load_funcs()

    def _load_funcs(self):
        self.linear = np.identity(self.num_com)

        self.sci_cost = SciTran(self.date, self.num_com, self.tran_type).get_cost
        self.sci_obje = SciObje(self.date, self.num_com).get_obje

        self.nlo_cost = NloTran(self.date, self.num_com, self.tran_type).get_cost
        self.nlo_obje = NloObje(self.date, self.num_com).get_obje

        self.cvx_cost = CVXTran(self.date, self.num_com, self.tran_type).get_cost
        self.cvx_obje = CVXObje(self.date, self.num_com).get_obje

        self.pym_cost = PymTran(self.date, self.num_com, self.tran_type).get_cost
        self.pym_obje = PymObje(self.date, self.num_com).get_obje

        self.grb_cost = GRBTran(self.date, self.num_com, self.tran_type).get_cost
        self.grb_obje = GRBObje(self.date, self.num_com).get_obje

    def _load_gek_funcs(self, model):
        self.gek_cost = GekTran(self.date, self.num_com, self.tran_type, model).get_cost
        self.gek_obje = GekObje(self.date, self.num_com, model).get_obje

    def scipy_opt(self, solver):
        # constraints
        bounds = [(0, 1) for _ in range(self.num_com)]
        if solver == 'SLSQP':
            options = {'ftol': 1e-6}
            constraints = (
                {'type': 'eq',   'fun': lambda weights: 1 - self.sci_cost(self.weights_i, weights) - np.sum(weights)}
            )
            """{'type': 'ineq', 'fun': lambda weights: 1 - np.max(np.dot(self.linear, weights))}"""
        else:
            raise AssertionError('Invalid solver')

        # solve
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = optimize.minimize(
                self.sci_obje,
                method=solver,
                x0=self.weights_i,
                bounds=bounds,
                constraints=constraints,
                options=options
            )

        success = results['success']
        solution = results['fun'] if success else None
        weights = results['x'] if success else None

        return success, solution, weights

    def nlopt_opt(self, solver):
        # optimizer
        if solver == 'COBYLA':
            optimizer = nlopt.opt(nlopt.LN_COBYLA, self.num_com)
            optimizer.set_maxtime(patience)
        else:
            raise AssertionError('Invalid solver')

        # constraints
        optimizer.set_upper_bounds(np.ones(self.num_com))
        optimizer.set_lower_bounds(np.zeros(self.num_com))
        cons1 = lambda weights, _: 1 - self.nlo_cost(self.weights_i, weights) - np.sum(weights)
        optimizer.add_equality_constraint(cons1, 0.0)
        """
        cons2 = lambda weights, _: np.max(np.dot(self.linear, weights) - 1)
        optimizer.add_inequality_constraint(cons2, 0.0)
        """

        # objective
        optimizer.set_min_objective(self.nlo_obje)

        # solve
        optimizer.set_ftol_rel(1e-6)
        weights = optimizer.optimize(self.weights_i)
        status = optimizer.last_optimize_result()
        success = True if status == 3 else False
        solution = optimizer.last_optimum_value() if success else None
        weights = weights if success else None

        return success, solution, weights

    def cvxpy_opt(self, solver):
        # variables
        weights = cp.Variable(self.num_com, value=self.weights_i, pos=True)

        # constraints
        constraints = [
            weights >= 0,
            weights <= 1,
            1 - self.cvx_cost(self.weights_i, weights) - cp.sum(weights) == 0
        ]
        """1 - cp.max(self.linear @ weights) >= 0"""

        # objective
        problem = cp.Problem(cp.Minimize(self.cvx_obje(weights)), constraints)

        # solve
        try:
            print("Problem is DCCP:", dccp.is_dccp(problem))
            result = problem.solve(qcp=True, method='dccp', solver=solver, warm_start=True)
            status = problem.status
            success = True if status in ['optimal', 'Converged'] else False
            solution = tuple(result)[0] if success else None
            weights = weights.value if success else None
        except (cp.error.SolverError, xpress.SolverError):
            success, solution, weights = False, None, None
            print('CVXPY SolverError')

        return success, solution, weights

    def pyomo_opt(self, solver):
        # variables
        model = pyomo.ConcreteModel()
        model.idx = range(self.num_com)
        initializer = {idx: weight_i for idx, weight_i in zip(model.idx, self.weights_i)}
        model.weights = pyomo.Var(model.idx, initialize=lambda model, i: initializer[i], bounds=(0.0, 1.0))
        model.diffs = pyomo.Var(model.idx, initialize=0.0, bounds=(-1.0, 1.0), domain=pyomo.Reals)
        model.diffs_pos = pyomo.Var(model.idx, initialize=0.0, bounds=(0.0, 1.0), domain=pyomo.NonNegativeReals)
        model.diffs_neg = pyomo.Var(model.idx, initialize=0.0, bounds=(-1.0, 0.0), domain=pyomo.NonPositiveReals)
        model.weights.domain = pyomo.NonNegativeReals

        # constraints: orthogonality automatically ensured
        cons1 = (1 - self.pym_cost(model.diffs_pos, model.diffs_neg) - sum(model.weights[i] for i in model.idx)) == 0
        model.cons1 = pyomo.Constraint(expr=cons1)
        model.consa = pyomo.ConstraintList()
        for i in model.idx:
            cons = (model.diffs[i] == model.weights[i] - self.weights_i[i])
            model.consa.add(cons)
        model.consb = pyomo.ConstraintList()
        for i in model.idx:
            cons = (model.diffs[i] == model.diffs_pos[i] + model.diffs_neg[i])
            model.consb.add(cons)
        # model.consc = pyomo.ConstraintList()
        # for i in model.idx:
        #     cons = (model.diffs_pos[i] * model.diffs_neg[i] == 0.0)
        #     model.consc.add(cons)

        """
        model.cons2 = pyomo.ConstraintList()
        for i in model.idx:
            cons = (1 - sum(self.linear[i, j] * model.weights[j] for j in range(len(self.linear)))) >= 0
            model.cons2.add(cons)
        """

        # objective
        model.objective = pyomo.Objective(expr=self.pym_obje(model.weights), sense=pyomo.minimize)

        # solve
        if solver == 'IPOPT':
            optimizer = pyomo.SolverFactory('IPOPT')
            optimizer.options['max_iter'] = int(1e4)
            optimizer.options['tol'] = 1e-6  # 1e-8
        elif solver == 'BONMIN':
            # solver options cannot be set
            optimizer = pyomo.SolverFactory('BONMIN')
        elif solver == 'SCIP':
            # solver options cannot be set
            optimizer = pyomo.SolverFactory('SCIP')
        elif solver == 'PATH':
            optimizer = pyomo.SolverFactory('path')
        elif solver == 'GUROBI':
            optimizer = pyomo.SolverFactory('gurobi_direct')
            optimizer.options['NonConvex'] = 2
            optimizer.options['FeasibilityTol'] = 1e-6
            optimizer.options['MIPGap'] = 1e-6
        elif solver == 'GECODE':
            optimizer = pyomo.SolverFactory('GECODE')
        else:
            raise AssertionError('Invalid Solver')

        try:
            solution = optimizer.solve(model, tee=False)
            status1, status2 = solution.solver.status, solution.solver.termination_condition
            success = True if (status1 == SolverStatus.ok) and (status2 == TerminationCondition.optimal) else False
            solution = pyomo.value(model.objective) if success else None
            weights = np.array([weight for idx, weight in model.weights.get_values().items()]) if success else None
        # (RuntimeError, ValueError, pyutilib.common._exceptions.ApplicationError)
        except Exception:
            success, solution, weights = False, None, None
            print('Pyomo SolverError')

        # diffs_pos = np.array([weight for idx, weight in model.diffs_pos.get_values().items()]) if success else None
        # diffs_neg = np.array([weight for idx, weight in model.diffs_neg.get_values().items()]) if success else None
        # print(f'constraint violation {1 - self.pym_cost(diffs_pos, diffs_neg) - sum(weights[i] for i in model.idx)}')

        return success, solution, weights

    def gekko_opt(self, solver):
        model = GEKKO(remote=False)
        self._load_gek_funcs(model)
        model.options.MAX_TIME = patience
        model.options.IMODE = 2
        model.options.RTOL = 1e-6
        idx = range(self.num_com)
        weights = model.Array(model.Var, dim=self.num_com, lb=0, ub=1)
        for i in idx:
            weights[i] = self.weights_i[i]

        # constraints
        cons1 = ((1 - self.gek_cost(self.weights_i, weights) - model.sum([foo for foo in weights])) == 0)
        model.Equation(cons1)
        """
        cons2 = ((1 - model.sum([self.linear[i, j] * weights[j] for j in idx])) >= 0 for i in range(len(self.linear)))
        model.Equations(cons2)
        """

        # objective
        model.Obj(obj=self.gek_obje(weights))

        # solve
        if solver == 'APOPT':
            model.options.SOLVER = 1
        elif solver == 'BPOPT':
            model.options.SOLVER = 2
        elif solver == 'IPOPT':
            model.options.SOLVER = 3
        else:
            raise AssertionError('Invalid Solver')

        try:
            model.solve(disp=False)
            success = bool(model.options.solvestatus)
            solution = model.options.objfcnval if success else None
            weights = weights if success else None
        except Exception:
            success, solution, weights = False, None, None
            print('GEKKO SolverError')

        return success, solution, weights

    def gurobi_opt(self, solver):
        # variables
        model = gurobipy.Model()
        model.params.OutputFlag = 0
        model.params.NonConvex = 2
        model.params.TimeLimit = patience
        idx = range(self.num_com)
        weights = model.addVars(idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='weights')
        diffs = model.addVars(idx, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name='diffs')
        diffs_pos = model.addVars(idx, lb=0.0, ub=+1.0, vtype=GRB.CONTINUOUS, name='diffs_pos')
        diffs_neg = model.addVars(idx, lb=-1.0, ub=0.0, vtype=GRB.CONTINUOUS, name='diffs_neg')
        for i in idx:
            weights[i].start, diffs[i].start = self.weights_i[i], 0.0
            diffs_pos[i].start, diffs_neg[i].start = 0.0, 0.0

        # constraints
        cons1 = 1 - self.grb_cost(diffs_pos, diffs_neg) - weights.sum()
        model.addConstr(lhs=cons1, sense=GRB.EQUAL, rhs=0.0, name='cons1')
        model.addConstrs(diffs[i] == weights[i] - self.weights_i[i] for i in range(self.num_com))
        model.addConstrs(diffs_pos[i] == gurobipy.max_(diffs[i], 0.0) for i in range(self.num_com))
        model.addConstrs(diffs_neg[i] == gurobipy.min_(diffs[i], 0.0) for i in range(self.num_com))
        """model.addMConstrs(A=self.linear, x=weights.values(), sense=GRB.LESS_EQUAL, b=np.ones(len(self.linear)))"""

        # objective
        model.setObjective(self.grb_obje(weights), sense=GRB.MINIMIZE)

        # solve
        model.optimize()
        status = model.Status
        success = True if status == GRB.OPTIMAL else False
        solution = model.objVal if success else None
        weights = np.array([weight.x for idx, weight in weights.items()]) if success else None

        return success, solution, weights


# standardize timeout
# standardize tolerance

# Notes
# Other tests: SCIP modelling tool and NEOS servers
# correlation of solving time vs. turnover and vs. the solving time in the same class
