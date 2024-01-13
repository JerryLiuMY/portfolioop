line_param = {
    "buy_b": 0.001,
    "sel_b": 0.002,
}

quad_param = {
    "buy_k": 0.002,
    "sel_k": 0.004,
}

gene_param = {
    "p0": 0,  # bias value
    "p1": 0.23,  # tran_obje 1 -- unobservable
    "p2": -0.01,  # time trend
    "p3": -0.62,  # capitalization
    "p4": -0.13,  # trading size 1
    "p5": 8.89,  # trading size 2
    "p6": 0.28,  # volatility of stock
    "p7": 0.015,  # volatility of market
    "p8": 0.04,  # tran_obje 2 -- unobservable
}

tran_dict = {
    'none': (),
    'line': line_param,
    'quad': quad_param,
    'gene': gene_param
}

expo = 10**6
