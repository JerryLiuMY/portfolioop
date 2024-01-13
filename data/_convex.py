# from tran_obje.sci_obje import get_sha
# import matplotlib.pyplot as plt
# import numpy as np
#
# weight1 = np.random.uniform(-20, 10, size=1000).reshape(-1, 1)
# weight2 = (1 - weight1).reshape(-1, 1)
# weight3 = np.zeros(shape=1000).reshape(-1, 1)
# weight4 = np.zeros(shape=1000).reshape(-1, 1)
# weight5 = np.zeros(shape=1000).reshape(-1, 1)
# samples = np.hstack([weight1, weight2, weight3, weight4, weight5])
#
# sharpes = np.array([])
# for weights in samples:
#     sharpe = get_sha(weights=weights)
#     sharpes = np.append(sharpes, sharpe)
# plt.plot(weight1.reshape(-1), sharpes, 'o')
