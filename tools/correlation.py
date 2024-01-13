# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from global_settings import DATA_FOLDER
# sns.set()
#
#
# def plot_series(exp_num, _ms_):
#     exp_path = os.path.join(DATA_FOLDER, 'backtest', f'experiment_{exp_num}')
#     fig, axes = plt.subplots(4, 1, figsize=(15, 6))
#
#     def ax_bar(tran_type, ax):
#         eff_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_eff.pkl'))
#         ax.stem(np.array(eff_df.loc[_ms_, :]), markerfmt=' ', basefmt=' ')
#         ax.scatter(np.array(eff_df.columns), np.array(eff_df.loc[_ms_, :]), marker='.')
#
#     ax_bar('none', axes[0]), ax_bar('line', axes[1])
#     ax_bar('quad', axes[2]), ax_bar('gene', axes[3])
#     plt.tight_layout()
#
#     eff_df = pd.read_pickle('/Users/mingyu/Desktop/opt_data/backtest/experiment_1/quad_eff.pkl')
#     wgt_df = pd.read_pickle('/Users/mingyu/Desktop/opt_data/backtest/experiment_1/quad_wgt.pkl')
#     wgt_arr = wgt_df.iloc[7, :]
#     eff_arr = eff_df.iloc[7, :]
#     eff = eff_arr[1:]
#     tov = np.array([sum(abs(wgt_arr[i] - wgt_arr[i - 1])) for i in range(1, len(wgt_arr))])
#     corr_coef = np.corrcoef(eff, tov)[1, 0]
#
