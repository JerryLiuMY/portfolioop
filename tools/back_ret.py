import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from global_settings import DATA_FOLDER
from tools.utils import ms_pal
sns.set()


# evaluation - efficiency test
def back_ret(exp_num):
    exp_path = os.path.join(DATA_FOLDER, 'backtest', f'experiment_{exp_num}')
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    def ax_bar(tran_type, ax):
        ret_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_ret.pkl')).dropna(axis=0, how='any')
        dates = np.array(ret_df.columns)
        years = [int(date.split('-')[0]) for date in ret_df.columns]

        for _ms_, ret_row in ret_df.iterrows():
            ax.plot(
                np.array(np.arange(len(dates))),
                np.log(np.cumprod(np.array(ret_row))),
                label=_ms_, color=ms_pal[_ms_]
            )

        xticks_foo(years, ret_df.columns, ax)
        ax.set_title(f'transaction_cost: {tran_type}', fontsize=13)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Cumulative log(1+ret)', fontsize=13)
        ax.legend(loc='upper left')

    def xticks_foo(years, dates, ax):
        xticks = range(0,  int(np.ceil(len(dates) / 126)) * 126 + 1, 126)
        labels_y = np.repeat(np.arange(min(years), (max(years) + 1) + 1), 2)
        labels_m = np.array(['JAN', 'JUN'] * int(len(labels_y) / 2))
        xticklabels = ['-'.join([m, str(y)]) for y, m in zip(labels_y, labels_m)][:len(xticks)]
        ax.set_xticks(xticks), ax.set_xticklabels(xticklabels)

    ax_bar('none', axes[0, 0]), ax_bar('line', axes[0, 1])
    ax_bar('quad', axes[1, 0]), ax_bar('gene', axes[1, 1])
    plt.tight_layout()

    plt_path = os.path.join(DATA_FOLDER, 'plots')
    fig.savefig(os.path.join(exp_path, 'back_ret.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plt_path, 'back_ret.pdf'), bbox_inches='tight')
