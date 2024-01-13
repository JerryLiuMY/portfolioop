import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from global_settings import DATA_FOLDER
from tools.utils import ms_pal
import seaborn as sns
from glob import glob
sns.set()


# evaluation - return accuracy test
def eval_ret(exp_num):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    def ax_bar(tran_type, ax):
        ax.set_title(f'transaction_cost: {tran_type}', fontsize=13)
        ax.set_xlabel('Number of Assets', fontsize=13)
        ax.set_ylabel('log(1 + ret)', fontsize=13)
        ax.add_artist(ax.legend(loc='lower right'))

    ax_foo = _ax_int if not exp_num == 'all' else _ax_all
    ax_foo('none', axes[0, 0], exp_num), ax_foo('line', axes[0, 1], exp_num)
    ax_foo('quad', axes[1, 0], exp_num), ax_foo('gene', axes[1, 1], exp_num)
    ax_bar('none', axes[0, 0]), ax_bar('line', axes[0, 1])
    ax_bar('quad', axes[1, 0]), ax_bar('gene', axes[1, 1])
    plt.tight_layout()

    if not exp_num == 'all':
        exp_path = os.path.join(DATA_FOLDER, 'evaluation', f'experiment_{exp_num}')
        fig.savefig(os.path.join(exp_path, 'eval_ret.pdf'), bbox_inches='tight')
    else:
        plt_path = os.path.join(DATA_FOLDER, 'plots')
        fig.savefig(os.path.join(plt_path, 'eval_ret.pdf'), bbox_inches='tight')


def _ax_int(tran_type, ax, exp_num):
    exp_path = os.path.join(DATA_FOLDER, 'evaluation', f'experiment_{exp_num}')
    ret_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_ret.pkl'))

    for _ms_, ret_row in ret_df.iterrows():
        ax.plot(
            np.array(ret_df.columns),
            np.array(ret_row),
            label=_ms_, color=ms_pal[_ms_]
        )


def _ax_all(tran_type, ax, exp_num):
    ret_dfs = list()
    exp_paths = sorted(glob(os.path.join(DATA_FOLDER, 'evaluation', 'experiment_*')))
    for exp_path in exp_paths:
        ret_df_ = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_ret.pkl'))
        ret_dfs.append(ret_df_)

    ret_df = sum(ret_dfs) / len(exp_paths)
    std_mt = np.dstack([rer_df.values for rer_df in ret_dfs]).std(axis=2, ddof=1)
    std_df = pd.DataFrame(std_mt, index=ret_df.index, columns=ret_df.columns)

    for _ms_, ret_row in ret_df.iterrows():
        std_row = std_df.loc[_ms_, :]
        ax.plot(
            np.array(ret_df.columns),
            np.log(np.array(ret_row)),
            label=_ms_, color=ms_pal[_ms_]
        )

        ax.fill_between(
            np.array(ret_df.columns),
            (np.log(np.array(ret_row) - np.array(std_row))),
            (np.log(np.array(ret_row) + np.array(std_row))),
            color=ms_pal[_ms_], alpha=0.1
        )
