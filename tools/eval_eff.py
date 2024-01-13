import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib import pyplot as plt
from global_settings import DATA_FOLDER
from tools.utils import ms_pal
sns.set()


# evaluation - efficiency test
def eval_eff(exp_num):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    def ax_bar(tran_type, ax):
        ax.set_title(f'transaction_cost: {tran_type}', fontsize=13)
        ax.set_xlabel('Number of Assets', fontsize=13)
        ax.set_ylabel('log(1+time)', fontsize=13)
        ax.legend(loc='upper left')

    ax_foo = _ax_int if not exp_num == 'all' else _ax_all
    ax_foo('none', axes[0, 0], exp_num), ax_foo('line', axes[0, 1], exp_num)
    ax_foo('quad', axes[1, 0], exp_num), ax_foo('gene', axes[1, 1], exp_num)
    ax_bar('none', axes[0, 0]), ax_bar('line', axes[0, 1])
    ax_bar('quad', axes[1, 0]), ax_bar('gene', axes[1, 1])
    plt.tight_layout()

    if not exp_num == 'all':
        exp_path = os.path.join(DATA_FOLDER, 'evaluation', f'experiment_{exp_num}')
        fig.savefig(os.path.join(exp_path, f'eval_eff.pdf'), bbox_inches='tight')
    else:
        plt_path = os.path.join(DATA_FOLDER, 'plots')
        fig.savefig(os.path.join(plt_path, 'eval_eff.pdf'), bbox_inches='tight')


def _ax_int(tran_type, ax, exp_num):
    exp_path = os.path.join(DATA_FOLDER, 'evaluation', f'experiment_{exp_num}')
    eff_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_eff.pkl'))
    eff_df = eff_df.sort_values(by=eff_df.columns[-1])

    for _ms_, eff_row in eff_df.iterrows():
        ax.plot(
            np.array(eff_df.columns),
            np.log(np.array(eff_row) + 1),
            label=_ms_, color=ms_pal[_ms_]
        )


def _ax_all(tran_type, ax, *args):
    eff_dfs = list()
    exp_paths = sorted(glob(os.path.join(DATA_FOLDER, 'evaluation', 'experiment_*')))
    for exp_path in exp_paths:
        eff_df_ = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_eff.pkl'))
        eff_dfs.append(eff_df_)

    eff_df = sum(eff_dfs) / len(exp_paths)
    std_mt = np.dstack([eff_df.values for eff_df in eff_dfs]).std(axis=2, ddof=1)
    std_df = pd.DataFrame(std_mt, index=eff_df.index, columns=eff_df.columns)
    eff_df = eff_df.sort_values(by=eff_df.columns[-1])
    std_df = std_df.reindex(eff_df.index)

    for _ms_, eff_row in eff_df.iterrows():
        std_row = std_df.loc[_ms_, :]
        ax.plot(
            np.array(eff_df.columns),
            np.log(np.array(eff_row) + 1),
            label=_ms_, color=ms_pal[_ms_]
        )

        ax.fill_between(
            np.array(eff_df.columns),
            (np.log(np.array(eff_row) - np.array(std_row) + 1)),
            (np.log(np.array(eff_row) + np.array(std_row) + 1)),
            color=ms_pal[_ms_], alpha=0.1
        )
