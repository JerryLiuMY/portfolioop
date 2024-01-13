import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from global_settings import DATA_FOLDER
from matplotlib.lines import Line2D
from tools.utils import ms_pal
sns.set()


def back_eff(exp_num):
    exp_path = os.path.join(DATA_FOLDER, 'backtest', f'experiment_{exp_num}')
    fig, axes = plt.subplots(4, 1, figsize=(17, 11.5))

    def ax_bar(tran_type, ax):
        eff_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_eff.pkl'))
        eff_us = eff_df.unstack().reset_index().rename(columns={'level_0': 'date', 'level_1': 'ms', 0: 'lapse'})
        eff_us['lapse'] = np.log(1+eff_us['lapse'])
        ylim = 1.5*np.log(np.nanmax(eff_df.values) + 1)
        sub = sns.boxplot(x='ms', y='lapse', data=eff_us, palette=ms_pal, ax=ax)

        # text box
        eff_ave = np.array(eff_df.mean(axis=1))
        eff_std = np.array(eff_df.std(axis=1))
        eff_nan = eff_df.isna().sum(axis=1)
        for i, (a, s, n) in enumerate(zip(eff_ave, eff_std, eff_nan)):
            text1 = f'$t={format(round(a, 2), ".2f")} \pm {format(round(s, 2), ".2f")}$'
            text2, color = f'failure cases: ${n}$', 'red' if n != 0 else 'green'
            prop1 = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.6)
            prop2 = dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.6)
            ax.text(i, 0.90*ylim, text1, fontsize=9.5, ha='center', bbox=prop1)
            ax.text(i, 0.78*ylim, text2, fontsize=9.5, ha='center', color=color, alpha=.75, bbox=prop2)

        # legend, title, label
        sorted_ms = list(eff_df.mean(axis=1).sort_values().index)
        ax.legend(
            handles=[Line2D([0], [0], color=ms_pal[_ms_], label=_ms_) for _ms_ in sorted_ms],
            loc='upper right', prop={'size': 9.5}
        )

        ax.set_title(f'transaction_cost: {tran_type}', fontsize=14)
        ax.set_ylabel('log(1+time)', fontsize=11.5)
        ax.set_ylim(top=ylim)
        ax.set_xlim(right=8.6)
        sub.set(xlabel=None)

    with mpl.rc_context(rc={'text.usetex': True}):
        ax_bar('none', axes[0]), ax_bar('line', axes[1])
        ax_bar('quad', axes[2]), ax_bar('gene', axes[3])

    plt.tight_layout()
    plt_path = os.path.join(DATA_FOLDER, 'plots')
    fig.savefig(os.path.join(exp_path, 'back_eff.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plt_path, 'back_eff.pdf'), bbox_inches='tight')

