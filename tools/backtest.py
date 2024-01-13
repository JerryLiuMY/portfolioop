import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from data.data_loader import load_summary
from global_settings import DATA_FOLDER
from tools.utils import tt, tt_pal
import seaborn as sns
sns.set()


def backtest(exp_num, _ms_):
    exp_path = os.path.join(DATA_FOLDER, 'backtest', f'experiment_{exp_num}')
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(10, 1)
    ax1 = fig.add_subplot(gs[0:6, :])
    ax2a, ax2b = fig.add_subplot(gs[6:7, :]), fig.add_subplot(gs[7:8, :])
    ax2c, ax2d = fig.add_subplot(gs[8:9, :]), fig.add_subplot(gs[9:10, :])

    sp500 = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'), index_col='Date')
    sp500['Ret'] = 1 + (sp500['Close'] - sp500.shift(1)['Close']) / sp500['Close']
    sp500 = sp500[sp500.index.to_series().apply(lambda x: 2015 <= int(x[:4]) < 2017)]
    FF = pd.read_csv(os.path.join(DATA_FOLDER, 'FF.csv'), index_col=0)
    FF = FF[FF.index.to_series().apply(lambda x: 2015 <= int(x[:4]) < 2017)]

    def ax1_bar(tran_type, ax):
        # num_columns not exactly divisible by 252
        ret_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_ret.pkl'))
        wgt_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_wgt.pkl'))
        ret_arr, wgt_arr = np.array(ret_df.loc[_ms_, :]), np.array(wgt_df.loc[_ms_, :])

        # calculate daily return
        daily = pow(np.cumprod(ret_arr)[-1], 1/len(ret_arr)) - 1

        # calculate sharpe
        dates = np.array(ret_df.columns)
        years = [int(date.split('-')[0]) for date in dates]
        exc_arr = ret_arr - np.array(FF.loc[dates, :]['RF']) - 1
        sharpe = exc_arr.mean() / exc_arr.std() * np.sqrt(252)

        # calculate turnover
        tov_arr = np.array(list())
        for idx, date in enumerate(list(ret_df.columns)[:-1]):
            wgt_vec = wgt_arr[idx]
            ret_vec = 1 + load_summary(date, len(wgt_vec))[0]
            tov_arr = np.append(tov_arr, sum(np.abs(wgt_arr[idx+1] - wgt_vec * ret_vec)))
        tov = sum(tov_arr) / (2*(len(ret_df.columns) - 1))

        ax.plot(np.arange(len(dates)), np.log(np.cumprod(ret_arr)), color=tt_pal[tran_type], label=tran_type)
        xticks_foo(years, ret_df.columns, ax), ax.set_xlabel('Date'), ax.set_ylabel('Cumulative log(1+ret)')
        stats = f'{tran_type} - daily: {round(daily*100, 2)}%, sharpe: {round(sharpe, 2)}, turnover: {round(tov, 2)}'

        return stats

    def ax2_bar(tran_type, ax):
        ret_df = pd.read_pickle(os.path.join(exp_path, f'{tran_type}_ret.pkl'))
        ret_arr = np.array(ret_df.loc[_ms_, :])
        years = [int(date.split('-')[0]) for date in ret_df.columns]

        ax.stem(np.arange(len(ret_df.columns)), ret_arr-1, linefmt=tt_pal[tran_type], markerfmt=' ', basefmt=' ')
        ax.scatter(np.arange(len(ret_df.columns)), ret_arr-1, color=tt_pal[tran_type], marker='.')
        xticks_foo(years, ret_df.columns, ax), ax.set_ylabel('ret')

    def xticks_foo(years, dates, ax):
        xticks = range(0,  int(np.ceil(len(dates) / 126)) * 126 + 1, 126)
        labels_y = np.repeat(np.arange(min(years), (max(years) + 1) + 1), 2)
        labels_m = np.array(['JAN', 'JUN'] * int(len(labels_y) / 2))
        xticklabels = ['-'.join([m, str(y)]) for y, m in zip(labels_y, labels_m)][:len(xticks)]
        ax.set_xticks(xticks), ax.set_xticklabels(xticklabels)

    leg_none, leg_line = ax1_bar('none', ax1), ax1_bar('line', ax1)
    leg_quad, leg_gene = ax1_bar('quad', ax1), ax1_bar('gene', ax1)
    ax1.plot(np.arange(len(sp500)), np.log(np.cumprod(np.array(sp500['Ret']))), color='y', label='sp500')
    ax1.add_artist(ax1.legend(loc='upper left', prop={'size': 11}))
    leg_dict = {'none': leg_none, 'line': leg_line, 'quad': leg_quad, 'gene': leg_gene}
    ax1.legend(
        handles=[Line2D([0], [0], label=leg_dict[_tt_], color=tt_pal[_tt_]) for _tt_ in tt],
        loc=(0.65, 0.2), prop={'size': 11}
    )
    ax2_bar('none', ax2a), ax2_bar('line', ax2b), ax2_bar('quad', ax2c), ax2_bar('gene', ax2d)
    fig.tight_layout()

    plt_path = os.path.join(DATA_FOLDER, 'plots')
    fig.savefig(os.path.join(exp_path, 'backtest.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(plt_path, 'backtest.pdf'), bbox_inches='tight')
