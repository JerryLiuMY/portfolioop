import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm_notebook
from global_settings import DATA_FOLDER, VAL_YEAR


def save_columns():
    # load df
    data_dir = os.path.join(DATA_FOLDER, 'data')
    prc_df = pd.read_csv(os.path.join(data_dir, 'prc.csv'), index_col=0)
    cap_df = pd.read_csv(os.path.join(data_dir, 'cap.csv'), index_col=0)
    vlm_df = pd.read_csv(os.path.join(data_dir, 'vlm.csv'), index_col=0)
    beta_df = pd.read_csv(os.path.join(data_dir, 'beta.csv'), index_col=0)
    ivol_df = pd.read_csv(os.path.join(data_dir, 'ivol.csv'), index_col=0)

    # intersecting tickers
    columns = set(prc_df.columns) & set(cap_df.columns) & set(vlm_df.columns)
    columns = columns & set(beta_df.columns) & set(ivol_df.columns)
    columns = sorted(list(columns))

    with open(os.path.join(data_dir, 'columns.pkl'), 'wb') as handle:
        pickle.dump(columns, handle)


def save_param(dates):
    data_dir = os.path.join(DATA_FOLDER, 'data')
    param_dir = os.path.join(DATA_FOLDER, 'param')
    columns = pd.read_pickle(os.path.join(data_dir, 'columns.pkl'))

    print(f'{datetime.now()} Loading data...')
    prc_df = pd.read_csv(os.path.join(data_dir, 'prc.csv'), index_col=0)[columns]
    cap_df = pd.read_csv(os.path.join(data_dir, 'cap.csv'), index_col=0)[columns]
    vlm_df = pd.read_csv(os.path.join(data_dir, 'vlm.csv'), index_col=0)[columns]
    std_df = pd.read_csv(os.path.join(data_dir, 'ivol.csv'), index_col=0)[columns]
    vix_df = pd.read_csv(os.path.join(data_dir, 'vix.csv'), index_col=0)

    for date in tqdm_notebook(dates):
        prc_df_ = prc_df.loc[:date, :].iloc[-int(252 * VAL_YEAR + 1): -1, :]
        cap_df_ = cap_df.loc[:date, :].iloc[-int(252 * VAL_YEAR + 1): -1, :]
        vlm_df_ = vlm_df.loc[:date, :].iloc[-int(252 * VAL_YEAR + 1): -1, :]
        std_df_ = std_df.loc[:date, :].iloc[-2, :]

        year = int(date.split('-')[0]) if int(date.split('-')[1]) > 6 else int(date.split('-')[0]) - 1
        clos, vlms = prc_df_.mean(), vlm_df_.mean()
        caps, stds = cap_df_.mean(), std_df_
        vix = vix_df.loc[date, 'Adj Close']

        param = {'year': year, 'clos': clos, 'vlms': vlms, 'caps': caps, 'stds': stds, 'vix': vix}
        with open(os.path.join(param_dir, f'{date}.pkl'), 'wb') as handle:
            pickle.dump(param, handle)


def save_summary(dates):
    data_dir = os.path.join(DATA_FOLDER, 'data')
    summary_dir = os.path.join(DATA_FOLDER, 'summary')
    columns = pd.read_pickle(os.path.join(data_dir, 'columns.pkl'))

    print(f'{datetime.now()} Loading data...')
    beta_df = pd.read_csv(os.path.join(data_dir, 'beta.csv'), index_col=0)[columns]
    ivol_df = pd.read_csv(os.path.join(data_dir, 'ivol.csv'), index_col=0)[columns]
    sp500 = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'), index_col=0)
    syn_df = pd.read_csv(os.path.join(DATA_FOLDER, 'synthesis.csv'), index_col=0)
    FF = pd.read_csv(os.path.join(DATA_FOLDER, 'FF.csv'), index_col=0)
    sp500['RM'] = sp500['Adj Close'].pct_change(fill_method=None)
    syn_df['RM'] = syn_df['Adj Close'].pct_change(fill_method=None)

    for date in tqdm_notebook(dates):
        sp500_ = sp500.loc[:date, :].iloc[-int(252 * VAL_YEAR + 1): -1, :]
        FF_ = FF.loc[:date, :].iloc[-int(252 * VAL_YEAR + 1): -1, :]

        if (len(sp500_) == 252 * VAL_YEAR) and (len(FF_) == 252 * VAL_YEAR):
            betas, ivols = beta_df.loc[date, :], ivol_df.loc[date, :]
            rp, rf = syn_df.loc[date, 'RM'] - FF.loc[date, 'RF'], FF.loc[date, 'RF']
            rp_var, rf_var = np.var(np.array(sp500_['RM']) - np.array(FF_['RF'])), np.var(np.array(FF_['RF']))
        else:
            raise AssertionError('Invalid market return and risk free rate')

        if (not np.isnan(rp)) and (not np.isnan(rf)) and (not np.isnan(rp_var)) and (not np.isnan(rf_var)):
            summary = {'betas': betas, 'ivols': ivols, 'rp': rp, 'rf': rf, 'rp_var': rp_var, 'rf_var': rf_var}
            with open(os.path.join(summary_dir, f'{date}.pkl'), 'wb') as handle:
                pickle.dump(summary, handle)
        else:
            raise AssertionError('np.nan value in risk premium and risk free rate')
