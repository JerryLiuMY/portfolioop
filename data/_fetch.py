import os
import pandas as pd
from pandas_datareader import data as web
from global_settings import DATA_FOLDER

assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
data_beg, data_end = '2008-01-01', '2019-12-31'


def save_yahoos():
    vix_df = web.DataReader(['^VIX'], data_source='yahoo', start=data_beg, end=data_end)['Adj Close']
    vix_df = vix_df.rename(columns={"^VIX": "VIX"}, errors="raise")
    prc_df = web.DataReader(assets, data_source='yahoo', start=data_beg, end=data_end)['Adj Close']
    vlm_df = web.DataReader(assets, data_source='yahoo', start=data_beg, end=data_end)['Volume']
    ret_df = prc_df.pct_change()

    vix_df.to_csv(os.path.join(DATA_FOLDER, 'vix_df.csv'))
    prc_df.to_csv(os.path.join(DATA_FOLDER, 'prc_df.csv'))
    vlm_df.to_csv(os.path.join(DATA_FOLDER, 'vlm_df.csv'))
    ret_df.to_csv(os.path.join(DATA_FOLDER, 'ret_df.csv'))


def save_caps():
    sp500 = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'))
    cap_df = pd.DataFrame(columns=assets, index=sp500['Date'])
    raw_df = pd.read_csv(os.path.join(DATA_FOLDER, 'raw.csv'))
    raw_df['date'] = raw_df['date'].astype(str).apply(lambda _: _[:4] + '-' + _[4:6] + '-' + _[6:8])
    for _ in assets:
        raw_df_ = raw_df[raw_df['TICKER'] == _].sort_values(['date']).set_index('date')
        cap_df[_] = raw_df_['PRC'] * raw_df_['SHROUT']

    cap_df.to_csv(os.path.join(DATA_FOLDER, 'cap_df.csv'))