from global_settings import DATA_FOLDER
from tqdm import tqdm_notebook
import pandas as pd
import os


def save_raw():
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    beta = pd.read_csv(os.path.join(DATA_FOLDER, 'beta.csv'))
    raw_dir = os.path.join(DATA_FOLDER, 'raw_beta')

    for group in tqdm_notebook(groups):
        raw_df = beta[beta['PERMNO'].apply(lambda _: str(_)[:2] == group)]
        raw_df.reset_index(drop=True, inplace=True)
        raw_df.to_csv(os.path.join(raw_dir, f'{group}.csv'))


def save_int():
    # define inputs
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    raw_dir = os.path.join(DATA_FOLDER, 'raw_beta')
    int_dir = os.path.join(DATA_FOLDER, 'int_beta')

    # filter out companies without enough data
    for group in tqdm_notebook(groups):
        raw_df = pd.read_csv(os.path.join(raw_dir, f'{group}.csv'), index_col=0)
        subset = ['DATE', 'TICKER', 'b_mkt', 'alpha', 'RET', 'exret', 'ivol', 'tvol']
        raw_df = raw_df.dropna(axis=0, how='any', subset=subset)
        raw_df['RET'] = raw_df['RET'].apply(lambda _: float(_[:-1])/100)
        raw_df['exret'] = raw_df['exret'].apply(lambda _: float(_[:-1])/100)
        raw_df['ivol'] = raw_df['ivol'].apply(lambda _: float(_[:-1])/100)
        raw_df['tvol'] = raw_df['tvol'].apply(lambda _: float(_[:-1])/100)
        raw_df['TICKER'] = raw_df['TICKER'].astype(str)

        int_df = pd.DataFrame(columns=raw_df.columns)
        for ticker in tqdm_notebook(sorted(set(raw_df['TICKER']))):
            raw_df_ = raw_df[raw_df['TICKER'] == ticker]
            if len(raw_df_) == 3021:
                int_df = pd.concat([int_df, raw_df_], axis=0)

        int_df.reset_index(drop=True, inplace=True)
        int_df.to_csv(os.path.join(int_dir, f'{group}.csv'))


def save_data():
    # define inputs
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    sp500 = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'))
    int_dir = os.path.join(DATA_FOLDER, 'int_beta')
    beta_dir = os.path.join(DATA_FOLDER, 'beta')
    ivol_dir = os.path.join(DATA_FOLDER, 'ivol')

    for group in tqdm_notebook(groups):
        int_df = pd.read_csv(os.path.join(int_dir, f'{group}.csv'), index_col=0)
        int_df['DATE'] = int_df['DATE'].apply(lambda _: '-'.join([str(_)[:4], str(_)[4:6], str(_)[6:]]))
        beta_df = pd.DataFrame(index=sp500['Date'], columns=sorted(set(int_df['TICKER'])))
        ivol_df = pd.DataFrame(index=sp500['Date'], columns=sorted(set(int_df['TICKER'])))

        for ticker in tqdm_notebook(sorted(set(int_df['TICKER']))):
            int_df_ = int_df[int_df['TICKER'] == ticker]
            for index, row in int_df_.iterrows():
                beta_df.loc[row['DATE'], ticker] = row['b_mkt']
                ivol_df.loc[row['DATE'], ticker] = row['ivol']

        beta_df.to_csv(os.path.join(beta_dir, f'{group}.csv'))
        ivol_df.to_csv(os.path.join(ivol_dir, f'{group}.csv'))


def combine_data():
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    sp500 = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'))
    data_dir = os.path.join(DATA_FOLDER, 'data')
    beta_dir = os.path.join(DATA_FOLDER, 'beta')
    ivol_dir = os.path.join(DATA_FOLDER, 'ivol')
    beta_full = pd.DataFrame(index=sp500['Date'])
    ivol_full = pd.DataFrame(index=sp500['Date'])

    for group in tqdm_notebook(groups):
        beta_df = pd.read_csv(os.path.join(beta_dir, f'{group}.csv'), index_col=0)
        ivol_df = pd.read_csv(os.path.join(ivol_dir, f'{group}.csv'), index_col=0)
        beta_full = pd.concat([beta_full, beta_df], axis=1)
        ivol_full = pd.concat([ivol_full, ivol_df], axis=1)

    beta_full.to_csv(os.path.join(data_dir, 'beta.csv'))
    ivol_full.to_csv(os.path.join(data_dir, 'ivol.csv'))
