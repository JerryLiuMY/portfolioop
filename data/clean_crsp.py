from global_settings import DATA_FOLDER
from tqdm import tqdm_notebook
import pandas as pd
import pickle
import os


def save_groups():
    crsp = pd.read_csv(os.path.join(DATA_FOLDER, 'crsp.csv'))
    groups = sorted(set([str(_)[:2] for _ in sorted(set(crsp['PERMNO']))]))
    with open(os.path.join(DATA_FOLDER, 'groups.pkl'), 'wb') as handle:
        pickle.dump(groups, handle)


def save_raw():
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    crsp = pd.read_csv(os.path.join(DATA_FOLDER, 'crsp.csv'))
    raw_dir = os.path.join(DATA_FOLDER, 'raw_crsp')

    for group in tqdm_notebook(groups):
        raw_df = crsp[crsp['PERMNO'].apply(lambda _: str(_)[:2] == group)]
        raw_df.reset_index(drop=True, inplace=True)
        raw_df.to_csv(os.path.join(raw_dir, f'{group}.csv'))


def save_int():
    # define inputs
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    raw_dir = os.path.join(DATA_FOLDER, 'raw_crsp')
    int_dir = os.path.join(DATA_FOLDER, 'int_crsp')

    # filter out companies without enough data
    for group in tqdm_notebook(groups):
        raw_df = pd.read_csv(os.path.join(raw_dir, f'{group}.csv'), index_col=0)
        raw_df = raw_df.dropna(axis=0, how='any', subset=['date', 'TICKER', 'PRC', 'VOL', 'RET', 'SHROUT', 'CFACPR'])
        raw_df['RET'] = raw_df['RET'].astype(str)
        raw_df = raw_df[raw_df['RET'] != 'C']
        raw_df['RET'] = raw_df['RET'].astype(float)
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
    int_dir = os.path.join(DATA_FOLDER, 'int_crsp')
    prc_dir = os.path.join(DATA_FOLDER, 'prc')
    vlm_dir = os.path.join(DATA_FOLDER, 'vlm')
    cap_dir = os.path.join(DATA_FOLDER, 'cap')

    # construct output dataframe
    for group in tqdm_notebook(groups):
        int_df = pd.read_csv(os.path.join(int_dir, f'{group}.csv'), index_col=0)
        int_df['date'] = int_df['date'].apply(lambda _: '-'.join([str(_)[:4], str(_)[4:6], str(_)[6:]]))
        prc_df = pd.DataFrame(index=sp500['Date'], columns=sorted(set(int_df['TICKER'])))
        vlm_df = pd.DataFrame(index=sp500['Date'], columns=sorted(set(int_df['TICKER'])))
        cap_df = pd.DataFrame(index=sp500['Date'], columns=sorted(set(int_df['TICKER'])))

        for ticker in tqdm_notebook(sorted(set(int_df['TICKER']))):
            int_df_ = int_df[int_df['TICKER'] == ticker]
            for index, row in int_df_.iterrows():
                prc_df.loc[row['date'], ticker] = abs(row['PRC'])
                vlm_df.loc[row['date'], ticker] = row['VOL']
                cap_df.loc[row['date'], ticker] = abs(row['PRC']) * row['SHROUT']

        prc_df.to_csv(os.path.join(prc_dir, f'{group}.csv'))
        vlm_df.to_csv(os.path.join(vlm_dir, f'{group}.csv'))
        cap_df.to_csv(os.path.join(cap_dir, f'{group}.csv'))


def combine_data():
    groups = pd.read_pickle(os.path.join(DATA_FOLDER, 'groups.pkl'))
    sp500 = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'))
    prc_dir = os.path.join(DATA_FOLDER, 'prc')
    vlm_dir = os.path.join(DATA_FOLDER, 'vlm')
    cap_dir = os.path.join(DATA_FOLDER, 'cap')
    data_dir = os.path.join(DATA_FOLDER, 'data')

    prc_full = pd.DataFrame(index=sp500['Date'])
    vlm_full = pd.DataFrame(index=sp500['Date'])
    cap_full = pd.DataFrame(index=sp500['Date'])

    for group in tqdm_notebook(groups):
        prc_df = pd.read_csv(os.path.join(prc_dir, f'{group}.csv'), index_col=0)
        vlm_df = pd.read_csv(os.path.join(vlm_dir, f'{group}.csv'), index_col=0)
        cap_df = pd.read_csv(os.path.join(cap_dir, f'{group}.csv'), index_col=0)
        prc_full = pd.concat([prc_full, prc_df], axis=1)
        vlm_full = pd.concat([vlm_full, vlm_df], axis=1)
        cap_full = pd.concat([cap_full, cap_df], axis=1)

    prc_full.to_csv(os.path.join(data_dir, 'prc.csv'))
    vlm_full.to_csv(os.path.join(data_dir, 'vlm.csv'))
    cap_full.to_csv(os.path.join(data_dir, 'cap.csv'))
