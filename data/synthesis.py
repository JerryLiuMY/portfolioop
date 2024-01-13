from global_settings import DATA_FOLDER
import pandas as pd
import numpy as np
import os
np.random.seed = 10


def synthesize(rho=0.2, samples=20):
    spy_df = pd.read_csv(os.path.join(DATA_FOLDER, 'sp500.csv'))
    syn_df = pd.DataFrame(index=spy_df['Date'], columns=['Adj Close'])

    chg = np.array(spy_df['Adj Close'].diff())
    std = np.array(spy_df['Adj Close'].diff().rolling(center=True, window=63).std())
    ave = np.array(spy_df['Adj Close'].diff().rolling(center=True, window=63).mean())

    foos = []
    for i in range(samples):
        foo = np.array([])
        for c2, m2, s2 in zip(chg, ave, std):
            m1, s1 = m2, s2
            if np.isnan(m2) or np.isnan(s2):
                c1 = np.nan
            else:
                m = m1 + np.sqrt(rho) * (c2 - m2)
                s = s1 - rho * s1
                c1 = float(np.random.normal(m, s, 1))
            foo = np.append(foo, c1)
        foos.append(foo)

    foo = np.mean(foos, axis=0)
    offset = int(np.argwhere([not f for f in np.isnan(foo)])[0])
    cum, prc = np.nancumsum(foo), np.array([np.nan] * len(spy_df))

    for i, f in enumerate(foo):
        if not np.isnan(f):
            prc[i] = cum[i] + spy_df.loc[offset, 'Adj Close']

    syn_df['Adj Close'] = prc
    syn_df.to_csv(os.path.join(DATA_FOLDER, 'synthesis.csv'))

