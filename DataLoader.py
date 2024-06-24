import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader():
    def __init__(self):
        self.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        self.xtra = ['MM']

    def LoadData(self):
        df = self.LoadMainData()
        df2 = self.LoadXtra(self.xtra)

        cc = pd.concat([df, df2], ignore_index=True)
        unique_ticks = len(cc['tic'].unique())
        cnt0 = cc.groupby(['date']).count()
        date_list = cnt0[cnt0["open"] < unique_ticks].index
        cc = cc[~cc['date'].dt.normalize().isin(date_list)]
        cc = cc.sort_values(by=["date", "tic"], ignore_index=True)
        cc.index = cc.date.factorize()[0]

        return cc

    def LoadMainData(self):
        return pd.read_csv("Data\\Prices.csv", sep=',', decimal='.', date_format='%Y-%m-%d', parse_dates=["date"])
    
    def LoadXtra(self, tickers):
        df = pd.DataFrame(columns = self.columns)
        for ticker in tickers:
            dft = pd.read_csv(f"Data\\{ticker}.txt", sep=';', decimal='.', date_format='%Y%m%d', parse_dates=["<DATE>"])
            dft['date']  = dft["<DATE>"]
            dft['open']  = dft["<CLOSE>"]
            dft['high']  = dft["<CLOSE>"]
            dft['low']   = dft["<CLOSE>"]
            dft['close'] = dft["<CLOSE>"]
            dft['volume']= dft["<VOL>"]
            dft['tic']   = ticker
            df = pd.concat([df, dft[self.columns]], ignore_index=True)
        return df