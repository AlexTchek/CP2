import pandas as pd
import numpy as np
import talib as ta
from datetime import datetime
from tqdm import tqdm
from finrl import config
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

class FeaturesAdder():
    def __init__(self, lookback = 22):
        self.rsi_period = 12
        self.rsi_high_low = 6
        self.indicators = ['HL_rsi12_up', 'HL_rsi12_down', 'rsi12', 'obv']#"rsi_12", "cci_12", "dx_12"] #config.INDICATORS
        self.cov_xtra_names = ['BZ', 'GD', 'IMOEX']#, 'USD']
        self.lookback = lookback

    def Process(self, df):
        df = self.AddIndicators(df)
        df.index = df.date.factorize()[0]
        df = self.AddCovariations(df)
        df.index = list(range(len(df)))
        return df
    
    def HL(self, row, n):
        hi = pd.Series(row)
        lo = pd.Series(row)
        uc = hi.rolling(n, min_periods=n).max()
        lc = lo.rolling(n, min_periods=n).min()
        return uc, lc

    def AddIndicators(self, df):
        stock = df.copy()
        unique_ticker = stock.tic.unique()

        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                tic_data = stock[stock['tic']==unique_ticker[i]].sort_values(by=["date"])
                _high  = tic_data['high'].to_numpy()
                _low   = tic_data['low'].to_numpy()
                _close = tic_data['close'].to_numpy()
                _volume = tic_data['volume'].to_numpy()
                rsi12 = ta.RSI(_close, timeperiod = self.rsi_period)
                hl_rsi12 = self.HL(rsi12, self.rsi_high_low)
                
                obv = ta.OBV(_close, _volume) / 1e7
                #mom = ta.MOM(_close, timeperiod=12)
                #temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                u, l = hl_rsi12
                temp_indicator = pd.DataFrame({'HL_rsi12_up' : u, 'HL_rsi12_down' : l})
                #temp_indicator["MOM"] = mom
                temp_indicator["rsi12"] = rsi12
                temp_indicator["obv"] = obv
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["date"] = stock[stock.tic == unique_ticker[i]][
                    "date"
                ].to_list()

                indicator_df = pd.concat(
                    [indicator_df, temp_indicator], axis=0, ignore_index=True
                )
            except Exception as e:
                print(e)

        stock = stock.merge(
            indicator_df, on=["tic", "date"], how="left"
        )
        stock = stock.sort_values(by=["date", "tic"])
        df = stock

        return df

    
    def AddCovariations(self, df):
        cov_list = []
        cov_xtra = []
        return_list = []
        for i in tqdm(range(self.lookback, len(df.index.unique()))):
            data_lookback = df.loc[i-self.lookback:i,:]
            price_lookback = data_lookback.pivot_table(index = 'date', columns = 'tic', values = 'close')
            #price_lookback['MMM'] = price_lookback['MM']
            #cov_names = list(set(price_lookback.columns)-set(self.cov_xtra_names))
            return_lookback = price_lookback.dropna()
            return_list.append(return_lookback)

            #covs = return_lookback.cov().values 
            cov_list.append([])

            state = np.array([])
            for index in self.cov_xtra_names:
                s = pd.Series({symbol: return_lookback[index].corr(return_lookback[symbol]) 
                            for symbol in return_lookback 
                            if symbol not in self.cov_xtra_names and symbol != 'date'})
                state = np.append(state, s.to_numpy())

            cov_xtra.append(state)
            

        df_cov = pd.DataFrame({'date':df.date.unique()[self.lookback:], 'cov_list':cov_list, 'cov_xtra':cov_xtra, 'return_list':return_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)

        for xtra in self.cov_xtra_names:
            df = df.drop(df[df.tic == xtra].index)
        
        return df