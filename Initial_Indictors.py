import pandas as pd
import numpy as np


class Initial:

    def CreateData(self,path_data):
        df = pd.read_csv(path_data)
        Date,Open,High,Low,Close,AdjClose,Volume = df.columns

        self.Date = Date
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Close = Close
        self.Volume = Volume
        df[Date ]=pd.to_datetime(df[Date], format='%Y-%m-%d')
        df.sort_values(by=Date,inplace=True)
        df.reset_index(drop=True,inplace=True)
        #df = df[(df[DateColumn]>'2011-03-21') & (df[DateColumn]<'2021-08-23')]
        df.reset_index(drop=True,inplace=True)

        return df

    ## ------------------- Add MA & MACD -------------------------
    def MA_MACD(df, n):

        df['SMA_C'] = df['Close'].rolling(window=n).mean()
        weights = np.arange(1, n + 1)
        df['WMA_C'] = df['Close'].rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        df['EMA_C'] = df['Close'].ewm(span=n).mean()
        ##
        df['SMA_O'] = df['<OPEN>'].rolling(window=n).mean()
        df['WMA_O'] = df['<OPEN>'].rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        df['EMA_O'] = df['<OPEN>'].ewm(span=n).mean()
        ###
        df['SMA_H'] = df['<HIGH>'].rolling(window=n).mean()
        df['WMA_H'] = df['<HIGH>'].rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        df['EMA_H'] = df['<HIGH>'].ewm(span=n).mean()
        ##
        df['SMA_L'] = df['<LOW>'].rolling(window=n).mean()
        df['WMA_L'] = df['<LOW>'].rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        df['EMA_L'] = df['<LOW>'].ewm(span=n).mean()
        ##
        ##
        df['SMA_LA'] = df['<LAST>'].rolling(window=n).mean()
        df['WMA_LA'] = df['<LAST>'].rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        df['EMA_LA'] = df['<LAST>'].ewm(span=n).mean()

        ##----------
        df['vol_diff'] = df['<VOL>'].diff()
        ##------------MACD-------------------
        exp1 = df['<CLOSE>'].ewm(span=12).mean()
        exp2 = df['<CLOSE>'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        return df

    ##-------------------------- RSI ------------------------------
    def rsi(df, periods=14, ema=True):
        """
        Returns a pd.Series with the relative strength index.
        """
        close_delta = df['<CLOSE>'].diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema == True:
            # Use exponential moving average
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window=periods).mean()
            ma_down = down.rolling(window=periods).mean()

        rsi = ma_up / ma_down
        df['rsi'] = 100 - (100 / (1 + rsi))
        return df

    def momentum(df, n, diff=True):
        if diff:
            df['Momentum'] = df['<CLOSE>'].diff(n)
        else:
            m = []
            for i in range(len(df)):
                if i < n:
                    value_m = np.nan
                    m.append(value_m)
                else:
                    value_m = (df.loc[i, '<CLOSE>'] / df.loc[i - n, '<CLOSE>']) * 100
                    m.append(value_m)
            df['Momentum'] = m
        return df

    def stochastic(df, k_period=14, d_period=3):
        # Adds a "n_high" column with max value of previous 14 periods
        df['n_high'] = df['<HIGH>'].rolling(k_period).max()
        # Adds an "n_low" column with min value of previous 14 periods
        df['n_low'] = df['<LOW>'].rolling(k_period).min()
        # Uses the min/max values to calculate the %k (as a percentage)
        df['%K'] = (df['<CLOSE>'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
        # Uses the %k to calculates a SMA over the past 3 values of %k
        df['%D'] = df['%K'].rolling(d_period).mean()
        return df

    def wr(df, n):
        highh = df['<HIGH>'].rolling(n).max()
        lowl = df['<LOW>'].rolling(n).min()
        df['wr'] = -100 * ((highh - df['<CLOSE>']) / (highh - lowl))
        return df

    def CCI(df, ndays):
        df['TP'] = (df['<HIGH>'] + df['<LOW>'] + df['<CLOSE>']) / 3
        df['sma'] = df['TP'].rolling(ndays).mean()
        df['mad'] = df['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad())
        df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
        return df

    def hurst(ts):
        ts = list(ts)
        N = len(ts)
        if N < 20:
            raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

        max_k = int(np.floor(N / 2))
        R_S_dict = []
        for k in range(10, max_k + 1):
            R, S = 0, 0
            # split ts into subsets
            subset_list = [ts[i:i + k] for i in range(0, N, k)]
            if np.mod(N, k) > 0:
                subset_list.pop()
                # tail = subset_list.pop()
                # subset_list[-1].extend(tail)
            # calc mean of every subset
            mean_list = [np.mean(x) for x in subset_list]
            for i in range(len(subset_list)):
                cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()
                R += max(cumsum_list) - min(cumsum_list)
                S += np.std(subset_list[i])
            R_S_dict.append({"R": R / len(subset_list), "S": S / len(subset_list), "n": k})

        log_R_S = []
        log_n = []
        print(R_S_dict)
        for i in range(len(R_S_dict)):
            R_S = (R_S_dict[i]["R"] + np.spacing(1)) / (R_S_dict[i]["S"] + np.spacing(1))
            log_R_S.append(np.log(R_S))
            log_n.append(np.log(R_S_dict[i]["n"]))

        Hurst_exponent = np.polyfit(log_n, log_R_S, 1)[0]
        return log_R_S, log_n, Hurst_exponent

    def add_anomaly(data):
        anomaly = []
        open = data['<OPEN>'].values
        close = data['<CLOSE>'].values
        for i in range(1, len(data)):
            if open[i] == close[i - 1]:
                anomaly.extend([0])
            else:
                anomaly.extend([1])
        data.drop(data.shape[0] - 1, axis=0, inplace=True)
        data['anomaly'] = anomaly
        return data

    def add_pro(data, n):
        pro = data['<CLOSE>'].diff(n).values[n:]
        for i in range(1, n + 1):
            data.drop(data.shape[0] - 1, axis=0, inplace=True)
        data['profit'] = pro
        return data

    def add_label(data):
        labels = []
        for i in data['profit']:
            if i >= 0:
                labels.extend([1])
            elif i < 0:
                labels.extend([2])
        data['label'] = labels
        return data

    def diff(data, feat1, feat2):
        data[feat1] = data[feat2] - data[feat1]
        return data
