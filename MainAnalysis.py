from Initial_Indictors import Initial
stock_data = 'StockData/'

Exploratory = Initial()
df = Exploratory.CreateData(stock_data+'AAP.csv')
df = Exploratory.MA_MACD(df,Exploratory.Close,10)


