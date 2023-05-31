from Initial_Indictors import Initial
stock_data = 'StockData/'

Exploratory = Initial()
df = Exploratory.CreateData(stock_data+'AAP.csv')
print(df)
