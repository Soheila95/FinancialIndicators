import pandas as pd

class Initial:

    def CreateData(self,path_data):
        df = pd.read_csv(path_data)
        DateColumn = df.columns[0]
        self.DateColumn = DateColumn
        df[DateColumn ]=pd.to_datetime(df[DateColumn], format='%Y-%m-%d')
        df.sort_values(by=DateColumn,inplace=True)
        df.reset_index(drop=True,inplace=True)
        #df = df[(df[DateColumn]>'2011-03-21') & (df[DateColumn]<'2021-08-23')]
        df.reset_index(drop=True,inplace=True)

        return df