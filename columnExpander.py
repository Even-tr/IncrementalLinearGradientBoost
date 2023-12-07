import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnExpander(BaseEstimator, TransformerMixin):
    def __init__(self, order:int=1, delimiter = '_'):
        super().__init__()
        self.order = order
        self.delimiter = delimiter

    def fit(self, X, y):
        return self.expand(X, n=self.order)

    def predict(self, X, y):
        return self.expand(X, n=self.order)


    def expand(self, df, n=2):
        p = len(df.columns)
        dfs = [df]

        for power in range(1, n):
            tmp = pd.DataFrame()
            for colname in df.columns:
                
                for df2 in dfs:
                    for colname2 in df2.columns:
                        tmp[str(colname) + self.delimiter + str(colname2)] = df[colname] * df2[colname2]
            dfs.append(tmp)
        
        return pd.concat(dfs, axis=1)


