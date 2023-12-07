"""
Class of baseline predicive models to compare with
"""
import numpy as np
import pandas as pd

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# ignore pesky warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

supported_models = {
    'OLS': LinearRegression(),
    'GBM': GradientBoostingRegressor()
    }

class Baseline:
    def __init__(self, type: str= 'OLS'):
        assert type in supported_models.keys(), f"'{type}' not supported"
        self.model = supported_models[type]

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
