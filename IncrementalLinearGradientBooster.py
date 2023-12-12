import numpy as np
import pandas as pd
from math import factorial
import matplotlib.pyplot as plt


class IncrementalLinearGradientBooster:
    def __init__(self, n_steps: int= 500, gamma: float=1, loss: str='L2', eps: float= 10**(-10), verbose=False, order=1, delimiter='_') -> None:
        self.supported_loss = { 
            'L2': self._loss_L2,
            'L1': self._loss_L1
            }
        
        assert gamma <= 1
        assert loss in self.supported_loss.keys()
        
        self.verbose    = verbose
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.loss       = self.supported_loss[loss]
        self.is_fitted  = False
        self.eps        = eps
        self.order      = order
        self.delimiter  = delimiter

    def _loss_L2(self, y, yhat):
        """
        Square loss, or Residual Sum of Squares
        """
        return np.sum((y-yhat)**2)
    
    def _loss_L1(self, y, yhat):
        """
        Absolute loss
        """
        return np.sum(np.abs((y-yhat)))
    
    def _recast_input(self, X):
        """
        TODO:
        fix bug where it does not accept pd.DataFrames
        """
        if type(X) == np.ndarray:
            return X
        elif type(X) == pd.core.frame.DataFrame:
            return X.to_numpy()
        else:
            raise TypeError('Only pandas data frames or numpy ndarrays are supported')


    def expand(self, df):
        try:
            p = len(df.columns)
        except:
            df = pd.DataFrame(data=df)
            p = len(df.columns)
        dfs = [df]

        for power in range(1, self.order):
            tmp = pd.DataFrame()
            for colname in df.columns:
                
                for df2 in dfs:
                    for colname2 in df2.columns:
                        tmp[str(colname) + self.delimiter + str(colname2)] = df[colname] * df2[colname2]
            dfs.append(tmp)
        
        return pd.concat(dfs, axis=1)

    def fit(self, X, y, Xtest= None, ytest=None):
        """
        X: features
        y: target

        At this stage, we assume that X and y are both preprocessed and scaled.
        We therefore do not model intercept.
        """
        try:
            expanded = self.expand(X)
        except:
            expanded = self.expand(pd.DataFrame(data=X))

        self.X = self._recast_input(expanded).copy()   # original X, as an array
        self.y = self._recast_input(y).copy()   # original y, as an array
        self.working_y = self.y.copy()          # the residuals for weak lerner i
        self.n_columns = self.X.shape[1]
        self.actual_steps = -1
        n_train = self.X.shape[0]               # number of columns

        self.trace = np.zeros((self.n_steps, self.n_columns))
        # array which stores the coefficients found
        self.coeff = np.zeros(self.X.shape[1])

        self.mse = []
        self.mse.append(self.loss(self.y, self._predict(self.X))/n_train)

        testset = False
        if Xtest is not None and ytest is not None:
            self.Xtest = Xtest
            self.ytest = ytest
            testset = True
            n_test = self.Xtest.shape[0]
            self.msetest = []

        for i in range(self.n_steps):
            best_coeff = 0              # value of best coeff to add to model
            best_coeff_index = -1       # index of the best model found on this iteration
            best_coeff_score_inv = 0    # 1/MSE s.t. 0 is infinitely bad.
            if self.verbose and i%(self.n_steps//10) == 0:
                print(f'{i/self.n_steps*100: .2f} %')


            for j in range(len(self.coeff)):
                x = self.X[:,j]

                beta = np.sum(x*self.working_y,)/np.sum(x*x)
                residuals = self.loss(self.working_y, x*beta)

                if 1/residuals > best_coeff_score_inv:
                    best_coeff_index = j
                    best_coeff = beta
                    best_coeff_score_inv = 1/residuals
                

            if abs(self.mse[i] - residuals)*self.n_steps < self.eps:
                self.actual_steps = i
                #break

            best_coeff_scaled               = best_coeff * self.gamma
            self.coeff[best_coeff_index]    += best_coeff_scaled
            self.trace[i,best_coeff_index]  = best_coeff_scaled
            self.working_y                  = self.y - self._predict(self.X)

            self.mse.append(self.loss(self._predict(self.X), self.y)/n_train)
            if testset:
                self.msetest.append(self.loss(self._predict(self.Xtest), self.ytest)/n_test)



        self.is_fitted = True

        if self.verbose:
            print('Done fitting', self.coeff)


    def _predict(self, X):
        """
        Interntal prediction for use during fitting.
        """
        return X@self.coeff
    
    def predict(self, X):
        """
        Predicts new y-s with the current model.

        Under asumption of linear models:
            y_hat = X*Beta
        """

        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before performing prediction')

        return self._predict(X)
    
    def score(self, X, y):
        """
        Return the coefficient of determination of the prediction.

        R^2 = 1 - RSS/TSS = 1 - Residual Sum of Squares / Total Sum of Squares

        1 is perfect prediction, while 0 is no prediction

        TODO:
        Implement score for different loss functions
        """
        y_pred = self.predict(X)
        u = ((y - y_pred)** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def trace_loss(self, X, y):
        """
        Calculates the mean loss on a new data set X with labels y for every step
        """
        try:
            X = self.expand(X)
        except:
            X = self.expand(pd.DataFrame(data=X))

        coef = np.zeros_like(self.coeff)
        mse = []
        n = X.shape[0]
        for i in range(self.n_steps):
            mse.append(self.loss(X@coef, y)/n)
            coef += self.trace[i]

        return mse
    
    def make_trace_plot(self, X, y, pltkwargs={}, outfile=None):
        mse = self.trace_loss(X, y)
        plt.plot(range(self.n_steps), mse, **pltkwargs)

        if outfile is not None:
            plt.savefig(outfile)
        else:
            plt.plot()



    
    def get_transformed_X(self):
        """
        returns the basis expanded input matrix 
        """
        return self.X


def test_full_linearity(n=10000, p=10, verbose=False):
    """
    tests that the Incremental Gradient Boost methood and Ordinary Least Squares methood
    gives approximate same solution, when fitted enough.
    """
    
    # construct data
    X = np.random.normal(size=[n, p])           # generate random X-s
    Betas = np.random.normal(loc= 1, size=[p])  # generate random coefficients
    y = X@Betas + np.random.normal()            # generate y-s, and add normal noise

    # estimate coefficients
    beta_ols = np.linalg.inv(X.T@X)@(X.T)@y     # Least square solution
    model = IncrementalLinearGradientBooster()
    model.fit(X,y)

    # Compare results
    deviance = np.sum((beta_ols - model.coeff)**2)

    if verbose:
        print('True coeff:\t', Betas)
        print('Ols coeff:\t', beta_ols)
        print('Boosted coeff:\t', model.coeff)
        print('Diff:\t\t', deviance)
    
    assert deviance < 10**(-10)


def test_only_linearity(n=10000, p=10, verbose=False, order = 2):
    """
    tests that the Incremental Gradient Boost methood and Ordinary Least Squares methood
    gives approximate same solution even when second order effect could be added.
    """
    
    # construct data
    X = np.random.normal(size=[n, p])           # generate random X-s
    Betas = np.random.normal(loc= 1, size=[p])  # generate random coefficients
    y = X@Betas + np.random.normal()            # generate y-s, and add normal noise

    # estimate coefficients
    beta_ols = np.linalg.inv(X.T@X)@(X.T)@y     # Least square solution
    model = IncrementalLinearGradientBooster(order=order)
    model.fit(X,y)

    if verbose:
        print(model.coeff)

    assert len(model.coeff) == expanded_columns_count(p, order)

def expanded_columns_count(p, order):
    # Calculates the number of columns after the expansion of a given order
    res = 0
    for i in range(1, order + 1):
        res += p**i
    return res

def map_order_to_penalty(order):
    pass

if __name__ == '__main__':
    # test_full_linearity()
    test_only_linearity(verbose=True, n=10**4)
    