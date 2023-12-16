import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize


def deviance(y, y_prob):
    """ 
    Loss function for binary classification. 'y' is the true class labels
    while 'y_prob' is the estimated class probability
    """
    return (-y*np.log(y_prob) - (1-y)*np.log(1-y_prob))

def sigmoid(z):
    """
    The sigmoid function: f(z) = 1/(1+ e^-z).

    Due to numerical considerations, it calculates the result in 
    two different ways, and returns the one which is numerically most stable.
    """
    res1 = 1/(1+np.exp(-z*(z>0)))
    res2 = np.exp(z*(z<=0))/(1+ np.exp(z*(z<=0)))
    return res1 * (z>0) + res2 * (z<=0)

def minimize_custom(x, y, x0=0):
    """
    Finds the maximum likelihood beta for a single column.
    It uses the newton raphson method.

    TODO:
        * add early stop and 
        * add former estimate as start estimate x0
    """
    beta = x0
    for i in range(20):
        p = sigmoid(beta*x)         # Probability under current model
        jac = (x*(y - p)).sum()     # Jacobian matrix
        hess = -(x*x*p*(1-p)).sum() # Hessian matrix

        beta += -1/hess * jac
        # print(i, beta, p, jac, hess)

    return beta


class Logistic:
    def __init__(self, n_steps = 500, gamma = 1, fit_intercept = True, verbose = False) -> None:
        self.gamma = gamma
        self.n_steps = n_steps
        self.intercept = 0
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.is_fitted = False
        self.verbose = verbose
        

    def fit_intercept_func(self, X, y):
        """
        Fits the intercept fully, as we don't penalize it.
        It could perhaps be done manually with the already implemented methodo.
        """
        return minimize(lambda x: deviance(y, sigmoid(x)).sum(), x0=0).x[0]
    
    def _predict_prob(self, X):
        z = X@self.coef_ + self.intercept
        return sigmoid(z)
    
    def predict_prob(self, X):
        """
        Predicts the probability of being a label.

        Must be fitted prior.
        """

        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before performing prediction')
        return self._predict_prob(X)
    
    def _predict(self, X, threshold=0.5):
        return (self._predict_prob(X) > threshold).astype(int)
    
    def predict(self, X, threshold=0.5):
        """
        Predicts the class if the probability is over the threshold. 

        Must be fitted prior.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before performing prediction')
        return self._predict(X, threshold=threshold)
    

    def fit(self, X, y):
        self.coef_ = np.zeros_like(X[0])
    
        if self.fit_intercept:    
            self.intercept = self.fit_intercept_func(X, y)

        n_columns = X.shape[1]
        working_y = y.copy()
        total_deviance = []
        self.trace = np.zeros((self.n_steps, n_columns))


        for i in range(self.n_steps):
            if self.verbose:
                print(f'step: {i}')

            best_coef = 0
            best_coef_col = 0
            best_coef_dev = 10**10

            for j in range(n_columns):
                x = X[:,j]
                beta = minimize_custom(x, working_y)
                probs = sigmoid(x*beta)
                dev = deviance(working_y, probs).sum()

                if dev < best_coef_dev:
                    best_coef = beta
                    best_coef_col = j
                    best_coef_dev = dev

            best_coef_scaled = best_coef*self.gamma
            self.coef_[best_coef_col] += best_coef_scaled
            working_y = y - self._predict_prob(X)
            total_deviance.append(deviance(y, self._predict_prob(X)).sum())
            self.trace[i, best_coef_col]  = best_coef_scaled

        self.total_deviance = np.array(total_deviance)
        self.is_fitted = True


# ============================================================
# Helper methods, not neccesary for fitting


    def score(self, X, y):
        """
        Returns the mean accuracy score
        """
        return (self.predict(X) == y).mean()

    def trace_loss(self, X, y, scale=True):
        """
        Calculates the mean loss on a new data set X with labels y for every step
        """

        coef_ = np.zeros_like(self.coef_)
        loss = []
        
        if scale:
            norm = X.shape[0]
        else:
            norm = 1

        for i in range(self.n_steps):
            z = X@coef_ + self.intercept
            loss.append(deviance(y, sigmoid(z))/norm)
            coef += self.trace[i]

        return loss
    

def test_full_linearity():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_features=10,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42,
        n_samples= 500
    )

    model = Logistic(n_steps=1000, gamma=0.7)
    model.fit(X, y)

    model2 = LogisticRegression()
    model2.fit(X,y)

    print((model._predict(X) == model2.predict(X)).mean())

if __name__ == '__main__':

    test_full_linearity()

    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    """
    X, y = make_classification(
        n_features=100,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42,
        n_samples= 500
    )


    X, Xtest, y ,ytest = train_test_split(X,y)
    model = Logistic2(n_steps=100, gamma=0.1)
    model.fit(X, y)
    print(model.coef_)
    plt.plot(range(len(model.total_deviance)),model.total_deviance/X.shape[0])
    # plt.plot(range(len(total_deviance_test)), np.array(total_deviance_test)/Xtest.shape[0])
    plt.show()
"""