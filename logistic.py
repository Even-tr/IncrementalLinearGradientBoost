import numpy as np
import pandas as pd

class Logistic:
    def __init__(self, max_iter = 100, fit_intercept=True) -> None:
        self.fitted = False
        self.fit_intercept = fit_intercept
        self.intercept = 0
        self.max_iter = max_iter
        self.learning_rate = 1

    def fit(self, X, y):
        """
        Fits a logisitc regression model (with intercept already fitted), using the algorithm as described
        in Elements of Statistical Learning (2 ed. 2009) page 120.
        """
        self.coef_ = np.zeros(shape=X.shape[1])
        if self.fit_intercept:
            self._fit_intercept(y)

        for i in range(self.max_iter):
            p = self._predict(X, as_probs=True) # Calculate probabilities with old intercept
            W = np.eye(X.shape[0])*p*(1-p)
            a = np.linalg.inv(X.T@W@X)
            b = X.T
            c = np.expand_dims((y-p), axis=0).T
            self.coef_ += self.learning_rate*(a@b@c).flatten()
            self.coef_/=self.coef_.sum()

        self.fitted = True

    def _fit_intercept(self, y):
        """
        Fits the intercept of a logistic regression model using the algorithm as described
        in Elements of Statistical Learning (2 ed. 2009) page 120.
        """
        X = np.ones_like(y)
        beta = self.intercept

        for i in range(self.max_iter):
            p = np.exp(beta)/(1 + np.exp(beta)) # Calculate probabilities with old intercept
            W = np.eye(X.shape[0])*p*(1-p)
            beta += self.learning_rate/(X.T@W@X) *X.T@(y-p)

        self.intercept = beta


    def _predict(self, X, threshold=0.5, as_probs=False):
        res = np.zeros(shape=X.shape[0])
        for i in range(len(res)):
            e = np.exp(self.intercept + self.coef_.T@X[i])
            res[i] = 1/(1 + e)

        if as_probs:
            return res
        else:
            return (res > threshold).astype(int)

    def predict(self, X, threshold=0.5, as_probs=False):
        if not self.fitted:
            raise RuntimeError('Model must be fitted before prediction')
        return self._predict(X, threshold=threshold, as_probs=as_probs)
    
    def score(self, X, y):
        yhat = self.predict(X)
        return np.mean(yhat == y)


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(2023)

    data = load_breast_cancer(as_frame=True)
    X = pd.DataFrame(data.data)
    y = pd.DataFrame(data.target)

    df = X.copy()
    df['target'] = y
    df = df[df['target'] != 2]
    # df = df.sample(frac=0.7, replace=False)

    y = df['target'].to_numpy()
    X = df.drop(['target'], axis=1).to_numpy()

    xtrain, Xtest, ytrain, ytest = train_test_split(X, y)


    model = Logistic( fit_intercept=False, max_iter=100,)
    model.fit(xtrain, ytrain)

    model2 = LogisticRegression(solver='newton-cg', fit_intercept=False)

    model2.fit(xtrain, ytrain)

    print('score 1', model.score(Xtest, ytest))
    print('score 2', model2.score(Xtest, ytest))

    print(model.intercept)
    print(model2.intercept_)

    print(model.coef_)
    print(model2.coef_)
