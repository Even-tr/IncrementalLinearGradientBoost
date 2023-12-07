import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from columnExpander import ColumnExpander

def make_preprocessor(numerical_features, binary_features, categorical_features):
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.1, sparse=False)),
        ]
    )

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())]
    )


    # General preprocesser which encodes and scales all features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
            ('binary', binary_transformer, binary_features)
        ],
        verbose_feature_names_out = True,
        remainder='drop'                # drop untouched features since after this step, as it is the last preprocessing one
    ).set_output(transform="pandas")    # Keep data frame format

    return preprocessor

def make_expansion_transformr(numerical_features, binary_features, categorical_features, order):
    ce = ColumnExpander(order=order)

    pipeline = ColumnTransformer(
        transformers=[
            ('Expand', ColumnExpander, numerical_features + binary_features + categorical_features)
        ]
    )

    return pipeline


def _load_iris():
    from sklearn import datasets
    import pandas as pd

    iris = datasets.load_iris(as_frame=True)
    df = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)
    df['target'] = y
    df = df[df['target'] != 2] # remove third class
    y = df['target']
    X = df.drop('target', axis=1)
    num_features = X.columns
    binary_features = []
    categorical_features = []

    return X, y, [num_features, binary_features, categorical_features] 



def test_make_preprocessor():
    X, y, features = _load_iris()
    preprocessor = make_preprocessor(*features)
    preprocessor.fit(X, y)
    X_transfomred = preprocessor.transform(X)

    assert X_transfomred.mean().array.mean() < 10**(-10)
    assert abs(X_transfomred.std().array.mean() - 1) < 0.1

def test_make_expansion():
    """
    DOES NOT WORK YET!
    """
    X, y, features = _load_iris()
    preprocessor = make_expansion_transformr(*features, order= 1)
    preprocessor.fit(X, y)
    X_transfomred = preprocessor.transform(X)

if __name__ == '__main__':
    test_make_preprocessor()
    # test_make_expansion()