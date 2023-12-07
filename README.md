A simple attempt at a (hopefully novel) boosting algorithm whit linear basis learners
such that the final model is itself a linear model

## API

I try to follow the sklearn API format as this will make integration in preexisting pipelines as easy as possible. It is however not done, and supports only the bare essentials of it to test the algorithm.

## Structure

The boosting algorithm is in `IncrementalLinearGradientBooster.py`.

`baseline.py` contains a collection of baseline model to compare against

`columnExpander.py` contains the (yet to be finished) column expander transformer, which will expand the features into all effects of order `order` and lower.

`preprocess.py` contains a function which makes a simple pipeline which preprocesses the data in a sensible manner. It will eventually add the column expansion as an optional step (although essential step for the justification of the algorithm).

`notebook.ipynb` contains some exploratory code to test an earlier version of the model.