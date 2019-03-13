__author__ = 'kr4in'
__date__ = '13-03-2019'
__description__ = 'Set of functions useful for preprocessing time series data.'


# ---------------
# IMPORTS -------
# ---------------

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


# ---------------
# FUNCTIONS -----
# ---------------

def to_supervised(train_data, n_input):
    """
    Create walk-forward training data from passed data and the number of points
    in each sequence value.

    Args:
        train_data (pd.DataFrame): 2-d time-series data (rows=timesteps,
        columns=features)

        n_input (integer): number of points in each sequence value (eg. if
        breaking down into weeks then n_input=7). Note that if the input data
        can't be factored by n_input then the data will be trimmed (ie. if
        len(train_data) % n_input != 0 then trim train_data).

    Returns:
        (np.array, np.array): return a tuple of two numpy arrays where array 1
        is the training data and array 2 is the testing data.

    """
    td_shape = train_data.shape
    data = train_data.reshape(-1, n_input, td_shape[-1])

    # The test value will always need to be the last split
    tscv = TimeSeriesSplit(n_splits=len(data)-1)

    X, y = [], []
    for train_i, test_i in tscv.split(data):
        X.append(data[train_i[-1]])
        y.append(data[test_i[0]])
    return np.array(X), np.array(y)
