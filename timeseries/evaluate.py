__author__ = 'kr4in'
__date__ = '13-03-2019'
__description__ = 'Set of functions useful for evaluating time series data.'


# ---------------
# IMPORTS -------
# ---------------

import numpy as np


# ---------------
# FUNCTIONS -----
# ---------------

def resids(mdl, x_test, y_test, n_input):
    """
    A function that returns residuals.

    Args:
        mdl (keras.model): Keras model that will predict new data.
        x_test (np.array): Input testing data.
        y_test (np.array): True values for evaluation.
        n_input (integer): Number of values in a sequence.

    Returns:
        np.array: Residuals (predicted - testing)

    """
    y_hats = []
    for row in x_test:
        input_x = row.reshape(1, n_input, -1)
        y_hat = mdl.predict(input_x)
        y_hats.append(y_hat)
    y_hats = np.array(y_hats).reshape(-1, n_input)

    if y_hats.shape == y_test.shape:
        resids = y_hats - y_test
        resids = resids.reshape(-1, 7)
        return resids
    else:
        print('Predicted shape: %s' % y_hats.shape)
        print('Evaluation shape: %s' % y_test.shape)
        print('Predictions and training data shape mismatch.')
