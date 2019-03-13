__author__ = 'kr4in'
__date__ = '13-03-2019'
__description__ = 'Set of functions useful for preprocessing time series data.'


# ---------------
# IMPORTS -------
# ---------------
from keras.models import Sequential
from keras.layers import Dense, LSTM


# ---------------
# FUNCTIONS -----
# ---------------

def build_model(n_timesteps, n_features, n_outputs):

    # Define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    return model
