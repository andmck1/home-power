__author__ = 'kr4in'
__date__ = '13-03-2019'
__description__ = 'Set of functions useful for preprocessing time series data.'


# ---------------
# IMPORTS -------
# ---------------
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector


# ---------------
# FUNCTIONS -----
# ---------------

def build_model(n_timesteps, n_features, n_outputs):

    # Define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    return model
