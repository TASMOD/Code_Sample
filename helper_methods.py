from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def project(X, p):
    """Projects the data using given method.

    Parameters:
        X (numpy.ndarray): The input data we want to project.
        p (projection method): The projection method we want to teach the nn

    Returns:
        projection_coordinates(numpy.ndarray):
             The coordinates of the projection.
    """
    X_new = p.fit_transform(X)
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_new)


def train_model(X, X_2d):
    """Creates and trains a neural network to learn the projection.

    Parameters:
        X (numpy.ndarray): The input data we want to project.
        X_2d (numpy.ndarray): The projection of the input data (y-true).

    Returns:
        m (keras.engine.sequential.Sequential): The trained neural network.
        hist (history.history): The training history and accuracy of training.
    """
    nnsettings = dict()
    nnsettings['std'] = dict()
    nnsettings['std']['wide'] = [256, 512, 256]

    stop = EarlyStopping(verbose=1, min_delta=0.00001, mode='min',
                         patience=10, restore_best_weights=True)
    callbacks = [stop]

    m = Sequential()

    layers = nnsettings['std']['wide']

    m.add(Dense(layers[0], activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001),
                input_shape=(X.shape[1],)))
    m.add(Dense(layers[1], activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.add(Dense(layers[2], activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.add(Dropout(0.5))

    m.add(Dense(2, activation='sigmoid',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.compile(loss='mean_absolute_error', optimizer='adam')

    hist = m.fit(X, X_2d, batch_size=32, epochs=1000, verbose=0,
                 validation_split=0.05, callbacks=callbacks)

    return m, hist
