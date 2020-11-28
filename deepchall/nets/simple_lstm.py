from .net import Net, UnsupportedNetParamError
import numpy as np
import tensorflow as tf 
from typing import Dict, Generator
from .utils import zero_pad_to_length, sample_categorical

class SimpleLSTM(Net):
    name = 'simple_lstm'

    desc = """
    A single-layer LSTM trained to predict 
    """

    net_params = {
        'units': ('Number of units in the LSTM layer', 30),
    }

    def __init__(self):
        self._model = None
        self._params = None
        self._length = None
        self._alphabet_size = None

    def init(self, params: Dict) -> None:
        self._params = params
        if params['max_length'] is None:
            raise UnsupportedNetParamError('max_length')

        self._length = params['max_length']+1
        self._alphabet_size = params['alphabet_size']+1

        self._model = self._make_model(
            self._length,
            self._alphabet_size,
        )

    def _make_model(self, length: int, alphabet_size: int):
        X = tf.keras.layers.Input(shape=(length, alphabet_size))
        tmp = tf.keras.layers.LSTM(
            self._params['units'], 
            return_sequences=True)(X)
        tmp = tf.keras.layers.Dense(alphabet_size, activation='softmax')(tmp)
        model = tf.keras.Model(inputs=[X],outputs=[tmp])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, gen: Generator[np.array, None, None]) -> None:
        tmp = zero_pad_to_length(list(gen), length=self._length-1, axis=1)

        # X starts with a zero indicating the beginning of the expr
        X = np.pad(tmp, [(0,0),(1,0)], 'constant', constant_values=0)

        # Y is like X but one step ahead and zero padded
        Y = np.pad(tmp, [(0,0),(0,1)], 'constant', constant_values=0)

        # Use one-hot encoding
        X = tf.one_hot(
            indices=X, 
            depth=self._alphabet_size, 
            on_value=1., 
            off_value=0.,
        )
        Y = tf.one_hot(
            indices=Y, 
            depth=self._alphabet_size, 
            on_value=1., 
            off_value=0.,
        )

        self._model.fit(
            x=X, 
            y=Y, 
            epochs=self._params['epochs'], 
            batch_size=20, 
            verbose=0,
        )

    def gen(self) -> np.array:
        X = tf.one_hot(
            indices=[[0,]], 
            depth=self._alphabet_size, 
            on_value=1., 
            off_value=0.,
        )
        for _ in range(self._length):
            preds = self._model(X)
            preds = sample_categorical(preds[:,-1,:], temperature=1.0, num_samples=1)
            preds = tf.one_hot(
                indices=preds, 
                depth=self._alphabet_size, 
                on_value=1., 
                off_value=0.,
            )
            X = tf.concat([X,preds],1)

        # From one-hot back to int representation
        X = tf.argmax(X, axis=2)

        # Remove initial zero 
        X = X[:, 1:]

        # Ignore all elements from the first zero 
        for i in range(X.shape[1]):
            if X[0,i] == 0:
                X = X[:,:i]
                break
        X -= 1
        return X.numpy()