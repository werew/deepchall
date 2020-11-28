import numpy as np
from typing import List
import tensorflow as tf

def zero_pad_to_length(arrays: List[np.array], length: int, axis: int = 1):
    """
    
    """
    for i in range(len(arrays)):
      a = arrays[i]
      padding = length - a.shape[axis]
      if padding < 0:
        raise ValueError(
            f'Trying to pad to length {length} '+
            f'axis of bigger size {a.shape(axis)}'
        )
      pad_width = [
        (0,padding) if j == axis else (0,0) 
        for j in range(len(a.shape))
      ]
      # Pad using -1 since 0 can be used by the actual 
      # expression and then increment by one
      arrays[i] = np.pad(a, pad_width, 'constant', constant_values=-1)

    return np.concatenate(arrays, axis=0) + 1

def sample_categorical(preds, temperature: float = 1.0, num_samples: int = 1):
    preds = tf.math.log(preds)/temperature
    return tf.random.categorical(preds,num_samples=num_samples)
 
#def generate(
#    model: keras.Model, 
#    alphabet_size: int, 
#    num_samples: int = 1, 
#    max_length: int = 20, 
#    temperature: float = 1.0):
#    samples = tf.one_hot([[0,]]*num_samples, depth=alphabet_size)
#    for _ in range(max_length):
#        preds = model(samples)
#        preds = TestUtils.sample_categorical(preds[:,-1,:],temperature=temperature)
#        preds = tf.one_hot(preds, depth=alphabet_size)
#        samples = tf.concat([samples,preds],axis=1)
#    return tf.argmax(samples, axis=2)
#
#def clean_exprs(exprs: np.array):
#    exprs = exprs[:, 1:]
#    for e in exprs:
#        end_expr = np.argwhere(e == 0)[0][0]
#        e[end_expr:] = 0
#    return expr