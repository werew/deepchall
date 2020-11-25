from ..backends.backend import Backend
from typing import Generator, Dict
import numpy as np

class UnsupportedNetParamError(RuntimeError):
    pass

class Net:
    """
    Name of the network
    """
    name = None

    """
    Description of the network
    """
    desc = None

    """
    A dictionary of extra parameters that can be passed to this
    network. For each parameter the dictionary should contains a tumple
    with a description of the parameter and its default value, example:

    {
        'units': (
            'Number of hidden units', 30
        ),
    }

    """
    net_params = {}

    def init(self, params: Dict) -> None:
        """
        This method is called before training the network. It should be used
        to perform any initialization operation. The params argument is a
        dict containing the following parameters definitions:
        alphabet_size: an int, the number of tokens used by the language to
            train on. If this is N, then the language tokens will be represented
            by integers v where 0 <= v < N
        max_samples: an int, the maximum amount of samples that will be
            generated by the language
        max_length: an optional int, the maximum length of the expressions
            generated by the language, if None then the language doesn't
            offer any guarantees about the maximum size
        epochs: an int, number of epochs to train, to be used by net as
            a hint about the amount of training 
        <param>: any network-specific parameters defined in net_params

        All parameters are guaranteed to be present in the params dictionary,
        including net-specific parameters as long as they are defined in 
        the net_params attribute.
        """
        raise NotImplementedError('Method init is not implemented')

    def train(self, gen: Generator[np.array, None, None]) -> None:
        """
        This method should contain the training logic. The generator
        passed as input provides expressions belonging to the language
        to train for, it is up to the network to decide what to do with
        this data.
        """
        raise NotImplementedError('Method train is not implemented')

    def gen(self) -> np.array:
        """
        This method is called after training has completed. It should
        generate and return a random expression every time it is called
        (potentially many times).
        """
        raise NotImplementedError('Method gen is not implemented')