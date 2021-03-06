from ..backends.backend import Backend
from typing import Dict

class Lang:
    """
    Name of the language
    """
    name = None

    """
    Description of the language
    """
    desc = None

    """
    Alphabet size used by the languge, must be an int > 0. For example, if
    alphabet_size = 3 then the language generates values v, where 0 <= v < 3.
    Generating values out of this range is considered a bug.
    """
    alphabet_size = None

    """
    The shape of the samples generated by the language. This must match the
    shape returned by the backend. The shape can contain a None element to
    indicate a variable-size dimension.
    """
    shape = None

    """
    A dictionary of extra parameters that can be passed to the language. For
    each parameter the dictionary should contains a tuple with a description
    of the parameter and its default value, example:

    {
        'units': (
            'Number of hidden units', 30
        ),
    }
    """
    params = {}

    def init(self, params: Dict) -> None:
        """
        This method is called before starting to generate samples. 
        It should be used to perform any initialization operation. The params
        argument is a dict containing the following parameters definitions.

        max_samples: an int, the maximum amount of samples that will be
        generated by the language

        max_length: an optional int, the maximum length of the expressions
        generated by the language, if None then the language doesn't offer
        any guarantees about the maximum size

        The dict will also include any language-specific parameters defined
        in extra_params.

        All parameters are guaranteed to be present in the params dictionary,
        including language-specific parameters as long as they are defined in 
        the extra_params attribute.
        """
        raise NotImplementedError('Method init not implemented')

    def get(self) -> Backend:
        raise NotImplementedError('Method get not implemented')