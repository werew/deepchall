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
    Alphabet size used by the languge, must be an int > 0.
    For example, if alphabet_size = 3 then the language
    generates values v, where 0 <= v < 3. Generating values
    out of this range is considered a bug.
    """
    alphabet_size = None

    shape = None

    """
    A dictionary of extra parameters that can be passed to the
    language. For each parameter the dictionary should contains a tumple
    with a description of the parameter and its default value, example:

    {
        'units': (
            'Number of hidden units', 30
        ),
    }
    """
    lang_params = {}

    def init(self, params: Dict) -> None:
        """
        This method is called before starting to generate samples. 
        It should be used to perform any initialization operation. The params
        argument is a dict containing the following parameters definitions:
        max_samples: an int, the maximum amount of samples that will be
            generated by the language
        max_length: an optional int, the maximum length of the expressions
            generated by the language, if None then the language doesn't
            offer any guarantees about the maximum size
        <param>: any language-specific parameters defined in lang_params

        All parameters are guaranteed to be present in the params dictionary,
        including language-specific parameters as long as they are defined in 
        the lang_params attribute.
        """
        raise NotImplementedError('Method init not implemented')

    def get(self) -> Backend:
        raise NotImplementedError('Method get not implemented')