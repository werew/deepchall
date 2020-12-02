from .lang import Lang
from ..backends.cfg import CFG
from ..backends.backend import Backend
from typing import Dict

class ToyCFG(Lang):
    name = 'toy_cfg'
    desc = (
        """
        A very simple grammar, including all expressions where
        any number of 0s is followed by the same number of 1s.
        Example: 01, 0011, 000111
        """
    )
    alphabet_size = 2
    shape = (1, None)
    extra_params = {
        "max_depth": ("Maximum expansion depth of the grammar", 20),
    }

    def init(self, params: Dict) -> None:
        self._max_depth = params["max_depth"]
        self._grammar = (
            """
            S -> '0' S '1' | 
            """
        )

    def get(self) -> Backend:
        return CFG(grammar=self._grammar,max_depth=self._max_depth)