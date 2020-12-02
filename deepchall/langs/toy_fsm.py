from .lang import Lang
from ..backends.fsm import FSM
from ..backends.backend import Backend
from typing import Dict

class ToyFSM(Lang):
    name = 'toy_fsm'
    desc = """
        A very simple (circular) 3-states FSM, with three transitions and all
        terminal states.
    """
    alphabet_size = 3
    shape = (1, None)
    extra_params = {}

    def init(self, params: Dict) -> None:
        pass 

    def get(self) -> Backend:
        s = [FSM(is_terminal=True) for _ in range(3)]
        s[0].add_transition(input_symbol=0, states=[s[0],s[1]])
        s[1].add_transition(input_symbol=1, states=[s[1],s[2]])
        s[2].add_transition(input_symbol=2, states=[s[2],s[0]])
        return s[0]