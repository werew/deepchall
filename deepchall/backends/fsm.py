from .backend import Backend, ShapePlaceholder
import random
from collections import deque
import numpy as np
from typing import Optional, Dict, List, Generator, Tuple

class FSM(Backend):
  name = 'fsm'
  desc = 'TODO'
  shape = (1, ShapePlaceholder.LENGTH)

  """
  A class representing a non-deterministic Finite State Machine (FSM).
  Each instance of this class itself represents a single FSM state,
  however multiple instance linked via transitions form non-deterministic FSMs 
  of arbitrary complexity.
  The FSM instance of the first state can be used as an interface to the
  whole FSM.

  Example of usage:
    s = [FSM() for _ in range(3)]
    state_a = FSM()
    state_b = FSM()
    state_c = FSM()
    state_a.add_transition()
    s[0].add_transition(input_symbol=1, states=[s[0],s[1]])
    s[1].add_transition(input_symbol=2, states=[s[0],s[2]])
    s[2].add_transition(input_symbol=3, states=[s[2]])
    s[2].set_terminal(True)
  """
  def __init__(self, transitions : Optional[Dict[str, List["FSM"]]] = None, is_terminal: bool = False):
    """
    Create a new state instance.

    transitions: optional dictionnary of transitions
    is_terminal: indicates whether this state must be considered terminal (note 
      that a state with no transitions if already considered terminal by default)
    """
    if transitions is None:
      transitions = {}
    self.transitions = transitions
    self._is_terminal_overwrite = is_terminal

  def set_terminal(self, is_terminal: bool = True) -> None:
    """
    If true forces this state to be considered as terminal
    """
    self._is_terminal_overwrite = is_terminal

  def is_terminal(self) -> bool:
    """
    Returns whether this state is terminal or not
    """
    return self._is_terminal_overwrite or len(self.transitions) == 0

  def add_transition(self, input_symbol: int, states: List["FSM"]) -> None:
    """
    Add one or more new transitions for a given input symbol.
    If there already exist transitions for that symbol, old transitions are kept.
    Note that transitions can be duplicated.

    input_symbol: an integer representing the input consumed by the FSM
    states: list of states which we can transition to given input symbol
    """
    self.transitions[input_symbol] = self.transitions.get(input_symbol, []) + states

  def traverse(self, input_symbol: int) -> List["FSM"]:
    """
    Returns the list of states we can transition to given an input symbol
    """
    return self.transitions.get(input_symbol, [])

  def gen(self) -> Generator[np.array, None, None]:
    """
    Generates up to max_samples well formed expressions beloging to the underlying
    language (see gen).
    """
    # Queue of tuples (expr, state)
    queue = deque([(np.array([[]]), self)])

    while queue:

      expr, state = queue.popleft()

      if state.is_terminal():
        yield expr

      for input_, states in state.transitions.items():
        new_expr = np.concatenate((expr,[[input_]]), axis=1)
        for new_state in states:
          queue.append((new_expr, new_state))

  def parse(self, sample: np.array) -> bool:
    """
    Parse an expression and return True if it is well formed (i.e. it belongs to
    the language encoded by the FSM and it is recognized by this last)
    """
    # We expect samples to have shape (1, N)
    assert sample.shape[0] == 1

    if sample.shape[1] == 0:
      return self.is_terminal()

    states = self.traverse(int(sample[0,0]))
    for state in states:
      if state.parse(sample[:, 1:]):
        return True
    return False
