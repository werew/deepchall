from .backend import Backend, ShapePlaceholder
import random
from collections import deque
import numpy as np
from typing import Optional, Dict, List, Generator, Tuple
import nltk
from nltk.parse.generate import generate
from nltk.parse.earleychart import EarleyChartParser

class CFG(Backend):
  name = 'cfg'
  desc = 'TODO'
  shape = (1, ShapePlaceholder.LENGTH)

  def __init__(self, grammar: str, max_depth: Optional[int] = None):
    self._grammar = nltk.CFG.fromstring(grammar)
    self._max_depth = max_depth

    # Collect terminals symbols
    self._terminals = set()
    for prod in self._grammar.productions():
      for item in prod.rhs():
        if nltk.grammar.is_terminal(item):
          self._terminals.add(item)

    # Make sure all symbols are covered
    self._grammar.check_coverage(self._terminals)

    # Have a way to convert terminal symbols into integers and viceversa
    self._int_2_terminal = {}
    self._terminal_2_int = {}

    cnt = 0
    for item in self._terminals:
      self._terminal_2_int[item] = cnt
      self._int_2_terminal[cnt] = item
      cnt += 1

  def gen(self) -> Generator[np.array, None, None]:
    # Queue of tuples (expr, state)
    for terminals in generate(self._grammar, depth=self._max_depth):
      # Convert terminal symbols to integers
      converted_terminals = [self._terminal_2_int[t] for t in terminals]

      # Convert to numpy array
      yield np.array([converted_terminals])

  def parse(self, sample: np.array) -> bool:
    # We expect samples to have shape (1, N)
    assert sample.shape[0] == 1

    # Convert sample into a sentence supported
    # by the grammar
    try:
      sent = [self._int_2_terminal[val] for val in sample[0,:]]
    except IndexError:
      return False

    parser = EarleyChartParser(self._grammar)
    return parser.parse_one(sent) is not None
