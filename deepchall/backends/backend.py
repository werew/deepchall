import numpy as np
from typing import Optional, Generator, Union, Tuple
from enum import Enum

class ShapePlaceholder:
  CONFIGURABLE = 'configurable'
  LENGTH = None

class Backend:
  """
  Name of the backend
  """
  name = None

  """
  Description of the backend
  """
  desc = None

  """
  Shape of the samples returned by the gen method.
  This is a tuple containing a mix of its and elements from
  ShapePlaceholder, each representing a dimension:
  - an int means that the dimension has a fixed size
  - ShapePlaceholder.CONFIGURABLE indicates that the dimension
    can be configured by the language (i.e. it is fixed by the 
    language using the backend)
  - ShapePlaceholder.LENGTH means that the dimension is variable
    and can differ among samples generated. There can only be one
    ShapePlaceholder.LENGTH dimension, this dimension determines
    the length of the sample
  Example, a backend generating lists of the top performing N currencies
  where N is configurable but the lists can have any length would
  have the following shape:
    (ShapePlaceholder.CONFIGURABLE, ShapePlaceholder.LENGTH)
  """
  shape = None

  def gen(self, max_length: Optional[int] = None
    ) -> Generator[np.array, None, None]:
    """
    Generates samples from the underlying backend.

    max_length: max length of the samples to be enforced, if None generate 
      samples of any length.
    """
    raise NotImplementedError("Method gen not implemented")

  def parse(self, sample: np.array) -> int:
    """
    Takes a sample and returns the appropriate label
    """
    raise NotImplementedError("Method label not implemented")