import sys

from .research import gwtb as _implementation

sys.modules[__name__] = _implementation
