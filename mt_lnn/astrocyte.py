import sys

from .research import astrocyte as _implementation

sys.modules[__name__] = _implementation
