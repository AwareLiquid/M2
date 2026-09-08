import sys

from .research import rhythm as _implementation

sys.modules[__name__] = _implementation
