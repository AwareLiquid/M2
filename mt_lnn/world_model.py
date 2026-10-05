import sys

from .research import world_model as _implementation

sys.modules[__name__] = _implementation
