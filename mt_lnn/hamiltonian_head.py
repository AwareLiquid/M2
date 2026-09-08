import sys

from .research import hamiltonian_head as _implementation

sys.modules[__name__] = _implementation
