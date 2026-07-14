import importlib

from .particle_init import init_particles
from . import resampling

__all__ = ['init_particles', 'resampling', 'torch_utils']


def __getattr__(name):
    if name == 'torch_utils':
        module = importlib.import_module('.torch_utils', __name__)
        globals()['torch_utils'] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")