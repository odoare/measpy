# __init__.py

from .signal import (Signal, 
                            Spectral,
                            Weighting,
                            PREF,
                            DBUREF,
                            DBVREF,
                            WDBA,
                            WDBC,
                            WDBM)
from .measurement import Measurement

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
