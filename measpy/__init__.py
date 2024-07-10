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

from .utils import (mic_calibration_level,
                        mic_calibration_freq,
                        siglist_to_wav)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
