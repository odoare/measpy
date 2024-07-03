# measpy/utils.py
#
# -----------------------------------
# Measpy : various utility functions
# -----------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

def mic_calibration(sig, current_cal, target_db = 94.0):
    """ Microphone calibration function
    """

    measured_spl = 20*np.log10(sig.rms()/mp.PREF)
    newsens = current_cal*10**( (measured_spl-target_db) / 20 )

    return newsens

