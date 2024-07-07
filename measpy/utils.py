# measpy/utils.py
#
# -----------------------------------
# Measpy : various utility functions
# -----------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

import numpy as np
import measpy as mp

def mic_calibration_level(sig, current_cal=1., target_db = 94.):
    """ Microphone calibration function

        Howto use:
            - Record mic signal with pistonphone using

            .. highlight:: python
            .. code-block:: python

                import measpy as mp
                FS = 48000
                DUR = 10
                pa = mp.Signal(fs=FS, unit='Pa')
                M = mp.Measurement( device_type='audio',
                                    fs = FS,
                                    in_sig=[pa],
                                    in_map=[1],
                                    dur=DUR,
                                    in_device='Default')
                audio_run_measurement(M)
            
            - Recorded signal is then analysed to get actual calibration

            .. highlight:: python
            .. code-block:: python

                newcal = mic_calibration_level(M.in_sig[0],current_cal=1.)

            - Calibration is then the value to use at signal creation

    """
    measured_spl = 20*np.log10(sig.rms/mp.PREF)
    return current_cal*10**( (measured_spl-target_db) / 20 )

def mic_calibration_freq(sig, sigref, Wref=None, noct=3, nperseg=None):
    """ Microphone frequency responde calibration function
    """
    if Wref==None:
        if nperseg==None:
            return abs(sig.tfe_welch(sigref)).nth_oct_smooth_to_weight(noct)
        else:
            return abs(sig.tfe_welch(sigref, nperseg)).nth_oct_smooth_to_weight(noct)
    else:
        if nperseg==None:
            return abs(sig.tfe_welch(sigref.rfft().apply_weighting(Wref).irfft())).nth_oct_smooth_to_weight(noct)
        else:
            return abs(sig.tfe_welch(sigref.apply_weighting(Wref), nperseg)).nth_oct_smooth_to_weight(noct)

