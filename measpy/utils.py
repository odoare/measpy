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
import scipy.io.wavfile as wav
from ._tools import siglist_to_array, all_equal
from .signal import PREF

def mic_calibration_level(sig, current_cal=1., target_db = 94.):
    """ Microphone calibration function

        :param sig: Signal recorded by the microphone to calibrate
        :type sig: measpy.signal.Signal
        :param current_cal: Calibration indicated in the recorded signal sig
        :type sig: float
        :param target_db: dB SPL level of the microphone calibrator
        :type target_db: float

        :return: The actual value to use as calibration level, so that measured dB SPL equals to target_db
        :rtype: float

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
    measured_spl = 20*np.log10(sig.rms/PREF)
    return current_cal*10**( (measured_spl-target_db) / 20 )

def mic_calibration_freq(sig, sigref, Wref=None, noct=3, nperseg=None):
    """ Microphone frequency response calibration function

        :param sig: Signal recorded by the microphone to calibrate
        :type sig: measpy.signal.Signal
        :param sigref: Signal recorded by the reference microphone
        :type sigref: measpy.signal.Signal
        :param Wref: Reference microphone response function as a Weighting object
        :type Wref: measpy.signal.Weighting
        :param noct: Specifies the frequency averaging to apply. The returned measpy.signal.Weighting object will be 1/noct frequency averaged
        :type noct: int
        :param nperseg: Window size when transfer function estimation is made
        :type nperseg: int

        :return: A Weighting object containing the frequency response of the microphone
        :rtype: measpy.signal.Weighting

        Howto use:
            - Record mic and refmic signals using for example:

            .. highlight:: python
            .. code-block:: python

                import measpy as mp
                FS = 48000
                DUR = 10
                pa = mp.Signal(fs=FS, unit='Pa',cal=...)
                paref = mp.Signal(fs=FS, unit='Pa', cal=...)
                M = mp.Measurement( device_type='audio',
                                    fs = FS,
                                    in_sig=[pa,paref],
                                    in_map=[1,2],
                                    dur=DUR,
                                    in_device='Default')
                audio_run_measurement(M)

            - Recorded signal is then analysed to get actual calibration

            .. highlight:: python
            .. code-block:: python

                micresp = mic_calibration_freq(M.in_sig[0],M.in_sig[1],Wref=...)

            Wref=... corresponds to the reference mic response curve.

            - Calibration curve is then the returned measpy.Weighting object

    """
    if Wref==None:
        if nperseg==None:
            return abs(sig.tfe_welch(sigref)).nth_oct_smooth_to_weight(noct)
        else:
            return abs(sig.tfe_welch(sigref, nperseg)).nth_oct_smooth_to_weight(noct)
    else:
        if nperseg==None:
            return abs(sig.tfe_welch(sigref.rfft().apply_weighting(Wref,inverse=True).irfft())).nth_oct_smooth_to_weight(noct)
        else:
            return abs(sig.tfe_welch(sigref.apply_weighting(Wref,inverse=True), nperseg)).nth_oct_smooth_to_weight(noct)

def siglist_to_wav(sigl,filename):
    """ Takes a list of signals and export it to a multichannel wave file.
        Sampling frequencies have to match. Use the resample method if necessary.

        :param sigl: A list of single channel signals
        :type sigl: list of measpy.signal.Signal
        :param filename: Name of wave file (without extension)
        :type filename: string

        :return: True if succeded, False if error
        :rtype: bool

    """
    print([s.fs for s in sigl])
    if not(all_equal([s.fs for s in sigl])):
        print('Not possible: different sampling frequencies \n Resample first')
        return False
    wav.write(filename+'.wav', int(round(sigl[0].fs)), siglist_to_array(sigl))
    return True
