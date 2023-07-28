# measpy/audio.py
#
# -----------------------------------
# Data acquisition with audio devices
# -----------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy


import sounddevice as sd

from ._tools import siglist_to_array, t_min

import numpy as np
from numpy.matlib import repmat

from datetime import datetime
from time import time, sleep

def audio_run_measurement(M):
    """
    Runs a measurement defined in the object of
    the class measpy.measurement.Measurement given
    as argument.

    Once the data acquisition process is terminated,
    the measurement object given in argument contains
    a property in_sig consisting of a list of signals.

    :param M: The measurement object that defines the measurement properties

    :type M: measpy.measurement.Measurement

    :return: Nothing, the measurement passed as argument is modified in place.

    """

    if M.device_type!='audio':
        print("Warning: deviceType != 'audio'. Changing to 'audio'.")
        M.device_type='audio'
    if M.in_device=='':
        print("Warning: no device specified, changing to None")
        M.in_device=None
    if M.out_sig!=None:
        if M.out_device=='':
            print("Warning: no device specified, changing to None")
            M.out_device=None

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Set the audio devices to use
    # And prepare the output arrays
    if M.out_sig!=None:
        outx = siglist_to_array(M.out_sig)
        tmin = t_min(M.out_sig)
        if M.in_sig!=None:
            sd.default.device=(M.in_device,M.out_device)
        else:
            sd.default.device=(M.out_device)
    else:
        tmin = 0
        if M.in_sig!=None:
            sd.default.device=(M.in_device)
        else:
            raise Exception('No input nor output defined.')

    if M.out_sig==None:
        y = sd.rec(int(M.dur * M.fs),
                samplerate=M.fs,
                mapping=M.in_map,
                blocking=False)
    else:
        if M.in_sig==None:
            y = sd.play(outx,
                    samplerate=M.fs,
                    mapping=M.out_map,
                    blocking=False)
        else:
            y = sd.playrec(outx,
                    samplerate=M.fs,
                    input_mapping=M.in_map,
                    output_mapping=M.out_map,
                    blocking=False)
            
    sd.wait()

    if M.in_sig!=None:
        print(M.in_sig)
        for i,s in enumerate(M.in_sig):
            s.raw = np.array(y[:,i])
            s.t0 = tmin

def audio_run_synced_measurement(M,in_chan=0,out_chan=0):
    """
    Before running a measurement, one second of silence
    is added at the begining and end of the selected output channel.
    The measurement is then run, and the time lag between
    a selected acquired signal and the output signal is computed
    from cross-correlation calculation.
    All the acquired signals are then re-synced from the time lag value.

    :param M: The measurement object
    :type M: measpy.measurement.Measurement
    :param out_chan: The selected output channel for synchronization. It is the index of the selected output signal in the list ``M.out_sig``
    :type out_chan: int
    :param in_chan: The selected input channel for synchronization. It is the index of the selected input signal in the list ``M.in_sig``
    :type in_chan: int

    """
    M=M.sync_prepare(out_chan=out_chan)
    audio_run_measurement(M)
    M=M.sync_render(out_chan=out_chan,in_chan=in_chan)
    return M

def audio_get_devices():
   """
   Returns a list of audio devices present in the system
   """
   return sd.query_devices()
 