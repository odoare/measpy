# %%
# measpy/examples/mic_calibration.py
#
# ------------------------------------------
# Example of task for microphone calibration
# ------------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

# import sys
# sys.path.insert(0, "..")

import measpy as mp
import numpy as np
from measpy.audio import audio_run_measurement, audio_get_devices
import matplotlib.pyplot as plt
from datetime import datetime

%matplotlib tk

# Duration of the recording
DUR = 5

# Sampling frequency
FS = 96000

# Mic_reference (model, serial number, etc.)
micref = 'testmic'

# Comment
comment = 'Preamp 12MIC, gain at 12dB'

# Soundcard device name
dev = 'Default'

# %% Reference level calibration using pistonphone
# ------------------------------------------------

pa = mp.Signal(fs=FS, unit='Pa')
M = mp.Measurement( device_type='audio',
                    fs = FS,
                    in_sig=[pa],
                    in_map=[1],
                    dur=DUR,
                    in_device=dev)

audio_run_measurement(M)
newcal = mp.mic_calibration_level(M.in_sig[0],current_cal=1.)
d=M.to_dir(micref+'_calibration_level')

f = open(d+'/README', "a")
f.write("\n")
f.write("\n Date: " + datetime.now().strftime("%Y-%m-%d"))
f.write("\n Time: " + datetime.now().strftime("%H:%M:%S"))
f.write("\n Device: "+dev)
f.write("\n Comment: "+comment)
f.write("\n Calibration level: "+str(newcal))
f.close()

# %% Frequency response calibration using calibration box
# -------------------------------------------------------

pa = mp.Signal(fs=FS, unit='Pa')
paref = mp.Signal(fs=FS, unit='Pa')
sout = mp.Signal.log_sweep(fs=FS,freq_min=20,freq_max=20000,dur=DUR)
Wref = mp.Weighting.from_csv('../../refmic.csv',asdB=True,asradians=False)

M = mp.Measurement( device_type='audio',
                    fs = FS,
                    in_sig=[pa,paref],
                    in_map=[1,2],
                    out_sig=[sout],
                    out_map=[1],
                    dur=DUR,
                    in_device=dev,
                    out_device=dev)

audio_run_measurement(M)

Wcal = mp.mic_calibration_freq(pa,paref,Wref=Wref)
plt.semilogx(Wcal.freqs,Wcal.adb)

d=M.to_dir(micref+'_calibration_freq')

f = open(d+'/README', "a")
f.write("\n")
f.write("\n Date: " + datetime.now().strftime("%Y-%m-%d"))
f.write("\n Time: " + datetime.now().strftime("%H:%M:%S"))
f.write("\n Device: "+dev)
f.write("\n Comment: "+comment)
f.write("\n Calibration data in wcal.csv file")
f.close()

Wcal.to_csv(d+"/wcal.csv")

# %%
