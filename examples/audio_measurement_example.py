# ------------------------
#
# measpy - A Python package to perform measurements an signal analysis
#
# (c) 2021 Olivier Doaré
#
# olivier.doare@ensta-paris.fr
#
# -------------------------

# Note : this Python scrip uses cell mode of the Vscode extension
# (cells begin with #%%)

#%% Import Packages

# Add to path the parent directory in order to use the local measpy 
# Comment in order to use the installed version (e.g. via pip install)
# import sys
# sys.path.insert(0, "..")

from unyt import Unit
import measpy as mp
from measpy.audio import audio_run_measurement, audio_get_devices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn')

%matplotlib auto


#%% Get the list of audio devices present on the system

l=audio_get_devices()
print(l)

# measpy wants the input and output devices as strings
# On Ubuntu, 'default' corresponds to the main input and output

indev = 'default'
outdev = 'default'
indev = 'Analog (1+2) (Multiface Analog (1+2))'
outdev = 'Analog (1+2) (Multiface Analog (1+2))'


#%% Define and run an audio measurement
#
# This is an example
#
# Outputs a logarithmic sweep at output 1 of 5 seconds duration
# (with a fade in and out of 10 samples at begining and end)
#
# We want to measure a pressure at input 1  (microphone)
# and an acceleration at input 2 (accelerometer)
#
# Pressure calibration is 2V/Pa
# Acceleration calibration is 01.V/Pa
#
# When 5V is sent to the line input of the soundcard, the sample value = 1
# Hence the 0dBFS (zero dB full scale) is equal to 5V
# This has to be measured for instance by sending a know signal
# with an external signal generator

M1 = mp.Measurement(out_sig='logsweep',
                    fs = 44100,
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['Pressure here','Acceleration there'],
                    in_cal=[2.0,0.1],
                    in_unit=['Pa','meter/second**2'],
                    in_dbfs=[5.0,5.0],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=5,
                    in_device=indev,
                    out_device=outdev,
                    io_sync=0,
                    out_amp=0.5)

# Run the measurement
audio_run_measurement(M1)

# Plot the data
M1.plot()

#%% Test save and load measurement data
# 
# A measurement can be saved in three different formats
M1.to_jsonwav('jtest')
M1.to_csvwav('ctest')
M1.to_pickle('test.pck')

# %%
# Here, we load the data into a new measurement object
M2 = mp.Measurement.from_csvwav('test')
#M2 = mp.Measurement.from_pickle('test.mpk')

# %% Basic signal manipulation and plotting

# A measurement stores its data as a dictionnary of signals
# into the data property. 
# 
sig1=M2.data['In1'] # This is the pressure signal
sig2=M2.data['In2'] # This is the acceleration signal

# Let us plot the first signal
# (the Signal.plot() method returns an axes object)
a = sig1.plot(lw=0.5)

# %%
# Note that most line/symbols properties recognized by matplotlib
# can be passed to the measpy.Signal.plot method as optionnal arguments
# Here, lw is the matplotlib argument to specify the line width
#

# To plot the smoothed rms of sig1 on the same axes, in black,
# we also set the ax argument to plot on the same plot :
sig1.rms_smooth(nperseg=1024).plot(ax=a,lw=2,c='k')
(-sig1.rms_smooth(nperseg=1024)).plot(ax=a,lw=2,c='k')

# %% 
# We might be interested in plotting the smoothed RMS of the signal
# converted in dB SPL. The acoustic reference pressure is hard coded
# into measpy (variable mp.PREF=20e-6 Pascals)
print(mp.PREF)
a2 = sig1.rms_smooth().dB_SPL().plot()

# %%
# Using the method dB() and giving as argument mp.PREF is equivalent
sig1.rms_smooth().dB(mp.PREF).plot(ax=a2)


# %%
# The dB method takes care of the units, if we do the same with
# the second input, which has the dimension of m/s**2, there should
# be an error

sig2.rms_smooth().dB(mp.PREF).plot(ax=a2)

# %%
# Other reference values are hard coded

print(mp.DBUREF)
print(mp.DBVREF)

# %%
