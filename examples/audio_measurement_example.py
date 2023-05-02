# Example of data acquisition task
# with an audio soundcard
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

#%% This python file uses cell mode (vscode python extension feature)

# Import Packages
# Add to path the parent directory in order to use the local measpy 
# Comment in order to use the global version (e.g. via pip install)
# import sys
# sys.path.insert(0, "..")

import measpy as mp
from measpy.audio import audio_run_measurement, audio_get_devices

#%% Get the list of audio devices present on the system
l=audio_get_devices()
print(l)

# measpy wants the input and output devices as strings
# On Ubuntu, 'default' corresponds to the main input and output

indev = 'default'
outdev = 'default'

# For example, if the card is a RME hdsp multiface card,
# it should appear like that on Linux
# indev = 'Analog (1+2) (Multiface Analog (1+2))'
# outdev = 'Analog (1+2) (Multiface Analog (1+2))'


#%% Define and run an audio measurement
#
# This is an example of typical audio measurement:
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
# When 5V is sent to the line input of the soundcard, the collected sample value = 1
# Hence the 0dBFS (zero dB full scale) is equal to 5V
# This has to be measured for instance by sending a known signal
# to the inputs with an external signal generator

M = mp.Measurement(out_sig='logsweep',
                   device_type='audio',
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
audio_run_measurement(M)

# Save the measurement as a pair of .csv (with properties) and .wav (with data) files
M.to_csvwav('my_pico_measurement')

# Load the measurement
M1 = mp.Measurement.from_csvwav('my_pico_measurement')

# Plot the acquired data
M1.plot()

# Plot an individual signal (channel 1)
M1.data['In1'].plot()

# Plot the Power spectral density of channel 2 signal 
# (Welch's method with windows of 2**14 points)
M1.data['In2'].psd(nperseg=2**14).plot()

# %%
