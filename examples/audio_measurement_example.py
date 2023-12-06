# Example of data acquisition task
# with an audio soundcard
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

#%% This python file uses cell mode (vscode python extension feature)

# Import Packages
# Add to path the parent directory in order to use the local measpy 
# Comment in order to use the global version (e.g. version installed via pip install)
# import sys
# sys.path.insert(0, "..")

import measpy as mp
from measpy.audio import audio_run_measurement_2, audio_get_devices

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
# This is an example of dynamic measurement using an audio card:
#
# Outputs a logarithmic sweep at output 1 of 5 seconds duration
# (with a fade in and out of 10 samples at begining and end)
#
# We want to measure a pressure at input 1  (microphone)
# and an acceleration at input 2 (accelerometer)
#
# Pressure calibration is 2V/Pa
# Acceleration calibration is 0.1V/(m/s^2)
#
# When 1.4V is sent to the line input of the soundcard, the collected sample value = 1
# Hence the 0dBFS (zero dB full scale) is equal to 1.4
# This has to be measured for instance by sending a known signal
# to the inputs with an external signal generator

# We first create the output signal
so = mp.Signal.log_sweep(fs=44100,freq_min=20,freq_max=20000,dur=5)

# Two (empty) input signals are then created
si1 = mp.Signal(unit='Pa',cal=2,dbfs=1.4,desc='Pressure here')
si2 = mp.Signal(unit='m/s**2',cal=0.1,dbfs=1.4, desc='Acceleration there')

M = mp.Measurement( device_type='audio',
                    fs = 44100,
                    in_sig=[si1,si2],
                    out_sig=[so],
                    out_map=[1],
                    in_map=[1,2],
                    dur=5,
                    in_device=indev,
                    out_device=outdev)

# Run the measurement
audio_run_measurement_2(M)

# Save the measurement as directory containing all data
# This command creates the directory containing:
#   - A README file indicating the measpy version
#   - A params.csv containing the measurement info
#   - Pairs of csv/wav files for each signal (output and inputs)

M.to_dir('my_audio_measurement')

# Load the measurement into the Measurement object M1
M1 = mp.Measurement.from_dir('my_audio_measurement')

#%%

# Plot the acquired signals on the same graph
a = M1.in_sig[0].plot()
M1.in_sig[1].plot(ax=a)

# Plot an individual signal (first input)
M1.in_sig[0].plot()

# Plot the Power spectral density of channel 2 signal 
# (Welch's method with windows of 2**14 points)
M1.in_sig[1].psd(nperseg=2**14).plot()



# %%
