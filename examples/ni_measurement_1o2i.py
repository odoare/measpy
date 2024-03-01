# Example of data acquisition task
# with a Native Instrument device
#
# Devices tested:
# - 4461
# - 4431
# - 625x
# - Mydaq
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
from measpy.ni import ni_run_measurement, ni_get_devices

#%% List devices
# This prints a list of identifiers for the devices present in the system
sysdevs = ni_get_devices().device_names
print(sysdevs)

# We choose the first fresent device
indev=outdev=sysdevs[0]



#%% Define and run a measurement with NI cards
#
# This is an example of dynamic measurement using a NI card:
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
# Contrary to audio cards, there is a 1/1 correspondence between
# input volts and actual sample value. Hence dbfs is not necessary
# It will there defaults to 1.

# We first create the output signal
so = mp.Signal.log_sweep(fs=44100,freq_min=20,freq_max=20000,dur=5)

# Two (empty) input signals are then created
si1 = mp.Signal(unit='Pa',cal=2,desc='Pressure here')
si2 = mp.Signal(unit='m/s**2',cal=0.1, desc='Acceleration there')

M = mp.Measurement(device_type='ni',
                    fs = 44100,
                    out_map=[1],
                    out_sig=[so],
                    in_map=[1,2],
                    in_sig=[si1,si2],
                    dur=5,
                    in_device=indev,
                    out_device=outdev)

# Run the measurement
ni_run_measurement(M)

# Save the measurement as directory containing all data
# This command creates the directory containing:
#   - A README file indicating the measpy version
#   - A params.csv containing the measurement info
#   - Pairs of csv/wav files for each signal (output and inputs)

M.to_dir('my_audio_measurement')

# Load the measurement into the Measurement object M1
M1 = mp.Measurement.from_dir('my_audio_measurement')

# Plot the acquired signals on the same graph
a = M1.in_sig[0].plot()
M1.in_sig[1].plot(ax=a)

# Plot an individual signal (channel 1)
M1.data['In1'].plot()

# Plot the Power spectral density of channel 2 signal 
# (Welch's method with windows of 2**14 points)
M1.in_sig[1].psd(nperseg=2**14).plot()


