
#%% This python file uses cell mode (vscode python extension feature)

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

# Import Packages
# Add to path the parent directory in order to use the local measpy 
# Not needed if using the global version (e.g. via pip install)
import sys
sys.path.insert(0, "../..")

import measpy as mp
from measpy.ni import ni_run_synced_measurement, ni_get_devices

#%matplotlib qt

#%% List devices
# This prints a list of identifiers for the devices present in the system
sysdevs = ni_get_devices().device_names
print(sysdevs)

# We choose the first present device
indev=outdev=sysdevs[0]

#%% Define and run a measurement with NI cards
#
# This is an example of dynamic measurement using a NI card,
# demonstrating the IO synchronization, using a cable loop
#
# To run this example, connect the first output to the first input
#
# It outputs a five sec. noise at the first output
# (with a fade in and out of 10 samples at begining and end)
#
# We then measure the sent voltage. Timelag between input
# and output is then estimated, and input signal is shifted
# to re-sync with output signal
#
# Contrary to audio cards, there is a 1/1 correspondence between
# input volts and actual sample value. Hence dbfs is not necessary
# It will there defaults to 1.

# We specify sampling frequency and duration
FS = 100000
DUR = 2

# We first create the output signal
s_out = mp.Signal.noise(fs=FS,freq_min=20,freq_max=20000,dur=DUR)

# An empty input signals is then created
s_in = mp.Signal(unit='V',desc='Input voltage')

# Measurement creation
# Note that the duration is optional below because
# the task duration is imposed by the output signal duration
# If different or not specified, the measurement duration
# is modified to correspond to the output signal duration
M = mp.Measurement(device_type='ni',
                    in_sig=[s_in],
                    out_sig=[s_out],
                    fs = 44100,
                    out_map=[1],
                    in_map=[1],
                    dur=DUR,
                    in_device=indev,
                    out_device=outdev)

# The measurement is now run with synchronization
# By default, 1 sec of silence is added before and at the end of the
# output signal. Next, measurement task is performed, and the time
# delay between input and output is estimated using timelag method.
# All input signals are then shifted and cut to re-synchronize
# 
# The silence duration, input and output channels can optionally be
# specified.
#
# ni_run_synced_measurement returns the measured time delay as float

d = ni_run_synced_measurement(M)

print("Measured delay in seconds: "+str(d))

# Save the measurement as directory containing all data
# This command creates the directory containing:
#   - A README file indicating the measpy version
#   - A params.csv containing the measurement info
#   - Pairs of csv/wav files for each signal (output and inputs)

d=M.to_dir('my_ni_measurement')

# Load the measurement into the Measurement object M1
M1 = mp.Measurement.from_dir(d)

# Plot the signals on the same graph
a = M1.in_sig[0].plot()
M1.out_sig[0].plot(ax=a)

# Plot the Power spectral density of channel 2 signal 
# (Welch's method with windows of 2**14 points)
# If input and outputs are plugged correctly, it should
# represent a transfer function at 0dB amplitude and zero phase
M1.in_sig[0].tfe_welch(M1.out_sig[0],nperseg=2**14).plot()

# %%
M1.in_sig[0].timelag(M1.out_sig[0])
# %%
