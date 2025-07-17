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
import sys
sys.path.insert(0, "../..")

import measpy as mp
from measpy.ni import ni_run_measurement, ni_get_devices

#%% List devices
# This prints a list of identifiers for the devices present in the system
sysdevs = ni_get_devices().device_names
print(sysdevs)

# We choose the first fresent device
indev=outdev=sysdevs[0]

#%% Define and run a measurement with NI cards

# Two (empty) input signals are then created
si1 = mp.Signal(desc='Channel 1')


M = mp.Measurement(device_type='ni',
                    fs = 44100,
                    in_map=[1,2],
                    in_sig=si1,
                    dur=5,
                    in_device=indev)

# Run the measurement
ni_run_measurement(M)

# Save the measurement as directory containing all data
# This command creates the directory containing:
#   - A README file indicating the measpy version
#   - A para[si1,si2]ms.csv containing the measurement info
#   - Pairs of csv/wav files for each signal (output and inputs)

d=M.to_dir('my_audio_measurement')

# Load the measurement into the Measurement object M1
M1 = mp.Measurement.from_dir(d)

# Plot the acquired signals on the same graph
a = M1.in_sig[0].plot()
M1.in_sig[1].plot(ax=a)

# Plot an individual signal (channel 1)
M1.in_sig[0].plot()

# Plot the Power spectral density of channel 2 signal 
# (Welch's method with windows of 2**14 points)
M1.in_sig[1].psd(nperseg=2**14).plot()

# %%
M1 = mp.Measurement.from_hdf5('hdf5file.h5')
# Plot the acquired signals on the same graph
a = M1.in_sig[0].plot()
M1.in_sig[1].plot(ax=a)

# %%
