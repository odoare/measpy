# Example of data acquisition task
# with a Picoscope of the ps4000 series
#
# (tested with 4461) 
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
from measpy.pico import ps4000_run_measurement

#%%
s1=mp.Signal(unit='Pa')
s2=mp.Signal(unit='Pa')
M = mp.Measurement(in_sig=[s1,s2],
                   fs=48000,
                   in_map=[1,2],
                   device_type='pico',
                   dur=2)

# Run the measurement
ps4000_run_measurement(M)

# Save the measurement in a directory
# If the directory exists, a renaming
# strategy is adopted
# The to_dir methods returns the actual
# name of the directory
d = M.to_dir('my_pico_measurement')

# Load the measurement
M1 = mp.Measurement.from_dir(d)

# Plot the acquired signals on the same graph
a = M1.in_sig[0].plot()
M1.in_sig[1].plot(ax=a)

# Plot the Power spectral density of channel 2 signal 
# (Welch's method with windows of 2**14 points)
M1.in_sig[1].psd(nperseg=2**14).plot()
