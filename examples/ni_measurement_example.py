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

# %% Do measurement (One output, two inputs)
M = mp.Measurement(out_sig='logsweep',
                    device_type='ni',
                    fs=48000,
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['Input 1','Input 2'],
                    in_cal=[1.0,1.],
                    in_unit=['V','V'],
                    in_dbfs=[1.0,1.0],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=5,
                    in_device=indev,
                    out_device=outdev,
                    io_sync=0,
                    out_amp=0.5)

# Run the data acquisition
ni_run_measurement(M)

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
