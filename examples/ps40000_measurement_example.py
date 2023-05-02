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

# Define the measurement
M = mp.Measurement( device_type='pico',
                    fs = 10000, # Sampling frequency
                    out_sig=None,   # We don't send any output, only recording
                    in_map=[1,2],   # Channel A is first input, channel B 2nd
                    in_desc=['Voltage input A','Voltage input B'], # Input descriptions
                    in_cal=[1.0,1.0], # Input calibrations
                    in_unit=['V','V'], # Input units
                    in_dbfs=[1.0,1.0], # Input dbfs (in general, should be 1.0 except for soundcard inputs)
                    extrat=[0,0], # No extra time before and after measurement
                    dur=2, # Measurement duration
                    in_device='default', #Not used with ps4000
                    out_device='default', # Not used with ps4000
                    in_range=['2V','2V'], # Select card input ranges
                    upsampling_factor=20, # The actual recording is made at higher frequency, befor decimation
                    in_coupling=['dc','dc']) # Input coupling calibrations

# Run the measurement
ps4000_run_measurement(M)

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
