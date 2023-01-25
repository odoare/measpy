#%%

# Create and run a measurement with a ps4000 device

import sys
sys.path.insert(0, "./measpy")

import measpy as mp
from measpy.pico import ps4000_run_measurement

# Define the measurement
M = mp.Measurement( out_sig=None,   # We don't send any output, only recording
                    in_map=[1,2],   # Channel A is first input, channel B 2nd
                    in_desc=['Voltage input A','Voltage input B'], # Input descriptions
                    in_cal=[1.0,1.0], # Input calibrations
                    in_unit=['V','V'], # Input units
                    in_dbfs=[1.0,1.0], # Input dbfs (in general, should be 1.0 except for soundcard inputs)
                    extrat=[0,0], # No extra time before and after measurement
                    dur=5, # Measurement duration
                    in_device='default', #Not used with ps2000
                    out_device='default', # Not used with ps2000
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
