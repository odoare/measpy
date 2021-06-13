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

# Add to path the parent directory in to 
import sys
sys.path.insert(0, "..")

from unyt import Unit
import measpy as mp
from measpy.audio import audio_run_measurement, audio_get_devices


import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn')

%matplotlib auto


#%% Get the list of audio devices present on the system

audio_get_devices()

# measpy wants the input and output devices as strings
# On Ubuntu, 'default' corresponds to the main input and output

indev = 'default'
outdev = 'default'


#%% Define and run an audio measurement

# Outputs a logarithmic sweep at output 1
# (with a fade in and out of 10 samples at begining and end)
#
# measure a pressure at input 1 and an acceleration at input 2
#
# Pressure calibration is 2V/Pa
# Acceleration calibration is 01.V/Pa

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
audio_run_measurement(M2)

# Plot the data
M1.plot()

#%% Save in three different formats
M1.to_jsonwav('jtest')
M1.to_csvwav('ctest')
M1.to_pickle('test.pck')

# %% Load the data into a new measurement object

M2 = mp.Measurement.from_csvwav('ctest')

# %% Basic signal manipulation and plotting

# A measurement stores its data as a dictionnary of signals
sig1=M2.data['In1'] # This is the pressure
sig2=M2.data['In2'] # This is the acceleration

# Let us plot the first signal
# (the Signal.plot() method returns an axes object)
a = sig1.plot(lw=0.5)

# To plot the smoothed rms of sig1 on the same axes, in black
sig1.rms_smooth(nperseg=1024).plot(ax=a,lw=2,c='k')
(-sig1.rms_smooth(nperseg=1024)).plot(ax=a,lw=2,c='k')


a2 = sig1.rms_smooth().dB_SPL().plot()

sp1=sig1.tfe_farina(m.out_sig_freqs)
plt.figure(2)
sp1.plot()

w1=sp1.nth_oct_smooth_to_weight_complex(24)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(w1.freqs,w1.adb,'*')
plt.subplot(2,1,2)
plt.plot(w1.freqs,w1.phase,'*')

sp1s=sp1.nth_oct_smooth_complex(24)
plt.figure(2)
sp1s.plot()

# %% Test smooth and dBSPL of signals
m=mp.Measurement.from_pickle('test.mpk')
sig1=m.data['In1'].rms_smooth(nperseg=4096)
print(sig1.desc)
sig2=sig1.dB(mp.PREF)
print(sig2.desc)
sig2.plot()

# %% Test impulse responses
m=mp.Measurement.from_pickle('test.mpk')
Gap = m.data['In1'].tfe_farina(m.out_sig_freqs).filterout([20,20000]).irfft()
Gap0 = m.data['In1'].tfe_farina(m.out_sig_freqs).irfft()
Gap0.plot()
Gap.plot()

# %%
import unyt as u

a = 1*u.metre
# %% Calibration carte Behringer

M1 = ma.Measurement(out_sig='logsweep',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[0.656],
                    in_map=[1],
                    in_desc=['Input voltage'],
                    in_cal=[1.0],
                    in_unit=['V'],
                    in_dbfs=[1.685,1.685],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=5,
                    in_device='default',
                    out_device='default')
M1.run_measurement()

M1.data['In1'].plot()
# %% Test a pure input measurement

M1 = mp.Measurement(out_sig=None,
                    in_map=[1],
                    in_desc=['Input voltage'],
                    in_cal=[1.0],
                    in_unit=['V'],
                    in_dbfs=[1.685,1.685],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=5,
                    in_device='default')
M1.run_measurement()

# %%

# %% Test carte ni

M1 = mp.Measurement(out_sig='logsweep',
                    fs=96000,
                    out_sig_freqs=[1.0,22000],
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['Input voltage 1','Input voltage 2'],
                    in_cal=[1.0,1.0],
                    in_unit=['V','V'],
                    in_dbfs=[1.0,1.0],
                    extrat=[0,0],
                    out_sig_fades=[0,0],
                    dur=5,
                    io_sync=1,
                    in_device='Dev3',
                    out_device='Dev3')
M1.run_measurement()

#%% Analyses carte ni
plt.figure(1)
M1.data['In1'].plot(linetype='o')
M1.data['Out1'].plot(linetype='*')

plt.figure(2)
M1.data['In1'].tfe_farina(M1.out_sig_freqs).plot()
M1.data['In1'].tfe_welch(M1.data['Out1']).plot()

plt.figure(3)
M1.data['In1'].tfe_farina(M1.out_sig_freqs).irfft().plot()

plt.figure(3)
M1.data['In1'].tfe_welch(M1.data['Out1']).irfft().plot()

plt.figure(4)
M1.data['Out1'].tfe_welch(M1.data['Out1']).plot()
M1.data['Out1'].tfe_farina(M1.out_sig_freqs).plot()

plt.figure(5)
M1.data['Out1'].tfe_welch(M1.data['Out1']).irfft().plot()
M1.data['Out1'].tfe_farina(M1.out_sig_freqs).irfft().plot()

plt.figure(6)
(M1.data['Out1']*(1*Unit('V'))+M1.data['Out1']*M1.data['Out1']).tfe_farina(M1.out_sig_freqs).irfft().plot()

# %%
