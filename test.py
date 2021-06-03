#%%

#from measpy import measpyaudio as ma
#from pint.unit import Unit
from unyt import Unit
import measpy as mp
from measpy.audio import audio_run_measurement, audio_get_devices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

plt.style.use('seaborn')
# matplotlib.use('TkAgg')
%matplotlib auto

#%% Define and run a measurement
M1 = mp.Measurement(out_sig='logsweep',
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
                    in_device='Default',
                    out_device='Default',
                    io_sync=1,
                    out_amp=0.5)
audio_run_measurement(M1)
M1.plot()

#%% Save in three different formats
M1.to_jsonwav('jtest')
M1.to_csvwav('jtest')
M1.to_pickle('test.pck')

# %%

M1 = mp.Measurement(out_sig='logsweep',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['V'],
                    in_dbfs=[1.0],
                    extrat=[0.0,0.0],
                    out_sig_fades=[10,10],
                    dur=5)
M1.run_measurement()
M1.plot_with_cal()

# %% Test weightings
wa=mp.Weighting.from_csv('measpy/data/dBA.csv')
wc=mp.Weighting.from_csv('measpy/data/dBC.csv')
sp=mp.Spectral(values=np.arange(44100),fs=44100)
spa=sp.similar(w=wa,desc='dBA')
spc=sp.similar(w=wc,desc='dBC')

plt.figure(1)
spa.plot(axestype='logdb')
spc.plot(axestype='logdb')
plt.plot(wa.freqs,wa.adb,'*')
plt.plot(wc.freqs,wc.adb,'*')
plt.title('dBA and dBC weighting functions')
plt.xlim([10,20000])


# %% Test measurement to weighting
m=mp.Measurement.from_pickle('test.mpk')
sig1=m.data['In1']
plt.figure(1)
sig1.plot()

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
