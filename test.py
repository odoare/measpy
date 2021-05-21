#%%

#from measpy import measpyaudio as ma
#from pint.unit import Unit
from unyt import Unit
import measpy.audio as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

# plt.style.use('classic')
# matplotlib.use('TkAgg')
%matplotlib auto

#%% Define and run a measurement
M1 = ma.Measurement(out_sig='logsweep',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['Pressure here','Acceleration there'],
                    in_cal=[1.0,1.0],
                    in_unit=['Pa','meter/second**2'],
                    in_dbfs=[1.0,1.0],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=5,
                    in_device='default',
                    out_device='default')
M1.run_measurement()
M1.plot_with_cal()

#%% Save in three different formats
M1.to_jsonwav('jtest')
M1.to_csvwav('jtest')
M1.to_pickle('test.pck')

# %%

M1 = ma.Measurement(out_sig='logsweep',
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
wa=ma.Weighting.from_csv('measpy/data/dBA.csv')
wc=ma.Weighting.from_csv('measpy/data/dBC.csv')
sp=ma.Spectral(values=np.arange(44100),fs=44100)
spa=sp.similar(w=wa,desc='dBA')
spc=sp.similar(w=wc,desc='dBC')

plt.figure(1)
spa.plot(axestype='logdb')
spc.plot(axestype='logdb')
plt.plot(wa.f,wa.adb,'*')
plt.plot(wc.f,wc.adb,'*')
plt.title('dBA and dBC weighting functions')
plt.xlim([10,20000])


# %% Test measurement to weighting
m=ma.Measurement.from_pickle('test.mpk')
sig1=m.data['In1']
plt.figure(1)
sig1.plot()

sp1=sig1.tfe_farina(m.out_sig_freqs)
plt.figure(2)
sp1.plot()

w1=sp1.abs().nth_oct_smooth_to_weight(24)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(w1.f,20*np.log10(np.abs(w1.a)),'*')
plt.subplot(2,1,2)
plt.plot(w1.f,np.unwrap(np.angle(w1.a)),'*')

sp1s=sp1.abs().nth_oct_smooth(24)
plt.figure(2)
sp1s.plot()

# %% Test smooth and dBSPL of signals
m=ma.Measurement.from_pickle('test.mpk')
sig1=m.data['In1'].rms_smooth(nperseg=4096)
print(sig1.desc)
sig2=sig1.dB(0.1*ma.ms.PREF)
print(sig2.desc)
sig2.plot()

# %% Test impulse responses
m=ma.Measurement.from_pickle('test.mpk')
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
