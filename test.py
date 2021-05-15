#%%

#from measpy import measpyaudio as ma
from pint.unit import Unit
import measpy.audio as ma
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

# plt.style.use('classic')
# matplotlib.use('TkAgg')
# %matplotlib auto

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
                    out_sig_fades=[1000,10000],
                    dur=5)
M1.run_measurement()
M1.plot_with_cal()

#%% Save in three different formats
M1.to_jsonwav('j1')
M1.to_csvwav('c1')
M1.to_pickle('1.pck')

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
sp=ma.Spectral(x=np.arange(44100),fs=44100)
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

w1=sp1.abs().nth_oct_smooth_to_weight(12)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(w1.f,w1.adb,'*')

# %% Test smooth and dBSPL of signals
m=ma.Measurement.from_pickle('test.mpk')
sig1=m.data['In1'].rms_smooth(l=4096)
print(sig1.desc)
sig2=sig1.dB(0.1*ma.ms.PREF)
print(sig2.desc)
sig2.plot()
# %%
