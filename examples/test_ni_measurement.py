#%%
# We add to the search path the parent directory
# in order to work and test on the local branch
import sys
sys.path.insert(0,'../')

from unyt import Unit
import measpy as mp
from measpy.ni import ni_run_measurement, ni_get_devices

import matplotlib.pyplot as plt

#plt.style.use('dark_background')
plt.style.use('seaborn')
%matplotlib auto

#%% List devices
print(ni_get_devices().device_names)

indev='myDAQ2'
outdev='myDAQ2'

# %% Do measurement (One output, two inputs)
M1 = mp.Measurement(out_sig='logsweep',
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
ni_run_measurement(M1)
M1.plot()

# %% Frequency response functions
S=M1.tfe()
a=S['In1'].plot()
S['In2'].plot(ax=a)

b=M1.data['Out1'].plot()
M1.data['In1'].plot(ax=b)
M1.data['In2'].plot(ax=b)

# %% Do measurement (One output, four inputs at 200kHz)
M1 = mp.Measurement(out_sig='logsweep',
                    fs=200000,
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2,3,4],
                    in_desc=['Input 1','Input 2','Input 3','Input 4'],
                    in_cal=[1.0,1.0,1.0,1.0],
                    in_unit=['V','V','V','V'],
                    in_dbfs=[1.0,1.0,1.0,1.0],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=5,
                    in_device=indev,
                    out_device=outdev,
                    io_sync=0,
                    out_amp=0.5)
ni_run_measurement(M1)
M1.plot()

# %% Plotting of frequency response function
# and comparison with resampled signals

S=M1.tfe()
a=S['In1'].plot()
S['In2'].plot(ax=a)
S['In3'].plot(ax=a)
S['In4'].plot(ax=a)

sp1=M1.data['In1'].resample(48000).tfe_farina(M1.out_sig_freqs)
sp2=M1.data['In2'].resample(48000).tfe_farina(M1.out_sig_freqs)
sp3=M1.data['In3'].resample(48000).tfe_farina(M1.out_sig_freqs)
sp4=M1.data['In4'].resample(48000).tfe_farina(M1.out_sig_freqs)
sp1.plot(ax=a,linestyle='--')
sp2.plot(ax=a,linestyle='--')
sp3.plot(ax=a,linestyle='--')
sp4.plot(ax=a,linestyle='--')
a[1].legend()

b=M1.data['Out1'].resample(48000).plot()
M1.data['In1'].resample(48000).plot(ax=b)
M1.data['In2'].resample(48000).plot(ax=b)
M1.data['In3'].resample(48000).plot(ax=b)
M1.data['In4'].resample(48000).plot(ax=b)

# %%
