#%%

# We add to the search path the parent directory
# in order to work and test on the local branch
import sys
sys.path.insert(0,'../')

from unyt import Unit
import measpy as mp
from measpy.ni import ni_run_measurement, ni_get_devices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

plt.style.use('seaborn')
%matplotlib auto

#%% List devices
print(ni_get_devices().device_names)

indev='Dev2'
outdev='Dev2'

# %% Do measurement (One output, four inputs)
M1 = mp.Measurement(out_sig='logsweep',
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
                    io_sync=1,
                    out_amp=0.5)
ni_run_measurement(M1)
M1.plot()

# %%
S=M1.tfe()
S['In1'].plot()
S['In2'].plot()
S['In3'].plot()
S['In4'].plot()

# %%
