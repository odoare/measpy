#%%
# We add to the search path the parent directory
# in order to work and test on the local branch
# import sys
# sys.path.insert(0,'../')

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

#%% List the devices present in the system
l=audio_get_devices()
print(l)
indev = 'Analog (1+2) (Multiface Analog (1+2))'
outdev = 'Analog (1+2) (Multiface Analog (1+2))'

#%% Define and run a measurement
M1 = mp.Measurement(out_sig='logsweep',
                    out_map=[1],
                    out_desc=['Out Left'],
                    out_dbfs=[1.0],
                    in_map=[1],
                    in_desc=['Microphone pressure Left Event out'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=10,
                    in_device=indev,
                    out_device=outdev,
                    io_sync=0,
                    out_amp=0.5)
audio_run_measurement(M1)
M1.plot()

#%% Define and run a measurement
M2 = mp.Measurement(out_sig='logsweep',
                    out_map=[2],
                    out_desc=['Out Right'],
                    out_dbfs=[1.0],
                    in_map=[1],
                    in_desc=['Microphone pressure Right Event out'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0],
                    extrat=[0,0],
                    out_sig_fades=[10,10],
                    dur=10,
                    in_device=indev,
                    out_device=outdev,
                    io_sync=0,
                    out_amp=0.5)
audio_run_measurement(M2)
M2.plot()

M1.to_csvwav('left_event_out')
M2.to_csvwav('right_event_out')

# %%
