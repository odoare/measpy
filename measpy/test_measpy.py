#%%

import measpyaudio as ma
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('dark_background')
%matplotlib auto

#%%
M1 = ma.Measurement(out_sig='noise',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['In1','In2'],
                    in_cal=[1.0,1.0],
                    in_unit=['V','V'],
                    in_dbfs=[1.0,1.0],
                    extrat=[1.0,2.0],
                    out_sig_fades=[500,500],
                    dur=2)
M1.run_measurement()
M1.to_jsonwav('j1')
M1.to_csvwav('c1')
M1.to_pickle('1.pck')

# %%
M3=ma.load_measurement_from_pickle('M1.pck')
plt.plot(M3.t,M3.x)
# %%
