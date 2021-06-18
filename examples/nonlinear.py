#%%

# We add to the search path the parent directory
# in order to work and test on the local branch
import sys
sys.path.insert(0,'..')

from unyt import Unit
import measpy as mp
from measpy.ni import ni_run_measurement, ni_get_devices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

plt.style.use('seaborn')
%matplotlib auto

#%%

M = mp.Measurement.from_csvwav('measpy/examples/test')
sp = M.data['In1'].tfe_farina(M.out_sig_freqs)
sp.plot()

gd = sp.group_delay()
gdc = gd.values[(gd.freqs > 20) and (gd.freqs < 20000)]
delay = np.mean(gdc)
a = gd.plot(dby=False,plotphase=False)
a.plot(M.out_sig_freqs,[delay,delay])


G=sp.irfft()
a=G.plot()

# Number of harmonics
Nh = 5

t1=0.1919
L = M.dur/np.log(M.out_sig_freqs[1]/M.out_sig_freqs[0])
dt=np.zeros(Nh)
dn=np.zeros(Nh)
tl = np.zeros(Nh)

for ii in range(Nh):
    dn[ii] = L*np.log(ii+1)*sp.fs
    dt[ii] = L*np.log(ii+1)

print (dt)

ns = np.round((G.dur-dt+t1)*G.fs)
ts = np.take(G.time,list(map(int,list(ns))),mode='wrap')
maxG = np.max(np.abs(G.values))
a.plot(ts,np.zeros_like(ts),'*')

# %%
