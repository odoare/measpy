#%%

# We add to the search path the parent directory
# in order to work and test on the local branch
import sys
sys.path.insert(0,'..')

from unyt import Unit
import measpy as mp
# from measpy.ni import ni_run_measurement, ni_get_devices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

plt.style.use('seaborn')
%matplotlib auto

#%%

M = mp.Measurement.from_csvwav('test')

#%%

sp = M.data['In1'].tfe_farina(M.out_sig_freqs)
a1=sp.plot(plotphase=False)

gd = sp.group_delay()
gdc = gd.values[(gd.freqs > 100)&(gd.freqs < 1000)]
delay = np.mean(gdc)
a = gd.plot(dby=False,plotphase=False)
a.plot(M.out_sig_freqs,[delay,delay])

G=sp.irfft()
a2=G.plot()

# Number of harmonics
Nh = 4

t1=0.19
l = 2**13
#t1=delay
# t1=0.18
L = M.dur/np.log(M.out_sig_freqs[1]/M.out_sig_freqs[0])
dt=np.zeros(Nh)
dn=np.zeros(Nh)
tl = np.zeros(Nh)

for ii in range(Nh):
    dn[ii] = L*np.log(ii+1)*sp.fs
    dt[ii] = L*np.log(ii+1)

print (dt)

ns = np.round((G.dur-dt+t1)*G.fs)#-0.5*l
ts = np.take(G.time,list(map(int,list(ns))),mode='wrap')
maxG = np.max(np.abs(G.values))
a2.plot(ts,np.zeros_like(ts),'*')

#%%
Gnl = {}
Hnl = {}
t = np.linspace(0,l/G.fs,l)
for ii in range(Nh):
    Gnl[ii]=G.similar(
        values=np.take(G.values,list(range(int(ns[ii]),int(ns[ii]+l))),mode='wrap'),
        desc = 'harmonic '+str(ii)
        )
    Gnl[ii].plot(ax=a2)
    Hnl[ii]=Gnl[ii].rfft().nth_oct_smooth_complex(6).filterout(M.out_sig_freqs)
    Hnl[ii].plot(ax=a1,plotphase=False,label='Harmonic '+str(ii))
a1.set_xlim([20,20000])
a1.legend()

# %%
