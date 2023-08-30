#%%

# Uncomment if importing from local measpy directory
import sys
sys.path.insert(0, "../../measpy")

import measpy as mp
from numpy import pi, sqrt
import matplotlib.pyplot as plt
%matplotlib tk

# A long logarithmic sweep signal is created
s = mp.Signal.log_sweep(dur=20,fs=96000)

# Non linear response
s1 = s + s*s + s*s*s
s1.desc = "Nonlinear response"

# Signals are plotted
a=s.plot()
s1.plot(ax=a)

# For one sinusoidal component in the Fourier space, we get:
# \sin(t)+\sin^2(t)+\sin^3(t)
# = -1/2 + (7/4) \sin(t) - (1/2) \cos(2t) - (1/4) \sin(3t)
#
# Amplitude of harmonic 0 : 20*log10(7/4) ~ 4.86
A0 = 4.86
phi0 = 0
# Amplitude of harmonic 1 : 20*log10(1/2) ~ -6.02 (and phase -pi/2)
A1 = -6.02
phi1 = -pi/2
# Amplitude of harmonic 2 : 20*log10(1/4) ~ -12.04 (and phase -pi)
A2 = -12.04
phi2 = pi

# THD
THD = 100*sqrt((1/2)**2+(1/4)**2)/sqrt((1/2)**2+(1/4)**2+(7/4)**2)

r = s1.harmonic_disto(win_max_length=2**18,debug_plot=False,freq_min=s.freq_min,freq_max=s.freq_max,nh=3)

# The frequency responses for each component,
# frequency aligned are in r[1]

a=r[1][0].plot(logx=True,unwrap_phase=False,unwrap_around=100,lw=3,label="Harminc 0")
for i in range(1,len(r[1])):
    a=r[1][i].plot(logx=True,ax=a,unwrap_phase=True,unwrap_around=100,lw=3,label="Harmonic "+str(i))   
a[0].set_xlim((s.freq_min,s.freq_max)) 
a[1].set_xlim((s.freq_min,s.freq_max))
a[0].plot((s.freq_min,s.freq_max),(A0,A0),ls='--',c='k')
a[0].plot((s.freq_min,s.freq_max),(A1,A1),ls='--',c='k')
a[0].plot((s.freq_min,s.freq_max),(A2,A2),ls='--',c='k')
a[1].plot((s.freq_min,s.freq_max),(phi0,phi0),ls='--',c='k',label='Theoretical')
a[1].plot((s.freq_min,s.freq_max),(phi1,phi1),ls='--',c='k')
a[1].plot((s.freq_min,s.freq_max),(phi2,phi2),ls='--',c='k')
a[0].set_ylim((-20,10)) 
a[1].set_ylim((-2*pi,2*pi)) 
plt.legend()

# Plot of the total harmonic distorsion
a=r[2].plot(plot_phase=False,dby=False,label="THD(%) calculated")
a.plot((s.freq_min,s.freq_max),(THD,THD),label="THD(%) theoretical")
a.set_xlim((s.freq_min,s.freq_max)) 
plt.legend()
# %%
