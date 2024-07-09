#%% 

# Uncomment if importing from local measpy directory
# import sys
# sys.path.insert(0, "..")

import measpy as mp
import matplotlib.pyplot as plt
%matplotlib tk

# A,C,M weighting

A = mp.Spectral(fs=44100,dur=10).similar(w=mp.WDBA,desc='A-Weighting')
C = mp.Spectral(fs=44100,dur=10).similar(w=mp.WDBC,desc='C-Weighting')
M = mp.Spectral(fs=44100,dur=10).similar(w=mp.WDBM,desc='M-Weighting')

a = A.plot(plot_phase=False)
C.plot(ax=a,plot_phase=False)
M.plot(ax=a,plot_phase=False)
plt.legend()
a.set_xlim((20,20000))
a.set_ylim((-50,20))
a.set_ylabel("dB")
plt.grid()
plt.show()

# %%

# Creates a white noise signal (with default values
# of 1sec duration, 44100Hz sampling frequency, dimensionless)

s = mp.Signal.noise()

# Plots the 

a = s.rfft().plot(plot_phase=False,label='White noise')
s.rfft().apply_dBA().plot(ax=a,plot_phase=False,label='White noise, dBA applied')
a.set_xlim((20,20000))
a.set_ylim((-20,80))
plt.legend()
plt.show()

# %%
