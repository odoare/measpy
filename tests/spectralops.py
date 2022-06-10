# ------------------------
#
# measpy - A Python package to perform measurements an signal analysis
#
# (c) 2021 Olivier Doaré
#
# olivier.doare@ensta-paris.fr
#
# -------------------------

# Test Spectral class

# Note : this Python script uses cell mode of the Vscode extension
# (cells begin with #%%)

#%% Import Packages

# Add to path the parent directory in to 
import sys
sys.path.insert(0, "..")

from unyt import Unit
import measpy as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
%matplotlib tk

#%% Compares division of two Spectral objects
S1=mp.Signal.noise(fs=48000,unit='m')
S2=mp.Signal.noise(fs=48000,unit='s')

Sp1=S1.rfft()
Sp2=S2.rfft()

A=Sp1/Sp2
Atest = Sp1.similar()
Atest.values = Sp1.values/Sp2.values
Atest.unit = Sp1.unit/Sp2.unit
a=A.plot()
Atest.plot(ax=a)

(A-Atest).plot(dby=False)

print(A.unit)
print(Atest.unit)
# %% Test of fft normalization
S1=mp.Signal.noise(fs=48000,unit='m')
Sf1 = S1.fft(norm="ortho")
Ef = sum(abs(Sf1.values)**2)
E = sum(abs(S1.values)**2)
print(E)
print(Ef)
S2 = Sf1.ifft()

S2.plot()
S1.plot()
# %%
