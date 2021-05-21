#%% Load modules

from unyt import Unit
import measpy.signal as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
%matplotlib auto

#%% Create testing signals

fs=44100.0
f1=1000
f2=2000
f3=2000

# s1 in Pascals
s1=mp.Signal(
    raw=np.sin(2*np.pi*f1*np.arange(0,1,1/fs)),
    fs=fs,
    unit='Pa',
    dbfs=5.0,
    cal=3,
    desc='Signal 1')

# s2 in N/cm**2 (different, but compatible with Pascals)
s2=mp.Signal(
    raw=np.sin(2*np.pi*f2*np.arange(0,1,1/fs)),
    fs=fs,
    unit='N/cm**2',
    dbfs=1,
    cal=0.3,
    desc='Signal 2')

# s3 is in m/s
s3=mp.Signal(
    raw=np.sin(2*np.pi*f2*np.arange(0,1,1/fs)),
    fs=fs,
    unit='m/s',
    dbfs=1,
    cal=0.3,
    desc='Signal 3')


# %% Test operations on signals

sadd = s1+s2
print(sadd)

#s2+s1 has the dimension of s2
sadd2 = s2+s1
print(sadd2)

print('Max value of s1+s2')
print(max(sadd.values))
print('Max value of s2+s1')
print(max(sadd2.values))
print('Conversion of s2+s1 to the units of s1+s2') 
print(str(max(sadd2.unit_to(sadd.unit).values)))
print('Conversion of s2+s1 to the standard unit of this dimension') 
print(max(sadd2.unit_to(sadd.unit).values))

plt.figure(1)
s1inv=~s1
print(s1inv)
s1inv.plot()

plt.figure(2)
smul = s1*s2
print(smul)
smul.plot()

plt.figure(3)
sdiv = s1/s2
print(sdiv)
sdiv.plot()


# %% Spectral data
sp1=(1000*s1+s2).rfft()
sp1.abs().plot(axestype='lin',ylabel1=str(sp1.unit))


# %%
