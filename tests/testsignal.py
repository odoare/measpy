# ------------------------
#
# measpy - A Python package to perform measurements an signal analysis
#
# (c) 2021 Olivier Doaré
#
# olivier.doare@ensta-paris.fr
#
# -------------------------

# Examples of signal manipulation

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
print('\n')

print('Max value of s1+s2 : ' + str(max(sadd.values)) + ' ' + str(sadd.unit) + '\n' )
print('Max value of s2+s1 : ' + str(max(sadd2.values)) + ' ' + str(sadd2.unit) + '\n')
print('Conversion of s2+s1 to the units of s1+s2') 
sadd3 = sadd2.unit_to(sadd.unit)
print ('Max value : ' + str(max(sadd3.values)) + ' ' + str(sadd3.unit) + '\n')
print("Conversion of s2+s1 to the standard unit of it's dimension") 
print('Max value : ' + str(max(sadd2.unit_to_std().values)))

s1inv=~s1
print('Inverse of signal 1 :')
print(s1inv)
a=s1inv.plot()

smul = s1*s2
print('Product of signal 1 and signal 2 :')
print(smul)
print('Conversion of this signal to a standard unit')
print(smul.unit_to_std())
b=smul.plot()

sdiv = s1/s2
print(sdiv)
c=sdiv.plot()


# %%
