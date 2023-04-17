# _tools.py
#
# Various tools for measpy
#
# (c) OD, 2021

import csv
import numpy as np

def csv_to_dict(filename):
    """ Conversion from a CSV (produced by the class Measurement) to a dict
          Default separator is (,)
          First row is the key string
          The value is a list
    """
    dd={}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dd[row[0]]=row[1:]
    return dd

def convl(fun,xx):
    if type(xx)==list:
        yy=list(map(fun,xx))
    else:
        yy=fun(xx) 
    return yy

def convl1(fun,xx):
    if type(xx)==list:
        yy=fun(xx[0])
    else:
        yy=fun(xx) 
    return yy

def add_step(a,b):
    return a+'\n -->'+b

def wrap(phase):
    """ Opposite of np.unwrap   
    """
    return np.mod((phase + np.pi), (2 * np.pi)) - np.pi

def unwrap_around_index(phase,n):
    """ Opposite of np.unwrap   
    """
    return np.hstack((np.unwrap(phase[n-1::-1])[::-1],np.unwrap(phase[n:])))

def smooth(in_array,l=20):
    ker = np.ones(l)/l
    return np.convolve(in_array,ker,mode='same')

def nth_octave_bands(n,fmin=5,fmax=20000):
    """ 1/nth octave band frequency range calculation """
    nmin = int(np.ceil(n*np.log2(fmin*10**-3)))
    nmax = int(np.ceil(n*np.log2(fmax*10**-3)))
    indices = range(nmin,nmax+1)
    f_centre = 1000 * (2**(np.array(indices)/n))
    f2 = 2**(1/n/2)
    f_upper = f_centre * f2
    f_lower = f_centre / f2
    return f_centre, f_lower, f_upper

def picv(long):
    """ Create a 1D-array of length long with a unitary peak in the middle """
    return np.hstack((np.zeros(long),1,np.zeros(long-1)))

def _create_time1(fs,dur):
    return np.linspace(0,dur-1/fs,int(round(dur*fs)))  # time axis

def _create_time2(fs,length):
    return np.linspace(0,(length-1)/fs,length)  # time axis

def create_time(fs,dur=None,length=None):
    if dur==None and length==None:
        raise Exception('dur=duration in s or length=number of samples must be specified.')
    if dur!=None and length!=None:
        raise Exception("dur and length can't be both specified.")
    if dur!=None:
        return _create_time1(fs,dur)
    else:
        return _create_time2(fs,length)

def apply_fades(s,fades):
    if fades[0]>0:
        s[0:fades[0]] = s[0:fades[0]] * ((-np.cos(np.arange(fades[0])/fades[0]*np.pi)+1) / 2)
    if fades[1]>0:
        s[-fades[1]:] = s[-fades[1]:] *  ((np.cos(np.arange(fades[1])/fades[1]*np.pi)+1) / 2)
    return s
