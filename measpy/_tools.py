# measpy/_tools.py
#
# ----------------------------
# Utilities for measpy package
# ----------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

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
    """ Unwraps a phase array around a specified index  
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

def noise(fs, dur, out_amp, freq_min, freq_max):
    """ Create band-limited noise """
    leng = int(dur*fs)
    lengs2 = int(np.ceil(leng/2))
    f = fs*np.arange(lengs2+1, dtype=float)/leng
    amp = ((f > freq_min) & (f < freq_max))*np.sqrt(leng)
    phase = 2*np.pi*(np.random.rand(lengs2+1)-0.5)
    fftx = amp*np.exp(1j*phase)
    s = out_amp*np.fft.irfft(fftx, leng)
    return s

def log_sweep(fs, dur, out_amp, freq_min, freq_max):
    """ Create log sweep """
    L = (dur-1/fs)/np.log(freq_max/freq_min)
    t = create_time(fs, dur=dur)
    s = np.sin(2*np.pi*freq_min*L*np.exp(t/L))
    return out_amp*s

def sine(fs, dur, out_amp, freq):
    s = out_amp*np.sin(2*np.pi*create_time(fs=fs, dur=dur)*freq)
    return (s)

def saw(fs, dur, out_amp, freq):
    return out_amp*wrap(2*np.pi*freq*create_time(fs,dur=dur))/np.pi
    
def tri(fs, dur, out_amp, freq):
    return out_amp*2*np.abs(wrap(2*np.pi*freq*create_time(fs,dur=dur))/np.pi)-1

def t_min(siglist):
    return min(s.t0 for s in siglist)

def t_max(siglist):
    return max(s.t0+s.dur for s in siglist)

def calc_dur_siglist(siglist):
    return t_max(siglist)-t_min(siglist)

def siglist_to_array(siglist):
    durtot = calc_dur_siglist(siglist)
    out = np.zeros((round(durtot*siglist[0].fs),len(siglist)))
    t0s = t_min(siglist)
    for i,s in enumerate(siglist):
        out[round((s.t0-t0s)*s.fs):round(((s.t0-t0s)+s.dur)*s.fs),i] = s.raw
    return out

def get_index(array,value):
    """
    Get the index of the nearest value
    """
    return np.argmin((array-value)**2)

# def _tfe_farina(y, fs, freqs):
#     """ Transfer function estimate
#         Farina's method """
#     leng = int(2**np.ceil(np.log2(len(y))))
#     Y = np.fft.rfft(y, leng)/fs
#     f = np.linspace(0, fs/2, num=round(leng/2)+1)  # frequency axis
#     L = len(y)/fs/np.log(freqs[1]/freqs[0])
#     S = 2*np.sqrt(f/L)*np.exp(-1j*2*np.pi*f*L *
#                               (1-np.log(f/freqs[0])) + 1j*np.pi/4)
#     S[0] = 0j
#     H = Y*S
#     return f, H

# def _tfe_welch(x, y, **kwargs):
#     """ Transfer function estimate (Welch's method)       
#         Arguments and defaults :
#         NFFT=None,
#         Fs=None,
#         detrend=None,
#         window=None,
#         noverlap=None,
#         pad_to=None,
#         sides=None,
#         scale_by_freq=None
#     """
#     if type(x) != type(y):
#         raise Exception(
#             'x and y must have the same type (numpy array or Signal object).')

#     # Set default values for welch's kwargs
#     if not "fs" in kwargs:
#         kwargs["fs"] = x.fs
#     if not "nperseg" in kwargs:
#         kwargs["nperseg"] = 2**(np.ceil(np.log2(x.fs)))

#     if type(x) == Signal:
#         f, p = welch(x.values_in_unit, **kwargs)
#         f, c = csd(y.values_in_unit, x.values_in_unit, **kwargs)
#         out = Spectral(desc='Transfer function between '+x.desc+' and '+y.desc,
#                        fs=x.fs,
#                        unit=y.unit+'/'+x.unit)
#         out.values = c/p
#         return out
#     else:
#         f, p = welch(x, **kwargs)
#         f, c = csd(y, x, **kwargs)
#     return f, c/p
