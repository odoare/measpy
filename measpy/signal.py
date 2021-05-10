# measpysignal.py
# 
# Signal helper functions for measpy
#
# OD - 2021

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.mlab import psd, csd
from scipy.signal import welch, csd, coherence, resample, convolve

# TODO :
# - Analysis functions of signals : levels dBSPL, resample
# - Coherence
# - Calibrations
# - Apply dBA, dBC or any calibration curve to a signal
# - add silence before, after

class Signal:
    """ Defines a signal object

        A signal is a temporal series of values.
        The object has the following properties :
            - desc : The description of the signal (string)
            - unit : The physical unit
            - cal : The calibration (in V/unit)
            - dbfs : The input voltage for a raw value of 1
            - fs : The sampling frequency
            - _values : A numpy array of raw values
    """
    def __init__(self,x=None,desc='A signal',fs=1,unit='1',cal=1.0,dbfs=1.0):
        self._rawvalues = np.array(x)
        self.desc = desc
        self.unit = unit
        self.cal = cal
        self.dbfs = dbfs
        self.fs = fs
        
    def as_signal(self,x):
        return Signal(x=x,fs=self.fs,unit=self.unit,cal=self.cal,dbfs=self.dbfs)

    def plot(self):
        plt.plot(self.time,self.values)
        plt.xlabel('Time (s)')
        plt.ylabel(self.desc+'  ['+self.unit+']')

    def psd(self,nperseg=2**15):
        out = Spectral_data('PSD of '+self.desc,self.fs,self.unit+'^2')
        _, out.values = welch(self.values, nperseg=nperseg, fs=self.fs)
        return out

    def rms_smooth(self,nperseg=100):
        out = self.as_signal(np.sqrt(smooth(self.values**2,nperseg)))
        out.desc=self.desc+'-->RMS smoothed on '+str(nperseg)+' data points'
        out.unit=self.unit+'^2'
        out.cal=1.0
        out.dbfs=1.0
        return out

    def resample(self,fs):
        out = self.as_signal(resample(self.raw,round(len(self.raw)*fs/self.fs)))
        out.desc=self.desc+'-->resampled to '+str(fs)+'Hz'
        return out

    @property
    def raw(self):
        return self._rawvalues
    @raw.setter
    def raw(self,val):
        self._rawvalues = val
    @property
    def values(self):
        return self._rawvalues*self.dbfs/self.cal
    @values.setter
    def values_in_unit(self,val):
        self._rawvalues = val*self.cal/self.dbfs
    @property
    def volts(self):
        return self._rawvalues*self.dbfs
    @volts.setter
    def volts(self,val):
        self._rawvalues = val/self.dbfs
    @property
    def time(self):
        return create_time(self.fs,length=len(self._rawvalues))

class Spectral_data():
    ''' Class that holds a set of values as function of evenly spaced
        frequencies. Usualy contains tranfert functions, spectral
        densities, etc.

        Frequencies are not stored. If needed they are constructed
        using sampling frequencies and length of the values array
        by calling the property freqs. 
    '''
    def __init__(self,desc='Spectral data',fs=1,unit='1'):
        self.desc = desc
        self.unit = unit
        self.fs = fs
        self._values = np.array([])
    def plot(self,axestype='logdb',xlabel=None,ylabel=None):
        if axestype=='logdb':
            plt.subplot(2,1,1)
            plt.semilogx(self.freqs,20*np.log10(np.abs(self.values)))
            plt.xlabel('Freq (Hz)')
            plt.ylabel('20 Log |H|')
            plt.title(self.desc)
            plt.subplot(2,1,2)
            plt.semilogx(self.freqs,20*np.angle(self.values))
            plt.xlabel('Freq (Hz)')
            plt.ylabel('Arg(H)')
    @property
    def values(self):
        return self._values
    @values.setter
    def values(self,val):
        self._values = val
    @property
    def freqs(self):
        return np.linspace(0, self.fs/2, num=len(self._values))

def picv(long):
    return np.hstack((np.zeros(long),1,np.zeros(long-1)))

def create_time1(fs,dur):
    return np.linspace(0,dur,int(round(dur*fs)))  # time axis

def create_time2(fs,length):
    return np.linspace(0,length/fs,length)  # time axis

def create_time(fs,dur=None,length=None):
    if dur==None and length==None:
        raise Exception('dur=duration in s or length=number of samples must be specified.')
    if dur!=None and length!=None:
        raise Exception("dur and length can't be both specified.")
    if dur!=None:
        return create_time1(fs,dur)
    else:
        return create_time2(fs,length)

def apply_fades(s,fades):
    if fades[0]>0:
        s[0:fades[0]] = s[0:fades[0]] * ((-np.cos(np.arange(fades[0])/fades[0]*np.pi)+1) / 2)
    if fades[1]>0:
        s[-fades[1]:] = s[-fades[1]:] *  ((np.cos(np.arange(fades[1])/fades[1]*np.pi)+1) / 2)
    return s

def noise(fs, dur, out_amp, freqs, fades):
    """ Create band-limited noise """
    t = create_time(fs,dur=dur)
    leng = int(dur*fs)
    lengs2 = int(leng/2)
    f = fs*np.arange(lengs2+1,dtype=float)/leng
    amp = ((f>freqs[0]) & (f<freqs[1]))*np.sqrt(leng)
    phase  = 2*np.pi*(np.random.rand(lengs2+1)-0.5)
    fftx = amp*np.exp(1j*phase)
    s = out_amp*np.fft.irfft(fftx)
    s = apply_fades(s,fades)
    return t,s

def tfe_welch(x, y, fs=None, nperseg=2**12,noverlap=None):
    """ Transfer function estimate (Welch's method)       
        Arguments and defaults :
        NFFT=None,
        Fs=None,
        detrend=None,
        window=None,
        noverlap=None,
        pad_to=None,
        sides=None,
        scale_by_freq=None
    """
    if type(x) != type(y):
        raise Exception('x and y must have the same type (numpy array or Signal object).')
    if type(x) == Signal:
        f, p = welch(x.values_in_unit , fs=x.fs, nperseg=nperseg, noverlap=noverlap )
        f, c = csd(y.values_in_unit ,x.values_in_unit, fs=x.fs, nperseg=nperseg, noverlap=noverlap)
        out = Spectral_data(desc='Transfer function between '+x.desc+' and '+y.desc,
                                fs=x.fs,
                                unit = y.unit+'/'+x.unit)
        out.values = c/p
        return out
    else:
        f, p = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        f, c = csd(y, x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, c/p

def log_sweep(fs, dur, out_amp, freqs, fades):
    """ Create log swwep """
    L = dur/np.log(freqs[1]/freqs[0])
    t = create_time(fs, dur=dur)
    s = np.sin(2*np.pi*freqs[0]*L*np.exp(t/L))
    s = apply_fades(s,fades)
    return t,out_amp*s

def tfe_farina(y, fs, freqs):
    """ Transfer function estimate
        Farina's method """
    leng = int(2**np.ceil(np.log2(len(y))))
    Y = np.fft.rfft(y,leng)/fs
    f = np.linspace(0, fs/2, num=round(leng/2)+1) # frequency axis
    L = len(y)/fs/np.log(freqs[1]/freqs[0])
    S = 2*np.sqrt(f/L)*np.exp(-1j*2*np.pi*f*L*(1-np.log(f/freqs[0])) + 1j*np.pi/4)
    S[0] = 0j
    H = Y*S
    return f, H

def plot_tfe(f, H):
    plt.subplot(2,1,1)
    plt.semilogx(f,20*np.log10(np.abs(H)))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('20 Log |H|')
    plt.subplot(2,1,2)
    plt.semilogx(f,20*np.angle(H))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Arg(H)')

def smooth(in_array,l=20):
    ker = np.ones(l)/l
    return np.convolve(in_array,ker,mode='same')

# class Signalb(np.ndarray):
#     def __new__(cls, input_array, fs=44100, cal=1.0, dbfs=1.0, unit='V'):
#         obj = np.asarray(input_array).view(cls)
#         obj.fs = fs
#         obj.cal = cal
#         obj.dbfs = dbfs
#         obj.unit = unit
#         return obj

#     def __array_finalize__(self, obj):
#         print('In __array_finalize__:')
#         print('   self is %s' % repr(self))
#         print('   obj is %s' % repr(obj))
#         if obj is None: return
#         self.fs = getattr(obj, 'fs', None)
#         self.cal = getattr(obj, 'cal', None)
#         self.dbfs = getattr(obj, 'dbfs', None)
#         self.unit = getattr(obj, 'unit', None)

#     # def __array_wrap__(self, out_arr, context=None):
#     #     print('In __array_wrap__:')
#     #     print('   self is %s' % repr(self))
#     #     print('   arr is %s' % repr(out_arr))
#     #     # then just call the parent
#     #     return super(Signalb, self).__array_wrap__(self, out_arr, context)

#     @property
#     def values_in_unit(self):
#         return self.__array__()*self.dbfs/self.cal
#     @values_in_unit.setter
#     def values_in_unit(self,val):
#         self.__array__ = val*self.cal/self.dbfs
#     @property
#     def values_in_volts(self):
#         return self.__array__()*self.dbfs
#     @values_in_volts.setter
#     def values_in_volts(self,val):
#         self.__array__ = val/self.dbfs
#     @property
#     def values(self):
#         return self.__array__()
#     @values.setter
#     def values(self,val):
#         self = Signalb(val,fs=self.fs,cal=self.cal,unit=self.unit,dbfs=self.dbfs)


# Old version that doesn't use Signals
# def create_noise(fs, dur, out_amp, freqs, fades):
#     """ Create band-limited noise """
#     t = create_time(fs,dur=dur)
#     leng = int(dur * fs)
#     lengs2 = int(leng/2)
#     f = fs*np.arange(lengs2+1,dtype=float)/leng
#     amp = ((f>freqs[0]) & (f<freqs[1]))*np.sqrt(leng)
#     phase  = 2*np.pi*(np.random.rand(lengs2+1)-0.5)
#     fftx = amp*np.exp(1j*phase)
#     s = np.fft.irfft(fftx)
#     s = apply_fades(s,fades)
#     return t,out_amp*s