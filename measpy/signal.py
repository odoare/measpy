# measpysignal.py
# 
# Signal helper functions for measpy
#
# OD - 2021

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, coherence, resample
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io.wavfile as wav
import csv
from pint import Unit

from measpy._tools import add_step

# TODO :
# - Analysis functions of signals : levels dBSPL, resample
# - Calibrations
# - Apply dBA, dBC or any calibration curve to a signal

PREF = 20e-6*Unit('Pa') # Acoustic pressure reference level

##################
##              ##
## Signal class ##
##              ##
##################

class Signal:
    """ Defines a signal object

        A signal is a temporal series of values.
        The object has the following properties :
            - desc : The description of the signal (string)
            - unit : The physical unit
            - cal : The calibration (in V/unit)
            - dbfs : The input voltage for a raw value of 1
            - fs : The sampling frequency
            - _rawvalues : A numpy array of raw values
        
        Setters and getters properties:
            - values (values expressed in unit, calibrations applied)
            - volts (only dbfs applied)
            - raw (same as _rawvalues)
            - length (data length)
            - dur (duration in seconds)
            - time (time array)
    """

    def __init__(self,raw=None,desc='A signal',fs=1,unit='1',cal=1.0,dbfs=1.0):
        self._rawvalues = np.array(raw)
        self.desc = desc
        self.unit = Unit(unit)
        self.cal = cal
        self.dbfs = dbfs
        self.fs = fs
    
    def similar(self, **kwargs):
        """ Returns a copy of the Signal object
            with properties changed as specified
            by the optionnal arguments:
            - raw: array of raw values
            - fs: sampling frequency
            - desc: description of the signal
            - cal: calibration
            - dbfs: volts for raw=1
        """
        raw = kwargs.setdefault("raw",self.raw)
        fs = kwargs.setdefault("fs",self.fs)
        desc = kwargs.setdefault("desc",self.desc)
        unit = kwargs.setdefault("unit",self.unit.format_babel())
        cal = kwargs.setdefault("cal",self.cal)
        dbfs = kwargs.setdefault("dbfs",self.dbfs)
        return Signal(raw=raw,fs=fs,desc=desc,unit=unit,cal=cal,dbfs=dbfs)

    def plot(self):
        """ Basic plotting of the signal """
        plt.plot(self.time,self.values)
        plt.xlabel('Time (s)')
        plt.ylabel(self.desc+'  ['+self.unit.format_babel()+']')

    def psd(self,**kwargs):
        """ Compute power spectral density of the signal object
            Optional arguments are the same as the welch function
            in scipy.signal

            Arguments are the same as scipy.welch()

            Returns : A Spectral object containing the psd
        """ 
        return Spectral(
            values=welch(self.values, **kwargs)[1],
            desc=add_step(self.desc,'PSD'),
            fs=self.fs,
            unit=self.unit**2
        )

    def rms_smooth(self,nperseg=100):
        """ Compute the RMS of the Signal over windows of width nperseg samples """
        return self.similar(
            raw=np.sqrt(smooth(self.values**2,nperseg)),
            desc=add_step(self.desc,'RMS smoothed on '+str(nperseg)+' data points')
        )

    def dB(self,ref):
        """ Computes 20*log10(self.values/ref)
            If the data is a smoothed RMS acoustic pressure, and ref is PREF,
            computes the Sound Pressure Level in dB,
            as the 20Log(P/Pref)
        """
        return self.similar(
            raw=20*np.log10(self.values*self.unit/ref),
            dbfs=1.0,
            cal=1.0,
            unit=Unit('decibel'),
            desc=add_step(
                self.desc,
                'dB ref '+'{:.2e}'.format(ref.magnitude)+ref.units.format_babel()
            )
        )

    def resample(self,fs):
        return self.similar(
            raw=resample(self.raw,round(len(self.raw)*fs/self.fs)),
            fs=fs,
            desc=add_step(self.desc,'resampled to '+str(fs)+'Hz')
        )

    def tfe_welch(self, x, **kwargs):
        """ Compute transfer function between signal x and the actual signal
        """
        if self.fs!=x.fs:
            raise Exception('Sampling frequencies have to be the same')
        if self.length!=x.length:
            raise Exception('Lengths have to be the same')

        return Spectral(
            values=csd(self.values, x.values, **kwargs)[1]/welch(x.values, **kwargs)[1],
            desc='Transfer function between '+x.desc+' and '+self.desc,
            fs=self.fs,
            unit=self.unit/x.unit,
            full=False
        )
    
    def coh(self, x, **kwargs):
        """ Compute the coherence between signal x and the actual signal
        """
        if self.fs!=x.fs:
            raise Exception('Sampling frequencies have to be the same')
        if self.length!=x.length:
            raise Exception('Lengths have to be the same')

        return Spectral(
            values=coherence(self.values, x.values, **kwargs)[1],
            desc='Coherence between '+x.desc+' and '+self.desc,
            fs=self.fs,
            unit=self.unit/x.unit,
            full=False
        )
    
    def cut(self,**kwargs):
        """ Cut signal between positions.
            Optionnal arguments:
            - pos: specify positions as indices
            - dur: specify positions as duration
        """
        if ('dur' in kwargs) and ('pos' in kwargs):
            raise Exception('Error: dur and pos cannot be both specified')
        if ('dur' in kwargs):
            pos = (int(round(kwargs['dur'][0]*self.fs)),int(round(kwargs['dur'][1]*self.fs)))
        if ('pos' in kwargs):
            pos = (kwargs['pos'][0],kwargs['dur'][1])
        else:
            pos = (0,-1)
        return self.similar(
            raw=self.raw[pos[0]:pos[1]],
            desc=add_step(self.desc,"Cut between "+str(pos[0])+" and "+str(pos[1]))
        )

    def fade(self,fades):
        return self.similar(
            raw=_apply_fades(self.raw,fades),
            desc=add_step(self.desc,"fades")
        )

    def add_silence(self,extrat=[0,0]):
        return self.similar(raw=np.hstack(
                (np.zeros(int(np.round(extrat[0]*self.fs))),
                self.raw,
                np.zeros(int(np.round(extrat[1]*self.fs))) ))
                )

    def tfe_farina(self, freqs):
        """ Compute the transfer function between x and the actual signal
            where x is a log sweep of same duration between freqs[0] and
            freq[1]
        """
        leng = int(2**np.ceil(np.log2(self.length)))
        Y = np.fft.rfft(self.values,leng)/self.fs
        f = np.linspace(0, self.fs/2, num=round(leng/2)+1) # frequency axis
        L = self.length/self.fs/np.log(freqs[1]/freqs[0])
        S = 2*np.sqrt(f/L)*np.exp(-1j*2*np.pi*f*L*(1-np.log(f/freqs[0])) + 1j*np.pi/4)
        S[0] = 0j
        return Spectral(values=Y*S,
            desc='Transfer function between input log sweep and '+self.desc,
            unit=self.unit/Unit('V'),
            fs=self.fs,
            full=False
        )
    
    def fft(self):
        """ FFT of the signal
            Returns a Spectral object
        """
        return Spectral(values=np.fft.fft(self.values),
                                fs=self.fs,
                                unit=self.unit,
                                full=True)
    
    def rfft(self):
        """ Real FFT of the signal
            Returns a Spectral object
        """
        return Spectral(values=np.fft.rfft(self.values),
                                fs=self.fs,
                                unit=self.unit,
                                full=False)
    
    def to_csvwav(self,filename):
        with open(filename+'.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['desc',self.desc])
            writer.writerow(['fs',self.fs])
            writer.writerow(['unit',self.unit.format_babel()])
            writer.writerow(['cal',self.cal])
            writer.writerow(['dbfs',self.dbfs])
        wav.write(filename+'.wav',int(round(self.fs)),self.raw)

    @classmethod
    def noise(cls,fs=44100,dur=2.0,amp=1.0,freqs=[20.0,20000.0],unit='1',cal=1.0,dbfs=1.0):
        return cls(
            raw=noise(fs,dur,amp,freqs),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc='Noise '+str(freqs[0])+'-'+str(freqs[1])+'Hz'
        ) 

    @classmethod
    def log_sweep(cls,fs=44100,dur=2.0,amp=1.0,freqs=[20.0,20000.0],unit='1',cal=1.0,dbfs=1.0):
        return cls(
            raw=log_sweep(fs,dur,amp,freqs),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc='Logsweep '+str(freqs[0])+'-'+str(freqs[1])+'Hz'
        ) 

    @classmethod
    def from_csvwav(cls,filename):
        out = cls()
        with open(filename+'.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0]=='desc':
                    out.desc=row[1]
                if row[0]=='fs':
                    out.fs=int(row[1])
                if row[0]=='unit':
                    out.unit=Unit(row[1])
                if row[0]=='cal':
                    out.cal=float(row[1])
                if row[0]=='dbfs':
                    out.dbfs=float(row[1])
        _, out._rawvalues = wav.read(filename+'.wav')
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
    def values(self,val):
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
    @property
    def length(self):
        return len(self._rawvalues)
    @property
    def dur(self):
        return len(self._rawvalues)/self.fs

    # END of Signal


####################
##                ##
## Spectral class ##
##                ##
####################

class Spectral:
    """ Class that holds a set of values as function of evenly spaced
        frequencies. Usualy contains tranfert functions, spectral
        densities, etc.

        Frequencies are not stored. If needed they are constructed
        using sampling frequencies and length of the values array
        by calling the property freqs.

        Creation arguments:
        - fs: sampling frequency (int)
        - desc: Description (str)
        - unit: Unit (string understandable by pint)
        - dur: duration in s (float)
        - values: values
        - full: If True, the full spectrum is given (between 0 and fs).
            If false, half spectrum is given (between 0 and fs/2)

        values and dur cannot be both specified.
        If dur is given, values are initialised at 0 
    """
    def __init__(self,**kwargs):
        if ('values' in kwargs) and ('dur' in kwargs):
            raise Exception('Error: values and dur cannot be both specified.')
        values = kwargs.setdefault("values",None)
        fs = kwargs.setdefault("fs",1)
        desc = kwargs.setdefault("desc",'Spectral data')
        unit = kwargs.setdefault("unit",'1')
        full = kwargs.setdefault("full",False)
        if 'dur' in kwargs:
            self._values=np.zeros(int(round(fs*kwargs['dur'])),dtype=complex)
        else:
            self._values=values
        self.desc = desc
        self.unit = Unit(unit)
        self.fs = fs
        self.full = full

    def similar(self,**kwargs):
        values = kwargs.setdefault("values",self.values)
        fs = kwargs.setdefault("fs",self.fs)
        desc = kwargs.setdefault("desc",self.desc)
        unit = kwargs.setdefault("unit",self.unit.format_babel())
        full = kwargs.setdefault("full",self.full)
        out = Spectral(values=values,fs=fs,desc=desc,unit=unit,full=full)
        if 'w' in kwargs:
            w = kwargs['w']
            spl = InterpolatedUnivariateSpline(w.f,w.a,ext=1)
            out.values = spl(self.freqs)
        return out

    def nth_oct_smooth_to_weight(self,n):
        """ Nth octave smoothing """
        fc,f1,f2 = nth_octave_bands(n)
        val = np.zeros_like(fc)
        for ii in range(len(fc)):
            val[ii] = np.mean(
                self.values[ (self.freqs>f1[ii]) & (self.freqs<f2[ii]) ]
            )
        # Check for NaN values (generally at low frequencies)
        for ii in range(len(fc)-1,-1,-1):
            if val[ii]!=val[ii]:
                val[ii]=val[ii+1]
        return Weighting(
            f=fc,
            a=val,
            desc=add_step(self.desc,'1/'+str(n)+'th oct. smooth')
        )

    def nth_oct_smooth_to_weight_complex(self,n):
        """ Nth octave smoothing """
        fc,f1,f2 = nth_octave_bands(n)
        val = np.zeros_like(fc,dtype=complex)
        for ii in range(len(fc)):
            module = np.abs(self.values[ (self.freqs>f1[ii]) & (self.freqs<f2[ii]) ])
            phase = np.unwrap(
                np.angle(self.values[(self.freqs>f1[ii]) & (self.freqs<f2[ii])])
            )
            val[ii] = np.mean(module) * np.exp(1j*np.mean(phase))
        return Weighting(
            f=fc,
            a=val,
            desc=add_step(self.desc,'1/'+str(n)+'th oct. smooth (complex)')
        )

    def nth_oct_smooth(self,n):
        return self.similar(
            w=self.nth_oct_smooth_to_weight(n),
            desc=add_step(self.desc,'1/'+str(n)+'th oct. smooth')
        )

    def irfft(self):
        """ Compute the real inverse Fourier transform
            of the spectral data set
        """
        if self.full:
            raise Exception('Error: the spectrum is full, use ifft instead')
        return Signal(raw=np.fft.irfft(self.values),
                            desc=add_step(self.desc,'IFFT'),
                            fs=self.fs,
                            unit=self.unit)

    def ifft(self):
        """ Compute the inverse Fourier transform
            of the spectral data set
        """
        if not(self.full):
            raise Exception('Error: the spectrum is not full, use irfft instead')
        return Signal(raw=np.fft.ifft(self.values),
                            desc=add_step(self.desc,'IFFT'),
                            fs=self.fs,
                            unit=self.unit)

    def filterout(self,freqsrange):
        """ Cancels values below and above a given frequency """
        return self.similar(
            values=self._values*(
                (self.freqs>freqsrange[0]) & (self.freqs<freqsrange[1]))
            )

    def abs(self):
        return self.similar(
            values=np.abs(self.values),
            desc=add_step(self.desc,"abs")
        )

    def apply_weighting(self,w):
        spl = InterpolatedUnivariateSpline(w.f,w.a,ext=1)
        return self.similar(
            values=self._values*spl(self.freqs),
            desc=add_step(self.desc,w.desc)
        )

    def plot(self,axestype='logdb_arg',ylabel1=None,ylabel2=None):
        if axestype=='logdb_arg':
            plt.subplot(2,1,1)
            plt.semilogx(self.freqs,20*np.log10(np.abs(self.values)))
            plt.xlabel('Freq (Hz)')
            if ylabel1!=None:
                plt.ylabel(ylabel1)
            else:
                plt.ylabel('20 Log |H|')
            plt.title(self.desc)
            plt.subplot(2,1,2)
            plt.semilogx(self.freqs,np.unwrap(np.angle(self.values)))
            plt.xlabel('Freq (Hz)')
            if ylabel2!=None:
                plt.ylabel(ylabel2)
            else:
                plt.ylabel('Arg(H)')
        if axestype=='logdb':
            plt.semilogx(self.freqs,20*np.log10(np.abs(self.values)))
            plt.xlabel('Freq (Hz)')
            if ylabel1!=None:
                plt.ylabel(ylabel1)
            else:
                plt.ylabel('20 Log |H|')
            plt.title(self.desc)

    @classmethod
    def tfe(cls,x,y):
        return y.tfe(x)

    @property
    def values(self):
        return self._values
    @values.setter
    def values(self,val):
        self._values = val
    @property
    def freqs(self):
        if self.full:
            return np.linspace(0, self.fs, num=len(self._values))
        else:
            return np.linspace(0, self.fs/2, num=len(self._values))

    # END of Spectral

#####################
##                 ##
## Weighting Class ##
##                 ##
#####################

class Weighting:
    def __init__(self,f,a,desc):
        self.f=f
        self.a=a
        self.desc=desc

    @classmethod
    def from_csv(cls,filename,asdB=True):
        out = cls([],[],'Weigting')
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            n=0
            for row in reader:
                if n==0:
                    out.desc=row[0]
                else:
                    out.f+=[float(row[0])]
                    out.a+=[float(row[1])]
                n+=1
        out.f=np.array(out.f)
        if asdB:
            out.a=10**(np.array(out.a)/20.0)
        else:
            out.a=np.array(out.a)
        return out

    def to_csv(self,filename,asdB):
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([self.desc])
            if asdB:
                for n in range(len(self.f)):
                    writer.writerow([self.f[n],20*np.log10(self.a[n])])
            else:
                for n in range(len(self.f)):
                    writer.writerow([self.f[n],self.a[n]])

    @property
    def adb(self):
        return 20*np.log10(np.abs(self.a))

    # END of Weighting


# Below are functions that may be useful (some cleaning should be done)

def picv(long):
    """ Create a 1D-array of length long with a unitary peak in the middle """
    return np.hstack((np.zeros(long),1,np.zeros(long-1)))

def _create_time1(fs,dur):
    return np.linspace(0,dur,int(round(dur*fs)))  # time axis

def _create_time2(fs,length):
    return np.linspace(0,length/fs,length)  # time axis

def create_time(fs,dur=None,length=None):
    if dur==None and length==None:
        raise Exception('dur=duration in s or length=number of samples must be specified.')
    if dur!=None and length!=None:
        raise Exception("dur and length can't be both specified.")
    if dur!=None:
        return _create_time1(fs,dur)
    else:
        return _create_time2(fs,length)

def _apply_fades(s,fades):
    if fades[0]>0:
        s[0:fades[0]] = s[0:fades[0]] * ((-np.cos(np.arange(fades[0])/fades[0]*np.pi)+1) / 2)
    if fades[1]>0:
        s[-fades[1]:] = s[-fades[1]:] *  ((np.cos(np.arange(fades[1])/fades[1]*np.pi)+1) / 2)
    return s


def noise(fs, dur, out_amp, freqs):
    """ Create band-limited noise """
    leng = int(dur*fs)
    lengs2 = int(leng/2)
    f = fs*np.arange(lengs2+1,dtype=float)/leng
    amp = ((f>freqs[0]) & (f<freqs[1]))*np.sqrt(leng)
    phase  = 2*np.pi*(np.random.rand(lengs2+1)-0.5)
    fftx = amp*np.exp(1j*phase)
    s = out_amp*np.fft.irfft(fftx)
    return s


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
        out = Spectral(desc='Transfer function between '+x.desc+' and '+y.desc,
                                fs=x.fs,
                                unit = y.unit+'/'+x.unit)
        out.values = c/p
        return out
    else:
        f, p = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        f, c = csd(y, x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, c/p


def log_sweep(fs, dur, out_amp, freqs):
    """ Create log swwep """
    L = dur/np.log(freqs[1]/freqs[0])
    t = create_time(fs, dur=dur)
    s = np.sin(2*np.pi*freqs[0]*L*np.exp(t/L))
    return out_amp*s

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

def nth_octave_bands(n):
    """ 1/nth octave band frequency range calculation """
    nmin = int(np.ceil(n*np.log2(1*10**-3)))
    nmax = int(np.ceil(n*np.log2(20e3*10**-3)))
    indices = range(nmin,nmax+1)
    f_centre = 1000 * (2**(np.array(indices)/n))
    f2 = 2**(1/n/2)
    f_upper = f_centre * f2
    f_lower = f_centre / f2
    return f_centre, f_lower, f_upper

# def noise(fs, dur, out_amp, freqs, fades):
#     """ Create band-limited noise """
#     t = _create_time(fs,dur=dur)
#     leng = int(dur*fs)
#     lengs2 = int(leng/2)
#     f = fs*np.arange(lengs2+1,dtype=float)/leng
#     amp = ((f>freqs[0]) & (f<freqs[1]))*np.sqrt(leng)
#     phase  = 2*np.pi*(np.random.rand(lengs2+1)-0.5)
#     fftx = amp*np.exp(1j*phase)
#     s = out_amp*np.fft.irfft(fftx)
#     s = _apply_fades(s,fades)
#     return t,s

# def log_sweep(fs, dur, out_amp, freqs, fades):
#     """ Create log swwep """
#     L = dur/np.log(freqs[1]/freqs[0])
#     t = _create_time(fs, dur=dur)
#     s = np.sin(2*np.pi*freqs[0]*L*np.exp(t/L))
#     s = _apply_fades(s,fades)
#     return t,out_amp*s


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