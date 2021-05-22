# measpysignal.py
# 
# Signal helper functions for measpy
#
# OD - 2021

from warnings import WarningMessage
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import ones_like
from scipy.signal import welch, csd, coherence, resample
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io.wavfile as wav
import csv

import unyt
from unyt import Unit

from measpy._tools import add_step

# TODO :
# - Calibrations
# - Spectral arithmetics
# - Test dBu, dBV
# - Test Functions for all basic weightings (dBA, dBC)
# - Improve plotting functions

PREF = 20e-6*Unit('Pa') # Acoustic pressure reference level
DBUREF = 1*Unit('V')
DBVREF = np.sqrt(2)*Unit('V')

##################
##              ##
## Signal class ##
##              ##
##################

class Signal:
    """ The class signal describes a sampled data, its sampling
        frequency, its unit, the calibration and dbfs used for
        data acquisition. Methods are provided to analyse and
        transform the data.

        :param raw: raw data of the signal, defaults to array(None)
        :type raw: 1D numpy array, optional
        :param volts: raw data of the signal
        :type volts: 1D numpy array, optional
        :param values: raw data of the signal
        :type values: 1D numpy array, optional
        :param desc: Description of the signal, defaults to 'A signal'
        :type desc: str, optional
        :param fs: Sampling frequency, defaults to 1
        :type fs: int, optional
        :param unit: Unit of the signal given as a string that pint can understand, defaults to '1'
        :type unit: str, optional
        :param cal: calibration in V/unit, defaults to 1.0
        :type cal: float, optional
        :param dbfs: dbfs of the input data acquisition card, defaults to 1.0
        :type dbfs: float, optional

        A signal is a temporal series of values.
        The object has the following properties:

        * desc : The description of the signal (string)
        * unit : The physical unit (pint.Unit)
        * cal : The calibration (in V/unit)
        * dbfs : The input voltage for a raw value of 1
        * fs : The sampling frequency
        * _rawvalues : A numpy array of raw values
        
        Setters and getters properties:

        * values (values expressed in unit, calibrations applied)
        * volts (only dbfs applied)
        * raw (same as _rawvalues)
        * length (data length)
        * dur (duration in seconds)
        * time (time array)
    """

    def __init__(self,**kwargs):
        """ Initializes a Signal object with the specified entries """
        self.desc = kwargs.setdefault('desc','A signal')
        unit = kwargs.setdefault('unit','1')
        self.unit=Unit(unit)
        self.cal = kwargs.setdefault('cal',1.0)
        self.dbfs = kwargs.setdefault('dbfs',1.0)
        self.fs = kwargs.setdefault('fs',1)
        if 'values' in kwargs:
            self.values = np.array(kwargs['values'])
        elif 'volts' in kwargs:
            self.volts = np.array(kwargs['volts'])
        elif 'raw' in kwargs:
            self.raw = np.array(kwargs['raw'])
        else:
            self.raw=np.array(None)

    def __repr__(self):
        out = "measpy.Signal("
        out += "fs="+str(self.fs)
        out += ", desc='"+str(self.desc)+"'"
        out += ", cal="+str(self.cal)
        out += ", unit='"+str(self.unit)+"'"
        out += ", dbfs="+str(self.dbfs)+')'
        return out

    def similar(self, **kwargs):
        """ Returns a copy of the Signal object
            with properties changed as specified
            by the optionnal arguments.

            :param fs: Sampling frequency
            :type fs: int, optional
            :param desc: Description
            :type desc: str, optional
            :param unit: Signal unit
            :type unit: str, unyt.Unit, optional
            :param cal: Calibration in volts/unit
            :type cal: float, optional
            :param dbfs: Input voltage for raw value = 1
            :type dbfs: float, optional
            :param values: Signal values given in unit
            :type values: numpy.array, optional
            :param volts: Signal values given in volts
            :type volts: numpy.array, optional
            :param raw: Signal values given as raw samples
            :type raw: numpy.array, optional
            :return: A signal
            :rtype: measpy.signal.Signal

            Only one of the following parameters should
            be specifified : raw, volts, values
            If values is specified, the two others are not
            taken into account. If volts and raw are given,
            only volts is taken into account.

        """
        fs = kwargs.setdefault("fs",self.fs)
        desc = kwargs.setdefault("desc",self.desc)
        unit = kwargs.setdefault("unit",str(self.unit.units))
        cal = kwargs.setdefault("cal",self.cal)
        dbfs = kwargs.setdefault("dbfs",self.dbfs)
        if 'values' in kwargs:
            return Signal(values=kwargs['values'],fs=fs,desc=desc,unit=unit,cal=cal,dbfs=dbfs)
        elif 'volts' in kwargs:
            return Signal(volts=kwargs['volts'],fs=fs,desc=desc,unit=unit,cal=cal,dbfs=dbfs)
        elif 'raw' in kwargs:
            return Signal(raw=kwargs['raw'],fs=fs,desc=desc,unit=unit,cal=cal,dbfs=dbfs)
        else:
            return Signal(raw=self.raw,fs=fs,desc=desc,unit=unit,cal=cal,dbfs=dbfs)

    def plot(self):
        """ Basic plotting of the signal """
        plt.plot(self.time,self.values)
        plt.xlabel('Time (s)')
        plt.ylabel(self.desc+'  ['+str(self.unit.units)+']')

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

    def rms_smooth(self,nperseg=512):
        """ Compute the RMS of the Signal over windows
            of width nperseg samples
            
            :param nperseg: Window size, defaults to 512
            :type nperseg: int, optionnal
            :return: A resampled signal
            :rtype: measpy.signal.Signal       
        """
        return self.similar(
            raw=np.sqrt(smooth(self.values**2,nperseg)),
            desc=add_step(self.desc,'RMS smoothed on '+str(nperseg)+' data points')
        )

    def dB(self,ref):
        """ Computes 20*log10(self.values/ref)
            ref is for instance a pressure or volage reference
            
            :param ref: Reference quantity that has to be of same dimension 
            :type ref: unyt.array.unyt_quantity
            :return: A signal of dimension dB
            :rtype: measpy.signal.Signal

        """
        if type(ref)!=unyt.array.unyt_quantity:
            raise Exception('ref is not a unyt quantity')
        if not self.unit.same_dimensions_as(ref.units):
            raise Exception('ref has an incompatible unit')
        ref.convert_to_units(self.unit)
        return self.similar(
            raw=20*np.log10(self.values*self.unit/ref),
            dbfs=1.0,
            cal=1.0,
            unit=Unit('decibel'),
            desc=add_step(
                self.desc,
                'dB ref '+'{:.2e}'.format(ref.v)+str(ref.units)
            )
        )

    def dB_SPL(self):
        """ Computes 20*log10(self.values/PREF).
            PREF is the reference pressure in air (20e-6 Pa)
        """
        return self.dB(PREF)


    def resample(self,fs):
        """ Changes sampling rate of the signal

            :param fs: Desired sampling rate
            :type fs: float
            :return: A resampled signal
            :rtype: measpy.signal.Signal
        """            
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

            :param x: Other signal to compute the coherence with
            :type x: measpy.signal.Signal
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

            :param pos: Start and stop positions of the new signal, given as indices, defaults to (0,-1)
            :type pos: tuple of int, optionnal
            :param dur: Start and stop positions of the new signal, given as indices
            :type dur: tuple of float, optionnal

            pos and dur cannot be both specified
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
        """Apply fades at the begining and the end of the signal

        :param fades: Tuple of ints specifying the fade in and fade out lengths
        :type fades: (int,int)
        :return: Faded signal
        :rtype: measpy.signal.Signal
        """
        return self.similar(
            raw=_apply_fades(self.raw,fades),
            desc=add_step(self.desc,"fades")
        )

    def add_silence(self,extrat=(0,0)):
        """Add zeros at the begining and the end of the signal

        :param extrat: number of samples before and after the original signal, defaults to [0,0]
        :type extrat: tuple, optional
        :return: New signal
        :rtype: measpy.signal.Signal
        """
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
                                full=True,
                                desc=add_step(self.desc,'FFT'))
    
    def rfft(self):
        """ Real FFT of the signal
            Returns a Spectral object
        """
        return Spectral(values=np.fft.rfft(self.values),
                                fs=self.fs,
                                unit=self.unit,
                                full=False,
                                desc=add_step(self.desc,'RFFT'))
    
    def to_csvwav(self,filename):
        """Saves the signal into a pair of files:

        * A CSV file with the signal parameters
        * A WAV file with the raw data

        If the str parameter filename='file', the created files are file.csv and file.wav

        :param filename: string for the base file name
        :type filename: str
        """
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
        """Load a signal from a pair of csv and wav files

        :param filename: base file name
        :type filename: str
        :return: The loaded signal
        :rtype: measpy.signal.Signal
        """
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

    def unit_to(self,unit):
        """Change Signal unit

        :param unit: Unit to convert to (has to be compatible)
        :type unit: unyt.unit or str
        :raises Exception: 'Incompatible units'
        :return: Signal converted to the new unit
        :rtype: measpy.Signal
        """
        if type(unit)==str:
            unit=Unit(unit)
        if not self.unit.same_dimensions_as(unit):
            raise Exception('Incompatible units')
        a=self.unit.get_conversion_factor(unit)[0]
        return self.similar(
            raw=a*self.values,
            cal=1.0,
            dbfs=1.0,
            unit=unit,
            desc=add_step(self.desc,'Unit to '+str(unit))
        )

    def unit_to_std(self):
        """Change Signal unit to the standard base equivalent

        :return: Signal converted to the new unit
        :rtype: measpy.Signal
        """
        return self.unit_to(self.unit.get_base_equivalent())

    def _add(self,other):
        """Add two signals

        :param other: Other signal to add
        :type other: Signal
        :return: Sum of signals
        :rtype: Signal
        """

        if not self.unit.same_dimensions_as(other.unit):
            raise Exception('Incompatible units in addition of sginals')
        if self.fs!=other.fs:
            raise Exception('Incompatible sampling frequencies in addition of signals')
        if self.length!=other.length:
            raise Exception('Incompatible signal lengths')

        return self.similar(
            raw=self.values+other.unit_to(self.unit).values,
            cal=1.0,
            dbfs=1.0,
            desc=self.desc+'\n + '+other.desc           
        )

    def __add__(self,other):
        """Add something to the signal

        :param other: Something to add to
        :type other: Signal, float, int, scalar quantity
        """
        if type(other)==Signal:
            return self._add(other)
    
        if (type(other)==float) or (type(other)==int):
            print('Add with a number without unit, it is considered to be of same unit')
            return self._add(
                self.similar(
                    raw=np.ones_like(self.raw)*other*self.dbfs/self.cal,
                    desc=str(other)
                )
            )

        if type(other)==unyt.array.unyt_quantity:
            if not self.unit.same_dimensions_as(other.units):
                raise Exception('Incompatible units in addition of sginals')
            a=other.units.get_conversion_factor(self.unit)[0]
            return self._add(
                self.similar(
                    raw=np.ones_like(self.raw)*a*self.dbfs/self.cal,
                    desc=str(other)
                )
            )
        else:
            raise Exception('Incompatible type when adding something to a Signal')

    def __radd__(self,other):
        """Addition of two signals

        :param other: something else to add
        :type other: Signal, float, int, scalar quantity
        """
        return self.__add__(other)

    def __neg__(self):
        return self.similar(raw=-1*self.raw,desc='-'+self.desc)

    def __sub__(self,other):
        """Substraction of two signals

        :param other: other signal
        :type other: Signal, int, float or quantity
        """
        return self.__add__(other.__neg__())

    def __rsub__(self,other):
        """Substraction of two signals

        :param other: other signal
        :type other: Signal
        """
        return self.__neg__().__add__(other)

    def _mul(self,other):
        """Multiplication of two signals

        :param other: other signal
        :type other: Signal
        """
        if self.fs!=other.fs:
            raise Exception('Incompatible sampling frequencies in multiplication of signals')
        if self.length!=other.length:
            raise Exception('Incompatible signal lengths in multiplication of signals')
        
        return self.similar(
            raw=self.values*other.values,
            unit=self.unit*other.unit,
            cal=1.0,
            dbfs=1.0,
            desc=self.desc+'\n * '+other.desc           
        )

    def __mul__(self,other):
        """Multiplication of two signals

        :param other: other signal
        :type other: Signal
        """
        if type(other)==Signal:
            return self._mul(other)

        if (type(other)==float) or (type(other)==int):
            return self.similar(raw=other*self.raw,desc=str(other)+'*'+self.desc)

        if type(other)==unyt.array.unyt_quantity:
            return self._mul(
                self.similar(
                    raw=np.ones_like(self.raw)*other.v*self.dbfs/self.cal,
                    unit=other.units,
                    cal=1.0,
                    dbfs=1.0,
                    desc=str(other)
                )
            )
        else:
            raise Exception('Incompatible type when multipling something with a Signal')

    def __rmul__(self,other):
        """Multiplication of two signals

        :param other: other signal
        :type other: Signal
        """
        return self.__mul__(other)

    def __invert__(self):
        """Signal inverse
        """
        # Calibration and dbfs are reset to 1.0 during the process
        return self.similar(
            values=self.values**(-1),
            unit=1/self.unit,
            cal=1.0,
            dbfs=1.0,
            desc='1/'+self.desc
        )

    def _div(self,other):
        """Division of two signals

        :param other: other signal
        :type other: Signal
        """
        # if self.fs!=other.fs:
        #     raise Exception('Incompatible sampling frequencies in addition of signals')
      
        return self.similar(
            raw=self.values/other.values,
            unit=self.unit/other.unit,
            cal=1.0,
            dbfs=1.0,
            desc=self.desc+' / '+other.desc
        )

    def __truediv__(self,other):
        """Division of two signals

        :param other: other signal
        :type other: Signal
        """
        if type(other)==Signal:
            if self.fs!=other.fs:
                raise Exception('Incompatible sampling frequencies in division of signals')
            return self._div(other)

        if (type(other)==float) or (type(other)==int):
            return self.similar(raw=self.raw/other,desc=self.desc+'/'+str(other))

        if type(other)==unyt.array.unyt_quantity:
            return self._div(
                self.similar(
                    raw=np.ones_like(self.raw)*other.v*self.dbfs/self.cal,
                    unit=other.units,
                    cal=1.0,
                    dbfs=1.0,
                    desc=str(other)
                )
            )
        else:
            raise Exception('Incompatible type when multipling something with a Signal')

    def __rtruediv__(self,other):
        return self.__invert__().__mul__(other)

    def abs(self):
        """ Absolute value
            Returns a Signal class object
        """
        return self.similar(
            raw=np.abs(self.raw),
            desc=add_step(self.desc,"abs")
        )

    def __abs__(self):
        """Absolute value of signal

        :param other: other signal
        :type other: Signal
        """

        return self.abs()


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

        :param fs: Sampling frequency, defaults to 1
        :type fs: int, optional
        :param desc: Description, defaults to 'Spectral data'
        :type desc: str, optional
        :param unit: Spectral data unit
        :type unit: str, unyt.Unit, optional
        :param values: Values of the pectral data
        :type values: numpy.array, optional
        :param full: If true, the full spectrum is given, from 0 to fs, if false, only up to fs/2
        :type full: bool, optionnal
        
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
            if full:
                self._values=np.zeros(int(round(fs*kwargs['dur'])),dtype=complex)
            else:
                self._values=np.zeros(int(round(fs*kwargs['dur']/2)+1),dtype=complex)
        else:
            self._values=values
        self.desc = desc
        self.unit = Unit(unit)
        self.fs = fs
        self.full = full

    def similar(self,**kwargs):
        """ Returns a copy of the Spectral object
            with properties changed as specified
            by the optionnal arguments.

            It is possible to construct a new Spectral object
            by interpolating a Weighting object (parameter w)

            :param fs: Sampling frequency
            :type fs: int, optional
            :param desc: Description
            :type desc: str, optional
            :param unit: unit
            :type unit: str, unyt.Unit, optional
            :param values: values of the spectral data
            :type values: numpy array, optionnal
            :param w: A Weighting object from which the spectrum is constructed by interpolation
            :type w: measpy.signal.Weighting
            :return: A Spectral object
            :rtype: measpy.signal.Spectral

        """
        values = kwargs.setdefault("values",self.values)
        fs = kwargs.setdefault("fs",self.fs)
        desc = kwargs.setdefault("desc",self.desc)
        unit = kwargs.setdefault("unit",str(self.unit.units))
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
        """ Cancels values below and above a given frequency
            Returns a Spectral class object
        """
        return self.similar(
            values=self._values*(
                (self.freqs>freqsrange[0]) & (self.freqs<freqsrange[1]))
            )

    def abs(self):
        """ Absolute value
            Returns a Spectral class object
        """
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

    def unit_to(self,unit):
        if type(unit)==str:
            unit=Unit(unit)
        if not self.unit.same_dimensions_as(unit):
            raise Exception('Incompatible units')
        a=self.unit.get_conversion_factor(unit)[0]
        return self.similar(
            values=a*self.values,
            desc=add_step(self.desc,'Unit to '+str(unit))
        )  
    
    def apply_dBA(self):
        w = Weighting.from_csv('measpy/data/dBA.csv')
        return self.apply_weighting(w)

    def apply_dBC(self):
        w = Weighting.from_csv('measpy/data/dBC.csv')
        return self.apply_weighting(w)

    def dB_SPL(self):
        return self.unit_to(Unit(PREF)).similar(
            values=20*np.log10(self._values/PREF.v),
            desc=add_step(self.desc,'dB SPL')
        )

    def dBV(self):
        return self.unit_to(Unit(PREF)).similar(
            values=20*np.log10(self._values/DBVREF.v),
            desc=add_step(self.desc,'dBV')
        )
    
    def dBu(self):
        return self.unit_to(Unit(PREF)).similar(
            values=20*np.log10(self._values/DBUREF.v),
            desc=add_step(self.desc,'dBu')
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
        if axestype=='lin':
            plt.semilogx(self.freqs,self.values)
            plt.xlabel('Freq (Hz)')
            if ylabel1!=None:
                plt.ylabel(ylabel1)
            else:
                plt.ylabel('H')
            plt.title(self.desc)


    @classmethod
    def tfe(cls,x,y,**kwargs):
        if (type(x)!=Signal) & (type(y)!=Signal):
            raise Exception('x and y inputs have to be Signal')      
        return y.tfe_welch(x,**kwargs)
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
    """ Class for weighting functions
    """
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