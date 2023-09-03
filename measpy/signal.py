# measpy/signal.py
#
# -------------------------------------------------
# This file defines the Signal and Spectral classes
# Namespaces :  measpy.signal.Signal
#               measpy.signal.Spectral
#               measpy.signal.Weighting
# -------------------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy


from warnings import WarningMessage
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from scipy.signal import (welch,
                          csd,
                          coherence,
                          resample,
                          iirfilter,
                          sosfilt,
                          correlate,
                          correlation_lags,
                          hilbert,
                          spectrogram,
                          convolve)
# from scipy.interpolate import InterpolatedUnivariateSpline
from csaps import csaps
import scipy.io.wavfile as wav
import csv
import copy
import numbers

import unyt
from unyt import Unit

from ._tools import (add_step,
                           smooth,
                           nth_octave_bands,
                           create_time,
                           apply_fades,
                           sine,
                           noise,
                           log_sweep,
                           saw,
                           tri,
                           unwrap_around_index,
                           get_index)

##################
##              ##
## Signal class ##
##              ##
##################

class Signal:
    """ Signal definition class

    The class signal encapsulates all the data describing
    the data acquisition of a physical signal:

    - the actual data as a numpy array (``_rawvalues``)
    - the sampling frequency (``fs``)
    - a descriptor string (``desc``)
    - the physical unit (``unit``, optional, dimensionless if not specified)
    - the calibration information, that allows to convert the data in volts to the actual physical unit (``cal``, optional, unitary if not specified)
    - an additionnal conversion factor used if the acquired data is proportionnal but not equal to the actual volts going into the AD converted, typically for soundcards (``dbfs``, optional, unitary if not specified)
    - start time of the signal (``t0``) : the actual time of the first sample
    - any additionnal user property (string, float, int)

    The class defines methods for signal processing, saving,
    restoring the data, plotting, etc. Most of the signal processing
    methods internally call numpy/scipy functions.

    Examples of signal creation:

    - An empty signal of unit Pascals, and sampling frequency 48kHz :

    .. highlight:: python
    .. code-block:: python

        import measpy as mp
        pa = mp.Signal(fs=48000, unit='Pa')

    - A signal of unit Pascals, and sampling frequency 48kHz,
      filled with 1000 random values, followed by a plot:

    .. highlight:: python
    .. code-block:: python
    
        import measpy as mp
        import numpy as np
        pa = mp.Signal(fs=48000, unit='Pa', values=np.random.rand(1000))
        pa.plot()

    - To generate some standard signals, there are some classical waveform creation methods.
      For example a five seconds logarithmic sweep from 20Hz to 20kHz, of dimension volts:

    .. highlight:: python
    .. code-block:: python
    
        import measpy as mp
        pa = mp.Signal.log_sweep(freq_min=20,freq_max=20,dur=5,unit='V)
        pa.plot()

    Other examples and tutorials are found in the example folder of measpy project.

    At creation, the following parameters can be given
    (transmitted to the __init__ function of the class Signal):

    :param desc: Description of the signal, defaults to 'A signal'
    :type desc: str, optional
    :param fs: Sampling frequency, defaults to 1
    :type fs: float, optional
    :param unit: Unit of the signal given as a string that unyt can understand, defaults to None
    :type unit: str or None, optional
    :param cal: calibration in V/unit, defaults to 1.0
    :type cal: float, optional
    :param dbfs: dbfs of the input data acquisition card, defaults to 1.0
    :type dbfs: float, optional
    :param raw: raw data of the signal, defaults to array(None)
    :type raw: 1D numpy array, optional
    :param volts: data of the signal given in volts
    :type volts: 1D numpy array, optional
    :param values: data of the signal givent in its physical units
    :type values: 1D numpy array, optional
    :param any: Any other parameter can be given. They are stored as a new property for a user-personalized use. For instance, the log_sweep creation method introduced above stores the low and high frequencies in the properties freq_min and freq_max, as it can be useful to keep track of these values for later signal processing.

    Some defined properties are calculated on demand when called. For instance, the duration of the signal depends on the number of samples and sampling frequency.

    .. highlight:: python
    .. code-block:: python
    
        import measpy as mp
        pa = mp.Signal(fs=48000, unit='Pa', values=np.random.rand(1000))
        # show duration
        print(pa.dur)
        # show the corresponding time vector (numpy array)
        print(pa.time)     

    The following properties are implemented:

    * values (values expressed in unit, calibrations applied)
    * volts (only dbfs applied)
    * raw (same as _rawvalues)
    * length (data length)
    * dur (duration in seconds)
    * time (time array)
    * max, min : max value, min value
    * tmax, tmin : time at max or min value
    * rms : Root mean square of the signal

    ------------------------------------------------------------
    """

    # #################################################################
    # Methods returning a signal
    # #################################################################

    def __init__(self, **kwargs):
        """
        Signal initialization

        If one optional parameter values, volts or raw is given, the created signal is initialized with the given values. If none of these optional parameter are given, the created signal is empty.

        :param fs: Sampling frequency, defaults to 1.
        :type fs: int, optional
        :param desc: Description, defaults to 1.
        :type desc: str, optional
        :param unit: Signal unit, defaults to '1' (dimensionless).
        :type unit: str, unyt.Unit, optional
        :param cal: Calibration in volts/unit, defaults to 1.
        :type cal: float, optional
        :param dbfs: Input voltage for raw value = 1, defaults to 1.
        :type dbfs: float, optional
        :param values: Signal values given in unit
        :type values: numpy.array, optional
        :param volts: Signal values given in volts
        :type volts: numpy.array, optional
        :param raw: Signal values given as raw samples
        :type raw: numpy.array, optional
        :return: A signal
        :rtype: measpy.signal.Signal

        """

        if 'fs' not in kwargs:
            self.fs = 1.0
        if 'desc' not in kwargs:
            self.desc = 'A signal'

        # We have to make sure that properties such as dbfs
        # and cal are the correct ones BEFORE values are calculated
        # Thus, we run the loop two times

        for arg in kwargs:
            if arg == 'values':
                pass
            elif arg == 'volts':
                pass
            elif arg == 'raw':
                pass
            elif arg == 'unit':
                self.unit = kwargs[arg]
            elif arg == 't0':
                self.t0 = kwargs[arg]
            elif arg == 'dbfs':
                self.dbfs = kwargs[arg]    
            elif arg == 'cal':
                self.cal = kwargs[arg]   
            elif arg == 'dur':
                raise AttributeError("Property 'dur' cannot be set")
            else:
                self.__dict__[arg] = kwargs[arg]

        self.raw = np.array([])

        for arg in kwargs:
            if arg == 'values':
                self.values = kwargs[arg]
            elif arg == 'volts':
                self.volts = kwargs[arg]
            elif arg == 'raw':
                self.raw = kwargs[arg]

    def similar(self, **kwargs):
        """ Returns a copy of the Signal object
            with properties changed as specified
            by the optional arguments.

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
            :param t0: Timeshift of the signal
            :type t0: float, optional
            :param any: Any other parameter can be specified
            :type any: float, int, string
            :return: A signal
            :rtype: measpy.signal.Signal

            Only one of the following parameters should
            be specifified : raw, volts, values
            If values is specified, the two others are not
            taken into account. If volts and raw are given,
            only volts is taken into account.

        """

        out = copy.deepcopy(self)

        # We have to make sure that properties such as dbfs
        # and cal are the correct ones BEFORE values are calculated
        # Thus, we run the loop two times
        for arg in kwargs:
            if arg == 'values':
                pass
            elif arg == 'volts':
                pass
            elif arg == 'raw':
                pass
            elif arg == 'unit':
                out.unit = kwargs[arg]  
            elif arg == 't0':
                out.t0 = kwargs[arg]
            elif arg == 'dbfs':
                out.dbfs = kwargs[arg]
            elif arg == 'cal':
                out.cal = kwargs[arg]
            else:
                out.__dict__[arg] = kwargs[arg]
        for arg in kwargs:
            if arg == 'values':
                out.values = kwargs[arg]
            elif arg == 'volts':
                out.volts = kwargs[arg]
            elif arg == 'raw':
                out.raw = kwargs[arg]
        return out

    def rms_smooth(self, nperseg=512):
        """ Compute the RMS of the Signal over windows
            of width nperseg samples

            :param nperseg: Window size, defaults to 512
            :type nperseg: int, optionnal
            :return: A resampled signal
            :rtype: measpy.signal.Signal       
        """
        return self.similar(
            values=np.sqrt(smooth(self.values**2, nperseg)),
            desc=add_step(self.desc, 'RMS smoothed on ' +
                          str(nperseg)+' data points'),
            cal=None,
            dbfs=None
        )
    
    def smooth(self, nperseg=512):
        """ Compute the RMS of the Signal over windows
            of width nperseg samples

            :param nperseg: Window size, defaults to 512
            :type nperseg: int, optionnal
            :return: A resampled signal
            :rtype: measpy.signal.Signal       
        """
        return self.similar(
            values=smooth(self.values, nperseg),
            desc=add_step(self.desc, 'Smoothed on ' +
                          str(nperseg)+' data points'),
            cal=None,
            dbfs=None
        )

    def dB(self, ref):
        """ Computes 20*log10(self.values/ref)
            ref is for instance a pressure or volage reference that
            has to be of same units as the signal.

            :param ref: Reference quantity that has to be of same dimension 
            :type ref: unyt.array.unyt_quantity
            :return: A signal of dimension dB
            :rtype: measpy.signal.Signal

        """
        if type(ref) != unyt.array.unyt_quantity:
            raise Exception('ref is not a unyt quantity')
        if not self.unit.same_dimensions_as(ref.units):
            raise Exception('ref has an incompatible unit')
        ref.convert_to_units(self.unit)
        return self.similar(
            raw=20*np.log10(self.values*self.unit/ref),
            dbfs=None,
            cal=None,
            unit=Unit('decibel'),
            desc=add_step(
                self.desc,
                'dB ref '+'{:.2e}'.format(ref.v)+str(ref.units)
            )
        )

    def dB_SPL(self):
        """ Computes 20*log10(self.values/PREF).
            PREF is the reference pressure in air (20e-6 Pa)

            :return: Signal of unit dB
            :rtype: measpy.signal.Signal
        """
        return self.dB(PREF)

    def dB_SVL(self):
        """ Computes 20*log10(self.values/VREF).
            VREF is the reference particle velocity (5e-8 m/s)
            
            :return: Signal of unit dB
            :rtype: measpy.signal.Signal
        """
        return self.dB(VREF)

    def resample(self, fs):
        """ Changes sampling rate of the signal

            :param fs: Desired sampling rate
            :type fs: float
            :return: A resampled signal
            :rtype: measpy.signal.Signal
        """
        return self.similar(
            raw=resample(self.raw, round(len(self.raw)*fs/self.fs)),
            fs=fs,
            desc=add_step(self.desc, 'resampled to '+str(fs)+'Hz')
        )

    def corr(self, x, **kwargs):
        """ Compute the cross correlation between signal x and the actual signal

            :param x: Other signal to compute the coherence with
            :type x: measpy.signal.Signal

            :return: A signal representing the cross-correlation
            :rtype: measpy.signal.Signal

        """

        return self.similar(
            values=correlate(self.values, x.values, **kwargs),
            desc=add_step(self.desc, 'correlation with '+x.desc),
            t0=(correlation_lags(self.length, x.length)[0]-0.5)/self.fs)

    def cut(self, **kwargs):
        """ Cut signal between positions.

            :param pos: Start and stop positions of the new signal, given as indices, defaults to (0,-1)
            :type pos: tuple of int, optional
            :param dur: Start and stop positions of the new signal, given as time values
            :type dur: tuple of float, optional

            :return: Faded signal
            :rtype: measpy.signal.Signal

            pos and dur cannot be both specified. An exception is raised in that case.

            Negative values are possible, as well as values beyond the end of the signal. The signal is looped in that case.

            pos[1] can be lower that pos[0], in that case, the signal is reversed.

            NOTE : cut ignores the timeshift _t0 property
        """
        if ('dur' in kwargs) and ('pos' in kwargs):
            raise Exception('Error: dur and pos cannot be both specified')
        elif ('dur' in kwargs):
            pos = (int(round(kwargs['dur'][0]*self.fs)),
                   int(round(kwargs['dur'][1]*self.fs)))
        elif ('pos' in kwargs):
            pos = (kwargs['pos'][0], kwargs['pos'][1])
        else:
            pos = (0, -1)
        return self.similar(
            raw=np.take(self.raw, range(pos[0], pos[1], np.sign(pos[1]-pos[0])), mode='wrap'),
            desc=add_step(self.desc, "Cut between " +
                          str(pos[0])+" and "+str(pos[1]))
        )
        # return self.similar(
        #     raw=self.raw[pos[0]:pos[1]],
        #     desc=add_step(self.desc,"Cut between "+str(pos[0])+" and "+str(pos[1]))
        # )

    def fade(self, fades):
        """Apply fades at the begining and the end of the signal

        :param fades: Tuple of ints specifying the fade in and fade out lengths
        :type fades: (int,int)
        :return: Faded signal
        :rtype: measpy.signal.Signal
        """
        return self.similar(
            raw=apply_fades(self.raw, fades),
            desc=add_step(self.desc, "fades")
        )

    def add_silence(self, extrat=(0, 0)):
        """Add zeros at the begining and the end of the signal

        :param extrat: time in seconds before and after the original signal, defaults to [0,0]
        :type extrat: tuple, optional
        :return: New signal
        :rtype: measpy.signal.Signal
        """
        return self.similar(raw=np.hstack(
            (np.zeros(int(np.round(extrat[0]*self.fs))),
             self.raw,
             np.zeros(int(np.round(extrat[1]*self.fs)))))
        )

    def iir(self, N=2, Wn=(20, 20000), rp=None, rs=None, btype='band',  ftype='butter'):
        """Infinite impulse response filter of a signal.

        The signal is filtered accordingly to the parameters. This method is a wrapper around the scipy.signal iir functions, most of the parameters are hence the same.

        :param N: Filter order, defaults to 2
        :type N: int, optional
        :param Wn: a cutoff frequency (if low or highpass) or a tuple of frequency (if bandpass/stop), defaults to (20,20000)
        :type Wn: tuple, optional
        :param rp: For Chebyshev and elliptic filters, provides the maximum ripple in the passband. (dB)
        :type rp: float, optional
        :param rs: For Chebyshev and elliptic filters, provides the minimum attenuation in the stop band. (dB)
        :type rs: float, optional
        :param btype: Type of filter (band, hp, lp), defaults to 'band'
        :type btype: str in {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        :param ftype: Type of filter (butter, elliptic, etc.), defaults to 'butter'
        :type ftype: str, optional
        :return: A filtered signal
        :rtype: measpy.signal.Signal
        """

        sos = iirfilter(N=N, Wn=Wn, rs=rs, rp=rp, btype=btype,
                        analog=False, ftype=ftype, fs=self.fs,
                        output='sos')
        return self.similar(
            values=sosfilt(sos, self.values),
            desc=add_step(self.desc, 'filtered'))

    def hilbert(self):
        """
        Computes the hilbert transform of a signal
        Warning: This method the imaginary part of the hilbert function of the scipy module.

        :return: A signal
        :rtype: measpy.signal.Signal
        """
        return self.similar(values=np.imag(hilbert(self.values)), desc=add_step(self.desc, 'hilbert'))

    def hilbert_ana(self):
        """
        Computes the analytical signal through hilbert transform of a signal
        Note: This method is exactly the hilbert function of the scipy module.

        :return: A signal
        :rtype: measpy.signal.Signal

        """
        return self.similar(values=hilbert(self.values), desc=add_step(self.desc, 'hilbert_ana'))

    def as_volts(self):
        """
        Returns the signal in volts, no calibration applied.

        :return: A signal
        :rtype: measpy.signal.Signal
        """

        return self.similar(unit='V', cal=None, dbfs=None, raw=self.volts, desc=add_step(self.desc, 'Voltage'))

    def as_raw(self):
        """
        Returns the signal raw values, dimensionless, no calibration applied, no dbfs applied.

        :return: A signal
        :rtype: measpy.signal.Signal
        """
        return self.similar(unit='1', cal=None, dbfs=None, raw=self.raw, desc=add_step(self.desc, 'Raw data'))

    def unit_to(self, unit):
        """
        Change Signal unit

        :param unit: Unit to convert to (has to be compatible)
        :type unit: unyt.unit or str
        :raises Exception: 'Incompatible units'
        :return: Signal converted to the new unit
        :rtype: measpy.signal.Signal
        """
        if type(unit) == str:
            unit = Unit(unit)
        if not self.unit.same_dimensions_as(unit):
            raise Exception('Incompatible units')
        a = list(self.unit.get_conversion_factor(unit))
        if a[1] == None:
            a[1] = 0
        return self.similar(
            values=a[0]*self.values-a[1],
            cal=None,
            dbfs=None,
            unit=unit,
            desc=add_step(self.desc, 'Unit to '+str(unit))
        )

    def unit_to_std(self):
        """Change Signal unit to the standard base equivalent

        :return: Signal converted to the new unit
        :rtype: measpy.signal.Signal
        """
        return self.unit_to(self.unit.get_base_equivalent())

    def normalize(self):
        """Normalize a signal

        :return: Dimensionless normalized signal
        :rtype: measpy.signal.Signal
        """

        return (self/self.max).similar(desc=add_step(self.desc, "Normalize"))

    def diff(self):
        """ Compute time derivative

        :return: Time derivative of signal (unit/s)
        :rtype: measpy.signal.Signal
        """
        return self.similar(values=np.diff(self.values)*self.fs, unit=self.unit/Unit('s'), desc=add_step(self.desc, 'diff'), cal=None, dbfs=None)

    def real(self):
        """ Real part of the signal, calibrations applied

        :return: The real part of the signal
        :rtype: measpy.signal.Signal        
        """
        return self.similar(
            values=np.real(self.values),
            desc=add_step(self.desc, "Real part")
        )

    def imag(self):
        """ Imaginary part of the signal, calibrations applied

        :return: The imaginary part of the signal
        :rtype: measpy.signal.Signal    
        """
        return self.similar(
            values=np.real(self.values),
            desc=add_step(self.desc, "Imaginary part")
        )

    def angle(self, unwrap=True):
        """ Compute the angle of the signal, if complex

        :param unwrap: If True, the angle data is unwrapped
        :type unwrap: bool

        :return: The angle part of the signal, unit=rad
        :rtype: measpy.signal  

        """
        vals = np.angle(self.values)
        if unwrap:
            vals = np.unwrap(vals)
            desc = add_step(self.desc, "Angle (unwraped)")
        else:
            desc = add_step(self.desc, "Angle")
        return self.similar(
            values=vals,
            desc=desc,
            unit='rad',
            cal=None,
            dbfs=None
        )
    
    def convolve(self,other,**kwargs):
        """
        Convolution of two signals

        Optional arguments are that of the scipy.signal.convolve function.
        
        :param other: Other signal to convolve with
        :type other: meapsy.signal.Signal
        :param mode: A tring indicating the size of the output. Optional. See scipy docs for details
        :type mode: str, {'full', 'valid', 'same'}
        :param method: A string indicating which method to use to calculate the convolution. Optional. See scipy docs for details
        :type method: str {'auto', 'direct', 'fft'}
        :return: A signal representing the linear convolution of the two signals.
        :rtype: measpy.signal.Signal

        """
        if self.fs != other.fs:
            raise Exception(
                'Incompatible sampling frequencies in convolution of two signals')
        return self.similar(
            values = convolve(self.values,other.values),
            desc = self.desc+' convolved with '+other.desc,
            unit = self.unit*other.unit,
            cal = None,
            dbfs = None
        )

    def delay(self,dt):
        """
        Returns a delayed signal by dt. The data arrays are not changed.
        Instead dt is added to the property t0.

        :param dt: Delay time
        :type dt: float

        :return: The delayed signal
        :rtype: measpy.signal.Signal
        """
        return self.similar(t0=self.t0+dt)

    # #################################################################
    # Methods that return an object of type Spectral
    # #################################################################

    def fft(self, norm="backward"):
        """ FFT of the signal.
            Returns a Spectral object. Unit is preserved during the process.
        """
        return Spectral(values=np.fft.fft(self.values, norm=norm),
                        fs=self.fs,
                        unit=self.unit*Unit('s'),
                        full=True,
                        norm=norm,
                        desc=add_step(self.desc, 'FFT'))

    def rfft(self, norm="backward"):
        """ Real FFT of the signal.
            Returns a Spectral object. Unit is preserved during the process.
        """
        odd = np.mod(self.length, 2) == 1
        return Spectral(values=np.fft.rfft(self.values, norm=norm),
                        fs=self.fs,
                        unit=self.unit*Unit('s'),
                        full=False,
                        norm=norm,
                        desc=add_step(self.desc, 'RFFT'),
                        odd=odd)

    def tfe_welch(self, x, **kwargs):
        """ Compute transfer function between signal x and the actual signal. Optional parameters are the same as scipy.signal.csd or scipy.signal.welch

        :param x: Other signal from which the transfert function is computed
        :type x: measpy.signal.Signal
        :param window: Window function type, default 'hann'
        :type window: str, optional
        :param nperseg: Length of each segment. Defaults to the power of two closest to one second duration, so that the frequency spacing is approx. 1Hz.
        :type nperseg: int, optional
        :param noverlap: Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
        :type noverlab: int, optional
        :param nfft: Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
        :type nfft: int or None, optional
        :param detrend: Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to the detrend function. If it is a function, it takes a segment and returns a detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.
        :type detrend: str or function or False, optional
        :param return_onesided: If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned. Defaults to True.
        :type return_onesided: bool, optional
        :param scaling: Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’
        :type scaling: str, optional
        :param average: Method to use when averaging periodograms. Defaults to ‘mean’.
        :type average: str, optional

        :return: A Spectral object
        """
        if self.fs != x.fs:
            raise Exception('Sampling frequencies have to be the same')
        if self.length != x.length:
            raise Exception('Lengths have to be the same')

        # Set default values for welch's kwargs
        if not "fs" in kwargs:
            kwargs["fs"] = self.fs
        if not "nperseg" in kwargs:
            kwargs["nperseg"] = 2**(np.ceil(np.log2(self.fs)))

        return Spectral(
            values=csd(x.values, self.values, **
                       kwargs)[1]/welch(x.values, **kwargs)[1],
            desc='Transfer function between '+x.desc+' and '+self.desc,
            fs=self.fs,
            unit=self.unit/x.unit,
            full=False
        )

    def coh(self, x, **kwargs):
        """ Compute the coherence between signal x and the actual signal

        :param x: Other signal to compute the coherence with
        :type x: measpy.signal.Signal
        :param window: Window function type, default 'hann'
        :type window: str, optional
        :param nperseg: Length of each segment. Defaults to the power of two closest to one second duration, so that the frequency spacing is approx. 1Hz.
        :type nperseg: int, optional
        :param noverlap: Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
        :type noverlab: int, optional
        :param nfft: Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
        :type nfft: int or None, optional
        :param detrend: Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to the detrend function. If it is a function, it takes a segment and returns a detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.
        :type detrend: str or function or False, optional
        :param return_onesided: If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned. Defaults to True.
        :type return_onesided: bool, optional
        :param scaling: Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’
        :type scaling: str, optional
        :param average: Method to use when averaging periodograms. Defaults to ‘mean’.
        :type average: str, optional

        :return: A Spectral object

        """
        if self.fs != x.fs:
            raise Exception('Sampling frequencies have to be the same')
        if self.length != x.length:
            raise Exception('Lengths have to be the same')

        return Spectral(
            values=coherence(self.values, x.values, **kwargs)[1],
            desc='Coherence between '+x.desc+' and '+self.desc,
            fs=self.fs,
            unit=self.unit/x.unit,
            full=False
        )

    def psd(self, **kwargs):
        """ Compute power spectral density of the signal object
            Optional arguments are the same as the welch function
            in scipy.signal

            Optional arguments are the same as scipy.welch()

            :param window: Window function type, default 'hann'
            :type window: str, optional
            :param nperseg: Length of each segment. Defaults to the power of two closest to one second duration, so that the frequency spacing is approx. 1Hz.
            :type nperseg: int, optional
            :param noverlap: Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
            :type noverlab: int, optional
            :param nfft: Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
            :type nfft: int or None, optional
            :param detrend: Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to the detrend function. If it is a function, it takes a segment and returns a detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.
            :type detrend: str or function or False, optional
            :param return_onesided: If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned. Defaults to True.
            :type return_onesided: bool, optional
            :param scaling: Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’
            :type scaling: str, optional
            :param average: Method to use when averaging periodograms. Defaults to ‘mean’.
            :type average: str, optional

            :return: A Spectral object containing the psd
            :rtype: measpy.signal.Spectral

        """

        # Set default values for welch's kwargs
        if not "fs" in kwargs:
            kwargs["fs"] = self.fs
        if not "nperseg" in kwargs:
            kwargs["nperseg"] = 2**(np.ceil(np.log2(self.fs)))

        return Spectral(
            values=welch(self.values, **kwargs)[1],
            desc=add_step(self.desc, 'PSD'),
            fs=self.fs,
            unit=self.unit**2*Unit('s')
        )

    def tfe_farina(self, freqs, in_unit='V'):
        """
        Compute the transfer function between x and the actual signal
        where x is was a logarithmic sweep of same duration between freqs[0] and freqs[1]
        (i.e. created with measpy.signal.Signal.log_sweep)

        :param freqs: The start and stop frequencies of the input logarithmic sweep whose actual signal is the response
        :type freqs: Tuple of floats
        :param in_unit: Unit of the input signal. Defaults to 'V'
        :type unit: str or unyt.Unit
        :return: The FRF calculated by the Farina's method (2000)
        :rtype: measpy.signal.Spectral
        """
        leng = int(2**np.ceil(np.log2(self.length)))
        Y = np.fft.rfft(self.values, leng)/self.fs
        f = np.linspace(0, self.fs/2, num=round(leng/2)+1)  # frequency axis
        L = (self.length-1)/self.fs/np.log(freqs[1]/freqs[0])
        S = 2*np.sqrt(f/L)*np.exp(-1j*2*np.pi*f*L *
                                  (1-np.log(f/freqs[0])) + 1j*np.pi/4)
        S[0] = 0j
        return Spectral(values=Y*S,
                        desc='Transfer function between input log sweep and '+self.desc,
                        unit=self.unit/Unit(in_unit),
                        fs=self.fs,
                        full=False
                        )

    #######################################################################
    # Classmethods
    #####################################################################

    @classmethod
    def noise(cls, fs=44100, dur=2.0, amp=1.0, freq_min = 20.0, freq_max=20000.0, unit=None, cal=None, dbfs=None, desc=None):
        """
        ogarithmic sweep signal creation

        :param fs: Sampling frequency. Defaults to 44100.
        :param dur: Duration in seconds. Defaults to 2.0.
        :param amp: Amplitude. Defaults to 1.0.
        :param freq_min: Start frequency. Defaults to 20.0.
        :param freq_max: Stop frequency. Defaults to 20000.0.
        :param unit: Unit of the generated signal. Defaults to None (->dimensionless)
        :param cal: Calibration. Defaults to None (->1).
        :param dbfs: Zero dB full scale value. Defaults to None (->1).
        :param desc: Description of the generated signal. Defaults to None, so that the default description is 'Noise freq_min-freq-max.

        :return: A noise signal
        :rtype: measpy.signal.Signal
        """
        if desc==None:
            desc = 'Noise '+str(freq_min)+'-'+str(freq_max)+'Hz'
        return cls(
            raw=noise(fs, dur, amp, freq_min, freq_max),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc=str(desc),
            freq_min = freq_min,
            freq_max = freq_max
        )

    @classmethod
    def log_sweep(cls, fs=44100, dur=2.0, amp=1.0, freq_min = 20.0, freq_max=20000.0, unit=None, cal=None, dbfs=None, desc=None):
        """
        ogarithmic sweep signal creation

        :param fs: Sampling frequency. Defaults to 44100.
        :param dur: Duration in seconds. Defaults to 2.0.
        :param amp: Amplitude. Defaults to 1.0.
        :param freq_min: Start frequency. Defaults to 20.0.
        :param freq_max: Stop frequency. Defaults to 20000.0.
        :param unit: Unit of the generated signal. Defaults to None (->dimensionless)
        :param cal: Calibration. Defaults to None (->1).
        :param dbfs: Zero dB full scale value. Defaults to None (->1).
        :param desc: Description of the generated signal. Defaults to None, so that the default description is 'Logsweep freq_min-freq-max.

        :return: A sweep signal
        :rtype: measpy.signal.Signal
        """
        if desc==None:
            desc = 'Logsweep '+str(freq_min)+'-'+str(freq_max)+'Hz'
        return cls(
            raw=log_sweep(fs, dur, amp, freq_min, freq_max),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc=str(desc),
            freq_min = freq_min,
            freq_max = freq_max
        )

    @classmethod
    def sine(cls, fs=44100, dur=2.0, amp=1.0, freq=1000.0, unit=None, cal=None, dbfs=None, desc=None):
        if desc==None:
            desc = 'Sine '+str(freq)+'Hz'
        return cls(
            raw=sine(fs, dur, amp, freq),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc=str(desc),
            freq=freq
        )
    
    @classmethod
    def saw(cls, fs=44100, dur=2.0, amp=1.0, freq=1000.0, unit=None, cal=None, dbfs=None, desc=None):
        if desc==None:
            desc = 'Saw '+str(freq)+'Hz'
        return cls(
            raw=saw(fs, dur, amp, freq),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc=str(desc),
            freq=freq
        )
    
    @classmethod
    def tri(cls, fs=44100, dur=2.0, amp=1.0, freq=1000.0, unit=None, cal=None, dbfs=None, desc=None):
        if desc==None:
            desc = 'Tri '+str(freq)+'Hz'
        return cls(
            raw=tri(fs, dur, amp, freq),
            fs=fs,
            unit=unit,
            cal=cal,
            dbfs=dbfs,
            desc=str(desc),
            freq=freq
        )

    @classmethod
    def from_csvwav(cls, filename):
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
                if row[0] == '_unit' or row[0] == 'unit':
                    out.__dict__[row[0]] = Unit(row[1])
                elif len(row) < 3:
                    try:
                        out.__dict__[row[0]] = float(row[1])
                    except:
                        out.__dict__[row[0]] = row[1]
                else:
                    out.__dict__[row[0]] = row[1:]
        _, out._rawvalues = wav.read(filename+'.wav')
        return out

    @classmethod
    def from_wav(cls, filename, **kwargs):
        """ Load a signal from a wav file

        :param filename: base file name
        :type filename: str
        :return: The loaded signal
        :rtype: measpy.signal.Signal
        """

        desc = kwargs.setdefault("desc", filename)
        unit = kwargs.setdefault("unit", None)
        cal = kwargs.setdefault("cal", None)
        dbfs = kwargs.setdefault("dbfs", None)
        out = cls(desc=desc, unit=unit, cal=cal, dbfs=dbfs)
        out.fs, out.raw = wav.read(filename)
        return out

    #######################################################################
    # Properties
    #####################################################################

    @property
    def unit(self):
        """
        Physical unit of the signal
        """
        if hasattr(self,'_unit'):
            return self._unit
        else:
            return Unit('1')   
    @unit.setter
    def unit(self,val):
        if val==None:
            try:
                del(self._unit)
            except:
                pass
        elif Unit(val)==Unit('1'):
            try:
                del(self._unit)
            except:
                pass
        else:
            self._unit = Unit(val)

    @property
    def dbfs(self):
        """
        dbfs properties specifies the ratio between voltage signal and actual recorded signal
        """
        if hasattr(self,'_dbfs'):
            return self._dbfs
        else:
            return 1.0   
    @dbfs.setter
    def dbfs(self,val):
        if val==None:
            try:
                del(self._dbfs)
            except:
                pass
        else:
            self._dbfs = val

    @property
    def cal(self):
        """
        cal property the calibration data (a value, tuple of values or a string function)
        """
        if hasattr(self,'_cal'):
            return self._cal
        else:
            return 1.0    
    @cal.setter
    def cal(self,val):
        if val==None:
            try:
                del(self._cal)
            except:
                pass
        else:
            self._cal = val

    @property
    def invcal(self):
        """
        invcal property the calibration data
        """
        if hasattr(self,'_invcal'):
            return self._invcal
        else:
            return 1.0 
    @invcal.setter
    def invcal(self,val):
        if val==None:
            try:
                del(self._invcal)
            except:
                pass
        else:
            self._invcal = val

    @property
    def raw(self):
        """
        Raw values as 1D numpy array
        """
        return self._rawvalues
    @raw.setter
    def raw(self, val):
        self._rawvalues = val

    @property
    def values(self):
        """
        Values as 1D numpy array
        """
        if isinstance(self.cal, (int, float)):
            return self._rawvalues*self.dbfs/self.cal
        elif type(self.cal) == str:
            d = {}
            d['x'] = self.raw*self.dbfs
            command = 'y='+self.cal
            exec(command, d)
            return d['y']
        else:
            print('cal property not recognized')
    @values.setter
    def values(self, val):
        if isinstance(self.cal, (int, float)):
            self._rawvalues = val*self.cal/self.dbfs
        elif type(self.cal) == str:
            if hasattr(self, 'invcal'):
                d = {'np': np, 'y': val}
                exec('x='+self.invcal, d)
                self._rawvalues = d['x']/self.dbfs
            else:
                print(
                    'cal property seems to be a function whereas no invcal property has been given: values cannot be set')

    @property
    def volts(self):
        """
        Volt values as 1D numpy array
        """
        return self.raw*self.dbfs
    @volts.setter
    def volts(self, val):
        self.raw = val/self.dbfs

    @property
    def time(self):
        """
        Time values of the signal as 1D numpy array
        """
        return create_time(self.fs, length=len(self._rawvalues))+self.t0
    @time.setter
    def time(self,val):
        raise AttributeError("Property 'time' cannot be set")

    @property
    def t0(self):
        """
        Time shifting of the signal (reads the t0 value if it exists, else 0)
        """
        if hasattr(self, '_t0'):
            return self._t0
        else:
            return 0
    @t0.setter
    def t0(self,val):
        if val==None or val==0:
            try:
                del(self._t0)
            except:
                pass
        else:
            self._t0 = val

    @property
    def mean(self):
        """
        Mean value
        """
        return np.mean(self.values)*Unit(self.unit)

    @property
    def length(self):
        """
        Length of the signal (number of samples)
        """
        return len(self.raw)
    @length.setter
    def length(self,val):
        raise AttributeError("Property 'length' cannot be set")

    @property
    def dur(oeuf):
        """
        Duration of the signal
        """
        return len(oeuf.raw)/oeuf.fs
    @dur.setter
    def dur(oeuf,val):
        raise AttributeError("Property 'dur' cannot be set")

    @property
    def max(self):
        """Max value of a signal

        :return: Max value
        :rtype: unyt.array.unyt_quantity
        """
        return max(self.values)*unyt.Unit(self.unit)
    @max.setter
    def max(self,val):
        raise AttributeError("Property 'max' cannot be set")

    @property
    def tmax(self):
        """Time at max value of a signal

        :return: Time of maximum
        :rtype: unyt.array.unyt_quantity
        """
        return self.time[argmax(self.values)]*unyt.Unit('s')
    @tmax.setter
    def tmax(self,val):
        raise AttributeError("Property 'tmax' cannot be set")

    @property
    def min(self):
        """Min value of a signal

        :return: Min value
        :rtype: unyt.array.unyt_quantity
        """
        return min(self.values)*unyt.Unit(self.unit)
    @min.setter
    def min(self,val):
        raise AttributeError("Property 'min' cannot be set")

    @property
    def rms(self):
        """ Compute the RMS of the complete Signal

            :return: A quantity
            :rtype: unyt.Quantity      
        """
        return np.sqrt(np.mean(self.values**2))*self.unit
    @rms.setter
    def rms(self,val):
        raise AttributeError("Property 'rms' cannot be set")

    # #################################################################
    # Operators
    # ###################################################################

    def _add(self, other):
        """Add two signals

        :param other: Other signal to add
        :type other: Signal
        :return: Sum of signals
        :rtype: Signal
        """

        if not self.unit.same_dimensions_as(other.unit):
            raise Exception('Incompatible units in addition of sginals')
        if self.fs != other.fs:
            raise Exception(
                'Incompatible sampling frequencies in addition of signals')
        if self.length != other.length:
            raise Exception('Incompatible signal lengths')

        return self.similar(
            values=self.values+other.unit_to(self.unit).values,
            cal=None,
            dbfs=None,
            desc=self.desc+'\n + '+other.desc
        )

    def __add__(self, other):
        """Add something to the signal

        :param other: Something to add to
        :type other: Signal, float, int, scalar quantity
        """
        if type(other) == Signal:
            return self._add(other)

        if (type(other) == float) or (type(other) == int) or (type(other) == complex) or isinstance(other, numbers.Number):
            # print('Add with a number without unit, it is considered to be of same unit')
            return self._add(
                self.similar(
                    values=np.ones_like(self.values)*other,
                    desc=str(other)
                )
            )
        if type(other) == unyt.array.unyt_quantity:
            if not self.unit.same_dimensions_as(other.units):
                raise Exception('Incompatible units in addition of sginals')
            a = other.units.get_conversion_factor(self.unit)[0]
            return self._add(
                self.similar(
                    raw=np.ones_like(self.raw)*a,
                    desc=str(other)
                )
            )
        if type(other) == np.ndarray:
            return self._add(
                self.similar(
                    value=other,
                    desc='array'
                )
            )
        if type(other) == unyt.array.unyt_array:
            return self._add(
                self.similar(
                    values=other.value,
                    unit=other.units,
                    desc='unyt array'
                )
            )
        else:
            raise Exception(
                'Incompatible type when adding something to a Signal')

    def __radd__(self, other):
        """Addition of two signals

        :param other: something else to add
        :type other: Signal, float, int, scalar quantity
        """
        return self.__add__(other)

    def __neg__(self):
        return self.similar(raw=-1*self.raw, desc='-'+self.desc)

    def __sub__(self, other):
        """Substraction of two signals

        :param other: other signal
        :type other: Signal, int, float or quantity
        """
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        """Substraction of two signals

        :param other: other signal
        :type other: Signal
        """
        return self.__neg__().__add__(other)

    def _mul(self, other):
        """Multiplication of two signals

        :param other: other signal
        :type other: Signal
        """
        if self.fs != other.fs:
            raise Exception(
                'Incompatible sampling frequencies in multiplication of signals')
        if self.length != other.length:
            raise Exception(
                'Incompatible signal lengths in multiplication of signals')

        return self.similar(
            raw=self.values*other.values,
            unit=self.unit*other.unit,
            cal=None,
            dbfs=None,
            desc=self.desc+'\n * '+other.desc
        )

    def __mul__(self, other):
        """Multiplication of two signals

        :param other: other signal
        :type other: Signal
        """
        if type(other) == Signal:
            return self._mul(other)

        if (type(other) == float) or (type(other) == int) or (type(other) == complex) or isinstance(other, numbers.Number):
            return self.similar(raw=other*self.raw, desc=str(other)+'*'+self.desc)

        if type(other) == unyt.array.unyt_quantity:
            return self._mul(
                self.similar(
                    raw=np.ones_like(self.raw)*other.v,
                    unit=other.units,
                    cal=None,
                    dbfs=None,
                    desc=str(other)
                )
            )
        if type(other) == np.ndarray:
            return self._mul(
                self.similar(
                    raw=other,
                    unit=None,
                    cal=None,
                    dbfs=None,
                    desc='array'
                )
            )
        if type(other) == unyt.array.unyt_array:
            return self._mul(
                self.similar(
                    values=other.value,
                    unit=other.units,
                    desc='unyt array'
                )
            )
        else:
            raise Exception(
                'Incompatible type when multipling something with a Signal')

    def __rmul__(self, other):
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
            cal=None,
            dbfs=None,
            desc='1/'+self.desc
        )

    def _div(self, other):
        """Division of two signals

        :param other: other signal
        :type other: Signal
        """
        # if self.fs!=other.fs:
        #     raise Exception('Incompatible sampling frequencies in addition of signals')

        return self.similar(
            raw=self.values/other.values,
            unit=self.unit/other.unit,
            cal=None,
            dbfs=None,
            desc=self.desc+' / '+other.desc
        )

    def __truediv__(self, other):
        """Division of two signals

        :param other: other signal
        :type other: Signal
        """
        if type(other) == Signal:
            if self.fs != other.fs:
                raise Exception(
                    'Incompatible sampling frequencies in division of signals')
            return self._div(other)

        if (type(other) == float) or (type(other) == int) or (type(other) == complex) or isinstance(other, numbers.Number):
            return self.similar(raw=self.raw/other, desc=self.desc+'/'+str(other))

        if type(other) == unyt.array.unyt_quantity:
            return self._div(
                self.similar(
                    raw=np.ones_like(self.raw)*other.v,
                    unit=other.units,
                    cal=None,
                    dbfs=None,
                    desc=str(other)
                )
            )
        else:
            raise Exception(
                'Incompatible type when multipling something with a Signal')

    def __rtruediv__(self, other):
        return self.__invert__().__mul__(other)

    def abs(self):
        """
        Absolute value
        
        :return: Another signal with the absolute values
        :rtype: measpy.signal.Signal
        """
        return self.similar(
            raw=np.abs(self.raw),
            desc=add_step(self.desc, "abs")
        )

    def __abs__(self):
        """Absolute value of signal

        :param other: other signal
        :type other: Signal
        """

        return self.abs()
    
    def __matmul__(self, other):
        """
        @ (matmul) operator convolves two signals

        :param other: other signal
        :type other: Signal
        :return: Another signal with the convolved signals
        :rtype: measpy.signal.Signal
        """
        return self.convolve(other)

    #####################################################################
    # Other methods
    #####################################################################

    def to_csvwav(self, filename):
        """Saves the signal into a pair of files:

        * A CSV file with the signal parameters
        * A WAV file with the raw data

        If the str parameter filename='file', the created files are file.csv and file.wav

        :param filename: string for the base file name
        :type filename: str
        """
        with open(filename+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for arg in self.__dict__.keys():
                if arg != '_rawvalues':
                    writer.writerow([arg, self.__dict__[arg]])
        wav.write(filename+'.wav', int(round(self.fs)), self.raw)

    def to_csvtxt(self, filename, datatype='raw', includetime=False):
        """Saves the signal into a pair of files:

        * A CSV file with the signal parameters
        * A TXT file with the data

        If the str parameter filename='file', the created files are file.csv and file.wav

        :param filename: string for the base file name
        :type filename: str
        :param datatype: string for the optionnal data format (defaults to 'raw')
        :type datatype: str
        :param includetime: does the txt contains a time column ?
        :type includetime: bool

        """
        with open(filename+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for arg in self.__dict__.keys():
                if arg != '_rawvalues':
                    writer.writerow([arg, self.__dict__[arg]])
        if datatype == 'raw':
            outdata = self.raw[:, None]
        elif datatype == 'volts':
            outdata = self.volts[:, None]
        elif datatype == 'values':
            outdata = self.values[:, None]
        else:
            raise Exception("'"+str(datatype) +
                            "' is not a possible choice for datatype option")
        if includetime:
            outdata = np.concatenate((self.time[:, None], outdata), 1)
        np.savetxt(filename+'_'+datatype+'.txt', outdata)
    
    def to_csv(self,filename, datatype='raw', includetime=False):
        if datatype == 'raw':
            outdata = self.raw[:, None]
        elif datatype == 'volts':
            outdata = self.volts[:, None]
        elif datatype == 'values':
            outdata = self.values[:, None]
        else:
            raise Exception("'"+str(datatype) +
                            "' is not a possible choice for datatype option")
        if includetime:
            outdata = np.concatenate((self.time[:, None], outdata), 1)
        with open(filename+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for arg in self.__dict__.keys():
                if arg != '_rawvalues':
                    writer.writerow([arg, self.__dict__[arg]])
            if includetime:
                writer.writerow(['First column is time in seconds'])
            writer.writerows(outdata)

    def harmonic_disto(self, nh=4, freq_min=20.0, freq_max=20000.0, delay=None, win_max_length=2**15, prop_before=0.25, nsmooth=24, debug_plot=False):
        """Compute the harmonic distorsion of an in/out system
        using the method proposed by Farina (2000) and adapted by
        Novak et al. (2015) to correctly estimate the phase of the
        higher harmonics.

        The signal object (```self```) is the response of a
        nonlinear system to a logarithmic sweep created with the
        ```Signal.log_sweep``` method.

        :param nh: number of harmonics, including harmonic 0 (the linear part of the response), defaults to 4
        :type nh: int, optional
        :param freqs: frequencies between which the output signal that was used sweeps, defaults to [20,20000]
        :type freqs: tuple, optional
        :param delay: the mean delay between output and input, defaults to None. If None, the delay is estimated looking at the max value of the cross correlation of the signal with the input logarithmic sweep.
        :type delay: float, optional
        :param win_max_length: Maximum window length for each harmonic Fourier analysis in number of samples. Has to be even. Defaults to 2**15. When treating higher harmonics, the window can be shortened so that there is no overlapping with the next window.
        :type delay: float, optional
        :param prop_before: Proportion of the window that is before the peak center for each harmonic content. Defaults to 0.25 (1/4th of the window is before the harmonic peak)
        :type prop_before: float, optionnal
        :param nsmooth: Parameter for 1/nsmooth smoothing before Weighting conversion, defaults to 12
        :type nsmooth: int
        :param debug_plot: Specifies if debugging plots are shown during the process, defaults to False
        :type debug_plot: bool
        :return: A four element tuple containing:
            - A dictionary of Spectral objects representing the different harmonics as function of the frequency, not frequency aligned
            - A dictionnary of Spectral objects, representing the different harmonics, smoothed and frequency aligned
            - The total harmonic distortion (THD) (Spectral object)
            - The delay between output (sent signal) and input (measure signal)
        :rtype: tuple
        """

        # Compute transfer function using Farina's method
        sp = self.tfe_farina((freq_min,freq_max))

        l = win_max_length

        # dl is the window shift for each Fourier transform computation
        # of the harmonic peaks l/4 is a standard value that keeps a part
        # of the signal before the peak itself
        dl = prop_before*l

        # Compute delay from cross correlation
        # (timelag method)
        if type(delay) == type(None):
            delay = self.timelag(Signal.log_sweep(
                fs=self.fs,
                dur=self.dur,
                freq_min=freq_min,
                freq_max=freq_max))

        # Delay calculation based on group delay (less robust)
        # if type(delay)==type(None):
        #     # Estimate the delay by calculating the mean value of
        #     # the group delay
        #     gd = sp.group_delay()
        #     delay = np.mean(
        #         gd.values[(gd.freqs > freqs[0])&(gd.freqs < freqs[1])]
        #     ) - 0.5*l/sp.fs
        # print (delay)

        # Green's function from Farina's spectrum
        G = sp.irfft()

        # Center positions of harmonics in the time signal G
        # and time shifting for phase reconstruction
        L = (self.dur-1/self.fs)/np.log(freq_max/freq_min)
        dt = L*np.log(np.arange(nh)+1)
        decal = dt*G.fs-np.ceil(dt*G.fs)
        # print("dt")
        # print(dt)
        # print("decal")
        # print(decal)
        ns = np.round((G.dur-dt+delay)*G.fs)-dl

        ts = np.take(G.time, list(map(int, list(ns))), mode='wrap')
        # print("ts")
        # print(ts)
        tf=ts.copy()
        for i,t in enumerate(ts):
            if i==0:
                tf[i] = t+l/sp.fs
            else:
                tf[i] = min(t+l/sp.fs,ts[i-1])

        if debug_plot:
            # print("tf")
            # print(tf)
            maxG = np.max(np.abs(G.values))
            axG = G.plot(label="IFFT of Farina's spectrum")
            for ii in range(nh):
                amp = (ii+1)/nh+1
                mG = maxG*amp
                axG.plot([ts[ii], ts[ii]], [-mG/10, mG/10], lw=1, c='k')
                axG.plot([tf[ii], tf[ii]], [-mG/10, mG/10], lw=1, c='k')
                axG.plot([ts[ii], tf[ii]], [mG/10, mG/10], lw=1, c='k')
                axG.plot([ts[ii], tf[ii]], [-mG/10, -mG/10], lw=1, c='k')

        Hnl = {}
        Wnl = {}
        Hfr = {}
        if debug_plot:
            a1 = sp.plot(plot_phase=False, label="Full spectrum")
        for ii in range(nh):

            # We extract each harmonic peak
            # Silence is added so that all windows are the same length
            # Then all specra have the same characteristics
            Hnl[ii] = G.cut(dur=(ts[ii], tf[ii])).add_silence((0,l/self.fs+ts[ii]-tf[ii])).rfft()

            # Phase of spectra are adjusted to compensate for various delays
            Hnl[ii] = Hnl[ii].similar(
                values=Hnl[ii].values*np.exp(-1j*Hnl[ii].freqs*2*np.pi*(-dl+decal[ii])/Hnl[ii].fs))

            # We create a weighting for each spectra
            Wnl[ii] = Hnl[ii].nth_oct_smooth_to_weight_complex(nsmooth)

            # Frequency alignment of higher harmonics
            Wnl[ii].freqs = Wnl[ii].freqs/(ii+1)

            # We create a spectrum from weighting
            Hfr[ii] = Spectral(
                fs=Hnl[ii].fs, dur=Hnl[ii].dur).similar(w=Wnl[ii])

            if debug_plot:
                Hfr[ii].plot(ax=a1, plot_phase=False,
                             label='Harmonic '+str(ii))
            # THD computation
            # THD = 100 * sqrt ( sum(squared nl harmonics)/sum(squared all harmonics))
            if ii == 1:
                thd = abs(Hfr[ii])**2
            elif ii > 1:
                thd += abs(Hfr[ii])**2
        thd = (thd**(1/2)*(abs(Hfr[0])**2+thd)**(-1/2))*100
        thd.desc = "THD (%)"
        if debug_plot:
            a2=thd.plot(plot_phase=False, dby=False,label='THD')
            a1.set_xlim((freq_min,freq_max))
            a1.legend()
            a2.set_xlim((freq_min,freq_max))
            a2.legend()

        return (Hnl, Hfr, thd, delay)

    def __repr__(self):
        out = "measpy.Signal("
        for arg in self.__dict__.keys():
            if arg == '_unit':
                out += 'unit='+str(self.__dict__[arg])+",\n"
            elif arg == '_cal':
                out += 'cal='+str(self.__dict__[arg])+",\n"
            elif arg == '_dbfs':
                out += 'dbfs='+str(self.__dict__[arg])+",\n"
            else:
                if type(self.__dict__[arg]) == str:
                    out += arg+"='"+self.__dict__[arg]+"',\n"
                else:
                    out += arg+"="+str(self.__dict__[arg])+",\n"
        out += ')'
        return out
 
    def timelag(self, x):
        """ Estimate the time delay between two correlated signals,
            by computing the time at maximum cross-correlation between
            the two signals.

            :param x: Other signal to compute the timelag with
            :type x: measpy.signal.Signal
        """
        c = self.corr(x)
        return c.time[np.argmax(c.values)]

    def plot(self, ax=None, **kwargs):
        """ Plot the signal with axes captions and labels. Except the axis specifier ax, all other optionnal arguments are passed to the matplotlib plot function.

            :param ax: specifies an existing axis object to plot on. If not specified, a new one is created and returned
            :type ax: matplotlib.axes._axes.Axes

            :return: an axes object
            :rtype: matplotlib.axes._axes.Axes
        """

        kwargs.setdefault("label", self.desc+' ['+str(self.unit.units)+']')

        if ax == None:
            _, ax = plt.subplots(1)
        ax.plot(self.time, self.values, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_position([0.1, 0.25, 0.85, 0.7])
        ax.legend(loc=(0.05, -0.32), ncol=2)
        return ax

    def spectrogram(self, ax=None, logy=False, dbvalue=False, **kwargs):
        """ Spectrogram plot of a signal
            :param logy: Logarithmic y (frequency) scale
            :type logy: bool, optional, default to False
            :param dbvalue: Amplitude in db
            :type dbvalue: bool, optional, default to False
            :param ax: an axes object to plot on
            :type ax: axis type

            Additionnal kwargs arguments are all passed scipy.signal.spectrogram function

            :return: an axes object
            :rtype: matplotlib.axes._axes.Axes
        """
        f, t, Sxx = spectrogram(self.values, self.fs, **kwargs)
        if ax == None:
            _, ax = plt.subplots(1)
        if dbvalue:
            ax.pcolormesh(t, f, 20*np.log10(Sxx), shading='gouraud')
        else:
            ax.pcolormesh(t, f, Sxx, shading='gouraud')
        ax.set_xlabel('Time (s)')
        if logy:
            ax.set_yscale('log')
        return ax

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
        :param norm: Type of normalization "backward", "ortho" or "full". See numpy.fft doc.
        :type norm: string, optionnal    

        values and dur cannot be both specified.
        If dur is given, values are initialised at 0 
    """

    def __init__(self, **kwargs):
        if ('values' in kwargs) and ('dur' in kwargs):
            raise Exception('Error: values and dur cannot be both specified.')
        values = kwargs.setdefault("values", None)
        fs = kwargs.setdefault("fs", 1)
        desc = kwargs.setdefault("desc", 'Spectral data')
        unit = kwargs.setdefault("unit", None)
        full = kwargs.setdefault("full", False)
        norm = kwargs.setdefault("norm", "backward")
        odd = kwargs.setdefault("odd", False)
        if 'dur' in kwargs:
            if full:
                self._values = np.zeros(
                    int(round(fs*kwargs['dur'])), dtype=complex)
            else:
                self._values = np.zeros(
                    int(round(fs*kwargs['dur']/2)+1), dtype=complex)
        else:
            self._values = values
        self.desc = desc
        self.unit = unit
        self.fs = fs
        self.full = full
        self.norm = norm
        self.odd = odd

    #####################################################################
    # Methods returning a Spectral object
    #####################################################################

    def similar(self, **kwargs):
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
            :type w: measpy.signal.Weighting, optionnal
            :return: A Spectral object
            :rtype: measpy.signal.Spectral

        """
        values = kwargs.setdefault("values", self.values)
        fs = kwargs.setdefault("fs", self.fs)
        desc = kwargs.setdefault("desc", self.desc)
        unit = kwargs.setdefault("unit", str(self.unit.units))
        full = kwargs.setdefault("full", self.full)
        norm = kwargs.setdefault("norm", self.norm)
        odd = kwargs.setdefault("odd", self.odd)
        out = Spectral(values=values, fs=fs, desc=desc,
                       unit=unit, full=full, norm=norm, odd=odd)
        if 'w' in kwargs:
            w = kwargs['w']
            spa = csaps(w.freqs, w.amp, smooth=0.9)
            spp = csaps(w.freqs, w.phase, smooth=0.9)
            out.values = spa(self.freqs)*np.exp(1j*spp(self.freqs))
        return out

    def nth_oct_smooth(self, n, fmin=5, fmax=20000):
        """ Nth octave smoothing
            Works on real valued spectra. For complex values,
            use nth_oct_smooth_complex.

            :param n: Ratio of smoothing (1/nth smoothing), defaults to 3
            :type n: int, optionnal
            :param fmin: Min value of the output frequencies, defaults to 5
            :type fmin: float, int, optionnal
            :param fmax: Max value of the output frequencies, defaults to 20000
            :type fmax: float, int, optionnal
            :return: A smoothed spectral object
            :rtype: measpy.signal.Spectral
        """
        return self.similar(
            w=self.nth_oct_smooth_to_weight(n, fmin=fmin, fmax=fmax),
            desc=add_step(self.desc, '1/'+str(n)+'th oct. smooth')
        ).filterout((fmin, fmax))

    def nth_oct_smooth_complex(self, n, fmin=5, fmax=20000):
        """ Nth octave smoothing
            Complex signal version 

            :param n: Ratio of smoothing (1/nth smoothing), defaults to 3
            :type n: int, optionnal
            :param fmin: Min value of the output frequencies, defaults to 5
            :type fmin: float, int, optionnal
            :param fmax: Max value of the output frequencies, defaults to 20000
            :type fmax: float, int, optionnal
            :return: A smoothed spectral object
            :rtype: measpy.signal.Spectral
        """
        return self.similar(
            w=self.nth_oct_smooth_to_weight_complex(n, fmin=fmin, fmax=fmax),
            desc=add_step(self.desc, '1/'+str(n)+'th oct. smooth')
        ).filterout((fmin, fmax))

    def filterout(self, freqsrange):
        """ Cancels values below and above a given frequency
            Returns a Spectral class object
        """
        return self.similar(
            values=self._values*(
                (self.freqs > freqsrange[0]) & (self.freqs < freqsrange[1]))
        )

    def apply_weighting(self, w, inverse=False):
        """ Applies weighting w to the spectral object

        :param inverse: If true, applies division instead of multiplication. Defaults to False.
        :type inverse: Bool, optional

        :return: New spectral object (with new unit)
        :rtype: measpy.signal.Spectral
        """
        if inverse:
            return self*(1/self.similar(w=w, unit=Unit('1'), desc=w.desc))
        else:
            return self*self.similar(w=w, unit=Unit('1'), desc=w.desc)

    def unit_to(self, unit):
        """ Converts to a new compatible unit

        :return: New spectral object (with new unit)
        :rtype: measpy.signal.Spectral
        """

        if type(unit) == str:
            unit = Unit(unit)
        if not self.unit.same_dimensions_as(unit):
            raise Exception('Incompatible units')
        a = self.unit.get_conversion_factor(unit)[0]
        return self.similar(
            values=a*self.values,
            desc=add_step(self.desc, 'Unit to '+str(unit))
        )

    def apply_dBA(self):
        """
        Apply dBA weighting

        :return: Weighted spectrum
        :rtype: measpy.signal.Spectral        
        """
        # w = Weighting.from_csv('measpy/data/dBA.csv')
        return self.apply_weighting(WDBA)

    def apply_dBC(self):
        """
        Apply dBC weighting

        :return: Weighted spectrum
        :rtype: measpy.signal.Spectral        
        """
        # w = Weighting.from_csv('measpy/data/dBC.csv')
        return self.apply_weighting(WDBC)

    def dB_SPL(self):
        """
        Convert to dB SPL (20 log10 ||P||/P0)
        Signal unit has to be compatible with Pa

        :return: Weighted spectrum
        :rtype: measpy.signal.Spectral        
        """
        return self.unit_to(Unit(PREF)).similar(
            values=20*np.log10(np.abs(self._values)/PREF.v),
            desc=add_step(self.desc, 'dB SPL')
        )

    def dB_SVL(self):
        """
        Convert to dB SVL (20 log10 ||V||/V0)
        Signal unit has to be compatible with m/s

        :return: Weighted spectrum
        :rtype: measpy.signal.Spectral        
        """
        return self.unit_to(Unit(VREF)).similar(
            values=20*np.log10(np.abs(self._values)/VREF.v),
            desc=add_step(self.desc, 'dB SVL')
        )

    def dBV(self):
        """
        Convert to dB dBV
        Signal unit has to be compatible with Volts

        :return: Weighted spectrum
        :rtype: measpy.signal.Spectral        
        """
        return self.unit_to(Unit(PREF)).similar(
            values=20*np.log10(np.abs(self._values)/DBVREF.v),
            desc=add_step(self.desc, 'dBV')
        )

    def dBu(self):
        """
        Convert to dB dBu
        Signal unit has to be compatible with Volts

        :return: Weighted spectrum
        :rtype: measpy.signal.Spectral        
        """
        return self.unit_to(Unit(PREF)).similar(
            values=20*np.log10(np.abs(self._values)/DBUREF.v),
            desc=add_step(self.desc, 'dBu')
        )

    def diff(self):
        """ Compute frequency derivative

        :return: Frequency derivative of spectral (unit/Hz)
        :rtype: measpy.signal.Spectral
        """
        return self.similar(values=np.diff(self.values)*self.dur, unit=self.unit/Unit('Hz'), desc=add_step(self.desc, 'diff'))

    def group_delay(self):
        """ Compute group delay

        :return: Group delay (s)
        :rtype: measpy.signal.Spectral
        """
        return self.similar(
            values=(self.angle().diff()/(-2)/np.pi).values,
            unit='s',
            desc='Group delay of '+self.desc
        )

    def real(self):
        """ Real part

        :return: Real part (same unit)
        :rtype: measpy.signal.Spectral
        """
        return self.similar(
            values=np.real(self.values),
            desc=add_step(self.desc, "Real part")
        )

    def imag(self):
        """ Imaginary part

        :return: Real part (same unit)
        :rtype: measpy.signal.Spectral
        """
        return self.similar(
            values=np.real(self.values),
            desc=add_step(self.desc, "Imaginary part")
        )

    def angle(self, unwrap=True):
        """ Compute the angle of the spectrum

        :param unwrap: If True, the angle data is unwrapped
        :type unwrap: bool

        :return: The angle part of the signal, unit=rad
        :rtype: measpy.signal  

        """
        vals = np.angle(self.values)
        if unwrap:
            vals = np.unwrap(vals)
            desc = add_step(self.desc, "Angle (unwraped)")
        else:
            desc = add_step(self.desc, "Angle")
        return self.similar(
            values=vals,
            desc=desc,
            unit='rad'
        )

    #####################################################################
    # Mehtods returning a Signal
    #####################################################################

    def irfft(self, l=None):
        """ Compute the real inverse Fourier transform
            of the spectral data set
        """
        if self.full:
            raise Exception('Error: the spectrum is full, use ifft instead')
        return Signal(raw=np.fft.irfft(self.values, n=self.sample_number, norm=self.norm),
                      desc=add_step(self.desc, 'IFFT'),
                      fs=self.fs,
                      unit=self.unit/Unit('s'))

    def ifft(self):
        """ Compute the inverse Fourier transform
            of the spectral data set
        """
        if not (self.full):
            raise Exception(
                'Error: the spectrum is not full, use irfft instead')
        return Signal(raw=np.fft.ifft(self.values, norm=self.norm),
                      desc=add_step(self.desc, 'IFFT'),
                      fs=self.fs,
                      unit=self.unit/Unit('s'))

    #####################################################################
    # Mehtods returning a Weighting object
    #####################################################################

    def nth_oct_smooth_to_weight(self, n=3, fmin=5, fmax=20000):
        """ Nth octave smoothing
            Works on real valued spectra. For complex values,
            use nth_oct_smooth_to_weight_complex.

            Converts a Spectral object into a Weighting object
            (a series of frequencies logarithmically spaced,
            with a corresponding complex value, expressed as
            amplitude and phase)

            :param n: Ratio of smoothing (1/nth smoothing), defaults to 3
            :type n: int, optionnal
            :param fmin: Min value of the output frequencies, defaults to 5
            :type fmin: float, int, optionnal
            :param fmax: Max value of the output frequencies, defaults to 20000
            :type fmax: float, int, optionnal
        """
        fc, f1, f2 = nth_octave_bands(n, fmin=fmin, fmax=fmax)
        val = np.zeros_like(fc)
        for ii in range(len(fc)):
            val[ii] = np.mean(
                self.values[(self.freqs > f1[ii]) & (self.freqs < f2[ii])]
            )
        # Check for NaN values (generally at low frequencies)
        # and remove the values
        itor = []
        for ii in range(len(fc)):
            if val[ii] != val[ii]:
                itor += [ii]
        fc = np.delete(fc, itor)
        val = np.delete(val, itor)
        return Weighting(
            freqs=fc,
            amp=val,
            desc=add_step(self.desc, '1/'+str(n)+'th oct. smooth')
        )

    def nth_oct_smooth_to_weight_complex(self, n, fmin=5, fmax=20000):
        """ Nth octave smoothing, complex version

            :param n: Ratio of smoothing (1/nth smoothing), defaults to 3
            :type n: int, optionnal
            :param fmin: Min value of the output frequencies, defaults to 5
            :type fmin: float, int, optionnal
            :param fmax: Max value of the output frequencies, defaults to 20000
            :type fmax: float, int, optionnal
            :return: A weighting object
            :rtype: measpy.signal.Weighting
        """
        fc, f1, f2 = nth_octave_bands(n, fmin=fmin, fmax=fmax)
        ampl = np.zeros_like(fc, dtype=float)
        phas = np.zeros_like(fc, dtype=float)
        angles = np.unwrap(np.angle(self.values))
        for ii in range(len(fc)):
            ampl[ii] = np.mean(
                np.abs(self.values[(self.freqs > f1[ii])
                       & (self.freqs < f2[ii])])
            )
            phas[ii] = np.mean(
                angles[(self.freqs > f1[ii]) & (self.freqs < f2[ii])]
            )

        # Check for NaN values (generally at low frequencies)
        # and remove the values
        itor = []
        for ii in range(len(fc)):
            if ampl[ii] != ampl[ii]:
                itor += [ii]
        fc = np.delete(fc, itor)
        ampl = np.delete(ampl, itor)
        phas = np.delete(phas, itor)

        return Weighting(
            freqs=fc,
            amp=ampl,
            phase=phas,
            desc=add_step(self.desc, '1/'+str(n)+'th oct. smooth (complex)')
        )

    #####################################################################
    # Operators
    #####################################################################

    def _add(self, other):
        """Add two spectra

        :param other: Other Spectral to add
        :type other: Spectral
        :return: Sum of spectra
        :rtype: Spectral
        """

        if not self.unit.same_dimensions_as(other.unit):
            raise Exception(
                'Incompatible units in addition of Spectral obk=jects')
        if self.fs != other.fs:
            raise Exception(
                'Incompatible sampling frequencies in addition of Spectral objects')
        if self.length != other.length:
            raise Exception('Incompatible lengths')
        if self.full != other.full:
            raise Exception(
                'Spectral objects are not of the same type (full property)')

        return self.similar(
            values=self.values+other.unit_to(self.unit).values,
            desc=self.desc+'\n + '+other.desc
        )

    def __add__(self, other):
        """Add something to the spectrum

        :param other: Something to add to
        :type other: Spectral, float, int, scalar quantity
        """
        if type(other) == Spectral:
            return self._add(other)

        if (type(other) == float) or (type(other) == int) or (type(other) == complex) or isinstance(other, numbers.Number):
            print('Add with a number without unit, it is considered to be of same unit')
            return self._add(
                self.similar(
                    values=np.ones_like(self.values)*other,
                    desc=str(other)
                )
            )

        if type(other) == unyt.array.unyt_quantity:
            if not self.unit.same_dimensions_as(other.units):
                raise Exception('Incompatible units in addition of sginals')
            a = other.units.get_conversion_factor(self.unit)[0]
            return self._add(
                self.similar(
                    values=np.ones_like(self.values)*a,
                    desc=str(other)
                )
            )
        else:
            raise Exception(
                'Incompatible type when adding something to a Signal')

    def __radd__(self, other):
        """Addition of two Spectral objects

        :param other: something else to add
        :type other: Signal, float, int, scalar quantity
        """
        return self.__add__(other)

    def __neg__(self):
        return self.similar(values=-1*self.values, desc='-'+self.desc)

    def __sub__(self, other):
        """Substraction of two spectra

        :param other: other Spectral object
        :type other: Spectral, int, float or quantity
        """
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        """Substraction of two spectra

        :param other: other Spectral object
        :type other: Spectral,, int, float or quantity
        """
        return self.__neg__().__add__(other)

    def _mul(self, other):
        """Multiplication of two spectra

        :param other: other Spectral object
        :type other: Signal
        """
        if self.fs != other.fs:
            raise Exception(
                'Incompatible sampling frequencies in multiplication of signals')
        if self.length != other.length:
            raise Exception(
                'Incompatible signal lengths in multiplication of signals')
        if self.full != other.full:
            raise Exception(
                'Spectral objects are not of the same type (full property)')

        return self.similar(
            values=self.values*other.values,
            unit=self.unit*other.unit,
            desc=self.desc+'\n * '+other.desc
        )

    def __mul__(self, other):
        """Multiplication of two spectra

        :param other: other Spectral object
        :type other: Spectral
        """
        if type(other) == Spectral:
            return self._mul(other)

        if (type(other) == float) or (type(other) == int) or (type(other) == complex) or isinstance(other, numbers.Number):
            return self.similar(values=other*self.values, desc=str(other)+'*'+self.desc)

        if type(other) == unyt.array.unyt_quantity:
            return self._mul(
                self.similar(
                    raw=np.ones_like(self.values)*other.v,
                    unit=other.units,
                    desc=str(other)
                )
            )
        else:
            raise Exception(
                'Incompatible type when multipling something with a Signal')

    def __rmul__(self, other):
        """Multiplication of two spectra

        :param other: other Spectral object
        :type other: Spectral
        """
        return self.__mul__(other)

    def __invert__(self):
        """Spectral inverse
        """
        # Calibration and dbfs are reset to 1.0 during the process
        return self.similar(
            values=self.values**(-1),
            unit=1/self.unit,
            desc='1/'+self.desc
        )

    def _div(self, other):
        """Division of two spectra

        :param other: other spectral object
        :type other: Spectral
        """
        # if self.fs!=other.fs:
        #     raise Exception('Incompatible sampling frequencies in addition of signals')

        safe_division = np.divide(self.values, other.values, out=np.zeros_like(
            self.values), where=np.abs(other.values) != 0)

        return self.similar(
            values=safe_division,
            unit=self.unit/other.unit,
            desc=self.desc+' / '+other.desc
        )

    def __truediv__(self, other):
        """Division of two spectral objects

        :param other: other spectral object
        :type other: Spectral
        """
        if type(other) == Spectral:
            if self.fs != other.fs:
                raise Exception('Incompatible sampling frequencies')
            if self.full != other.full:
                raise Exception('Incompatible spectral types (full)')
            return self._div(other)

        if (type(other) == float) or (type(other) == int) or (type(other) == complex) or isinstance(other, numbers.Number):
            safe_division = np.divide(self.values, other, out=np.zeros_like(
                self.values), where=np.abs(other) != 0)
            return self.similar(values=safe_division, desc=self.desc+'/'+str(other))

        if type(other) == unyt.array.unyt_quantity:
            return self._div(
                self.similar(
                    values=np.ones_like(self.values)*other.v,
                    unit=other.units,
                    desc=str(other)
                )
            )
        else:
            raise Exception(
                'Incompatible type when dividing something with a Signal')

    def __rtruediv__(self, other):
        return self.__invert__().__mul__(other)

    def _abs(self):
        """ Absolute value
            Returns a Spectral class object
        """
        return self.similar(
            values=np.abs(self.values),
            desc=add_step(self.desc, "abs")
        )

    def __abs__(self):
        """Absolute value """
        return self._abs()
    
    def __pow__(self,number):
        return self.similar(values=self.values**number,
                            unit=self.unit**number,
                            cal=1.0,
                            desc=add_step(self.desc, "**"+str(number)))

    #####################################################################
    # Classmethods
    #####################################################################

    @classmethod
    def tfe(cls, x, y, **kwargs):
        """
        Initializes a spectral object by computing the transfer function between two signals of same sampling frequency and length. Optional arguments are the same as measpy.Signal.tfe_welch

        :param x: Input signal
        :type x: measpy.Signal.signal
        :param y: Output signal
        :type y: measpy.Signal.signal
        :return: A spectral object
        :rtype: measpy.Signal.spectral
        """
        if (type(x) != Signal) & (type(y) != Signal):
            raise Exception('x and y inputs have to be Signal')
        return y.tfe_welch(x, **kwargs)

    #####################################################################
    # Properties
    #####################################################################

    @property
    def values(self):
        """
        Values as 1D numpy array
        """
        return self._values

    @values.setter
    def values(self, val):
        self._values = val

    @property
    def freqs(self):
        """
        Frequencies as 1D numpy array. If the property full=True, max frequency is fs. If full=False, max frequency is fs/2 or fs*(n-1)/(2n) if the sample_number is even or odd respectively.
        """
        if self.full:
            return np.fft.fftfreq(self.sample_number, 1/self.fs)
        else:
            return np.fft.rfftfreq(self.sample_number, 1/self.fs)

    @property
    def length(self):
        """
        Length of the spectral data (i.e. number of elements in its array values or freqs properties)
        """
        return len(self._values)

    @property
    def sample_number(self):
        """
        Number of samples of the signal in time domain that corresponds to this spectral object. If the property full=True, sample_number=length. If full=False (half spectrum of a real signal), the number of samples depends on the odd property.
        """
        if self.full:
            return self.length
        else:
            return 2*self.length-1 if self.odd else 2*self.length-2

    @property
    def dur(self):
        """
        Duration of the signal in time domain that corresponds to this spectral object.
        """
        return self.sample_number/self.fs
    
    @property
    def unit(self):
        if hasattr(self,'_unit'):
            return self._unit
        else:
            return Unit('1')   
    @unit.setter
    def unit(self,val):
        if val==None:
            try:
                del(self._unit)
            except:
                pass
        elif Unit(val)==Unit('1'):
            try:
                del(self._unit)
            except:
                pass
        else:
            self._unit = Unit(val)

    #####################################################################
    # Other methods
    #####################################################################

    def values_at_freqs(self, freqlist):
        """ Get a series of values of the spectral object at
            given frequencies, using interpolation
            :param freqlist: A list of frequencies
            :type freqlist:  Number or list or Numpy array
            :return: A complex number or an array of complex numbers
        """
        spamp = np.interp(freqlist,self.freqs, abs(self.values))
        spangle = np.interp(freqlist,self.freqs, self.angle().values)
        return spamp*np.exp(1j*spangle)

    def plot(self, ax=None, logx=True, dby=True, plot_phase=True, unwrap_phase=True, unwrap_around=0, **kwargs):
        """Plot spectral data

        :param ax: Axis where to plot the data, defaults to None
        :type ax: Axis type, optional
        :param logx: If true, the frequency axis is in log scale, defaults to True
        :type logx: bool, optional
        :param dby: If true dB are plotted (20 log10 of absolute value), defaults to True
        :type dby: bool, optional
        :param plot_phase: If True, also plots the phase , defaults to True
        :type plot_phase: bool, optional
        :param unwrap_phase: If True, phase is unwrapped, defaults to True
        :type unwrap_phase: bool, optional
        :param unwrap_around: Frequency around which phase is unwrapped, defaults to 0.
        :type unwrap_around: float
        :return: An axes type object if plotphase is False, a list of two axes objects if plotphase is True
        :rtype: axes, or list of axes
        """

        kwargs.setdefault("label", self.desc+' ['+str(Unit(self.unit))+']')

        if type(ax) == type(None):
            if plot_phase:
                _, ax = plt.subplots(2)
                ax_0 = ax[0]
            else:
                _, ax = plt.subplots(1)
                ax_0 = ax
        else:
            if plot_phase:
                ax_0 = ax[0]
            else:
                ax_0 = ax

        if dby:
            if (self.unit == Unit("Pa")):
                modulus_to_plot = self.dB_SPL().values
                label = r'20 Log $|P|/P_0$ (dB)'
            elif (self.unit == Unit("m/s")):
                modulus_to_plot = self.dB_SVL().values
                label = r'20 Log $|V|/V_0$ (dB)'
            else:
                modulus_to_plot = 20*np.log10(np.abs(self.values))
                label = r'20 Log $|$H$|$ (-)'

            # Only keep finite values
            valid_indices = np.isfinite(modulus_to_plot)

            frequencies_to_plot = self.freqs[valid_indices]
            modulus_to_plot = modulus_to_plot[valid_indices]
            phase_to_plot = np.angle(self.values)[valid_indices]
            if unwrap_phase:
                if unwrap_around==0:
                    phase_to_plot = np.unwrap(phase_to_plot)
                else:
                    phase_to_plot = unwrap_around_index(phase_to_plot,get_index(self.freqs,unwrap_around))

        else:
            modulus_to_plot = np.abs(self.values)

            # Only keep positive values
            valid_indices = np.where(modulus_to_plot > 0)

            frequencies_to_plot = self.freqs[valid_indices]
            modulus_to_plot = modulus_to_plot[valid_indices]
            phase_to_plot = np.angle(self.values)[valid_indices]
            if unwrap_phase:
                if unwrap_around==0:
                    phase_to_plot = np.unwrap(phase_to_plot)
                else:
                    phase_to_plot = unwrap_around_index(phase_to_plot,get_index(self.freqs,unwrap_around))
            label = r'$|$H$|$'

        ax_0.plot(frequencies_to_plot, modulus_to_plot, **kwargs)
        ax_0.set_xlabel('Freq (Hz)')
        ax_0.set_ylabel(label)
        if logx:
            ax_0.set_xscale('log')
        if plot_phase:
            ax[1].plot(frequencies_to_plot, phase_to_plot, **kwargs)
            ax[1].set_ylabel('Phase (rad)')
            ax[1].set_xlabel('Freq (Hz)')
            if logx:
                ax[1].set_xscale('log')
        return ax
    
    def __repr__(self):
        out = "measpy.Spectral("
        for arg in self.__dict__.keys():
            if arg == '_unit':
                out += 'unit='+str(self.__dict__[arg])+",\n"
            else:
                if type(self.__dict__[arg]) == str:
                    out += arg+"='"+self.__dict__[arg]+"',\n"
                else:
                    out += arg+"="+str(self.__dict__[arg])+",\n"
        out += ')'
        return out

    #  END of Spectral

#####################
##                 ##
## Weighting Class ##
##                 ##
#####################


class Weighting:
    """ Class for weighting functions

        Amplitudes are stored as absolute values and phase (in radians)

        A Weighting object stores:

        - A list of frequencies (numpy.array)

        - Corresponding amplitudes (numpy.array)

        - Corresponding phases in radians (numpy.array)

        - A descriptor (string)       
    """

    def __init__(self, freqs, amp, phase=None, desc='Weigthing function'):
        self.freqs = freqs
        if type(phase) == type(None):
            self.phase = np.zeros_like(amp)
        else:
            self.phase = phase
        # if type(amp)==float or type(amp)==int:
        #     self.amp=float(amp)
        # elif type(amp)==complex:
        #     self.amp=np.abs(amp)
        #     self.phase=np.angle(amp)
        self.amp = amp
        self.desc = desc

    @classmethod
    def from_csv(cls, filename, asdB=True, asradians=True):
        """
        Loads a weighting object from a csv file
        The file must contain three columns:

        - One frequency column

        - One amplitude column (linear or as dB, which must be specified in the asdB boolean optional argument)

        - One phase column (as radians or degree, which must be specified in the asradians boolean optional argument)

        :param filename: File name of the csv file to load
        :type filename: str
        :param asdB: Specifies if the amplitude is given in dB or not
        :type asdB: bool
        :param asradians: Specifies if the phase is given in radians or degrees
        :type asradians: bool
        :returns: A Weighting object
        :rtype: measpy.weighting.Weighting
        """
        out = cls([], [], 'Weighting')
        out.phase = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            n = 0
            for row in reader:
                if n == 0:
                    out.desc = row[0]
                else:
                    out.freqs += [float(row[0])]
                    if asdB:
                        out.amp += [10**(float(row[1])/20.0)]
                    else:
                        out.amp += [float(row[1])]
                    if asradians:
                        try:
                            out.phase += [float(row[2])]
                        except:
                            out.phase += [0.0]
                    else:
                        try:
                            out.phase += [np.pi*float(row[2])/180.0]
                        except:
                            out.phase += [0.0]
                n += 1
        out.freqs = np.array(out.freqs)
        out.amp = np.array(out.amp)
        out.phase = np.array(out.phase)
        return out

    def to_csv(self, filename, asdB=True, asradians=True):
        """
        Saves a weighting object to a csv file
        The file then contains three columns:

        - One frequency column

        - One amplitude column (linear or as dB, which must be specified in the asdB boolean optional argument)
        
        - One phase column (as radians or degree, which must be specified in the asradians boolean optional argument)

        :param filename: File name of the csv file to load
        :type filename: str
        :param asdB: Specifies if the amplitude is given in dB or not
        :type asdB: bool
        :param asradians: Specifies if the phase is given in radians or degrees
        :type asradians: bool
        """

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.desc])
            if asdB:
                outamp = 20*np.log10(np.abs(self.amp))
            else:
                outamp = self.amp

            if asradians:
                outphase = self.phase
            else:
                outphase = 180*self.phase/np.pi

            for n in range(len(self.freqs)):
                writer.writerow(
                    [self.freqs[n],
                     outamp[n],
                     outphase[n]]
                )

    @property
    def adb(self):
        """
        Amplitude in dB
        Computes 20 log10 of the modulus of the amplitude 
        """
        return 20*np.log10(np.abs(self.amp))

    @property
    def acomplex(self):
        """
        Weighting values represented as a complex number
        """
        return self.amp*np.exp(1j*self.phase)

    # END of Weighting


# Constants

PREF = 20e-6*Unit('Pa')  # Acoustic pressure reference level
VREF = 5e-8*Unit('m/s')  # Reference particle velocity
DBUREF = 1*Unit('V')
DBVREF = np.sqrt(2)*Unit('V')

WDBA = [
    [6.3, -85.4],
    [8, -77.8],
    [10, -70.4],
    [12.5, -63.4],
    [16, -56.7],
    [20, -50.5],
    [25, -44.7],
    [31.5, -39.4],
    [40, -34.6],
    [50, -30.2],
    [63, -26.2],
    [80, -22.5],
    [100, -19.1],
    [125, -16.1],
    [160, -13.4],
    [200, -10.9],
    [250, -8.6],
    [315, -6.6],
    [400, -4.8],
    [500, -3.2],
    [630, -1.9],
    [800, -0.8],
    [1000, 0],
    [1250, 0.6],
    [1600, 1.0],
    [2000, 1.2],
    [2500, 1.3],
    [3150, 1.2],
    [4000, 1.0],
    [5000, 0.5],
    [6300, -0.1],
    [8000, -1.1],
    [10000, -2.5],
    [12500, -4.3],
    [16000, -6.6],
    [20000, -9.3]]
WDBA = Weighting(
    freqs=np.array(WDBA)[:, 0],
    amp=10**(np.array(WDBA)[:, 1]/20),
    desc='dBA weightings')

WDBC = [
    [6.3, -21.3],
    [8, -17.7],
    [10, -14.3],
    [12.5, -11.2],
    [16, -8.5],
    [20, -6.2],
    [25, -4.4],
    [31.5, -3.0],
    [40, -2.0],
    [50, -1.3],
    [63, -0.8],
    [80, -0.5],
    [100, -0.3],
    [125, -0.2],
    [160, -0.1],
    [200, 0.0],
    [250, 0.0],
    [315, 0.0],
    [400, 0.0],
    [500, 0.0],
    [630, 0.0],
    [800, 0.0],
    [1000, 0.0],
    [1250, 0.0],
    [1600, -0.1],
    [2000, -0.2],
    [2500, -0.3],
    [3150, -0.5],
    [4000, -0.8],
    [5000, -1.3],
    [6300, -2.0],
    [8000, -3.0],
    [10000, -4.4],
    [12500, -6.2],
    [16000, -8.5],
    [20000, -11.2]]
WDBC = Weighting(
    freqs=np.array(WDBC)[:, 0],
    amp=10**(np.array(WDBC)[:, 1]/20),
    desc='dBC weightings')

WDBM=[[31.5 , -29.9],[63 , -23.9],[100 , -19.8],[200 , -13.8],[400 , -7.8],[800 , -1.9],[1_000 , 0.0],[2_000 , +5.6],[3_150 , +9.0],[4_000 , +10.5],[5_000 , +11.7],[6_300 , +12.2],[7_100 , +12.0],[8_000 , +11.4],[9_000 , +10.1],[10_000 , +8.1],[12_500 , 0.0],[14_000 , -5.3],[16_000 , -11.7],[20_000 , -22.2],[31_500 , -42.7]]
WDBM = Weighting(
    freqs=np.array(WDBM)[:, 0],
    amp=10**(np.array(WDBM)[:, 1]/20),
    desc='M-weighting')


# Below are functions that may be useful (some cleaning should be done)



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
