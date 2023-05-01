# measpy/qpectral.py
#
# ------------------------------------
# This file defines the Spectral class
# Namespace : measpy.spectral.Spectral
# ------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy


from warnings import WarningMessage
import numpy as np
import matplotlib.pyplot as plt

from csaps import csaps
import numbers

import unyt
from unyt import Unit

from ._tools import (add_step,
                           nth_octave_bands)

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
        unit = kwargs.setdefault("unit", '1')
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
        self.unit = Unit(unit)
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
        Convert to dB SPL (20 log10 |P|/P0)
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
        Convert to dB SVL (20 log10 |V|/V0)
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
                      unit=self.unit)

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
                      unit=self.unit)

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

    def plot(self, ax=None, logx=True, dby=True, plot_phase=True, unwrap_phase=True, **kwargs):
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
        :return: An axes type object if plotphase is False, a list of two axes objects if plotphase is True
        :rtype: axes, or list of axes
        """

        kwargs.setdefault("label", self.desc+' ['+str(self.unit.units)+']')

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
                label = '20 Log |P|/P0'
            elif (self.unit == Unit("m/s")):
                modulus_to_plot = self.dB_SVL().values
                label = '20 Log |V|/V0'
            else:
                modulus_to_plot = 20*np.log10(np.abs(self.values))
                label = '20 Log |H|'

            # Only keep finite values
            valid_indices = np.isfinite(modulus_to_plot)

            frequencies_to_plot = self.freqs[valid_indices]
            modulus_to_plot = modulus_to_plot[valid_indices]
            phase_to_plot = np.angle(self.values)[valid_indices]
            if unwrap_phase:
                phase_to_plot = np.unwrap(phase_to_plot)

        else:
            modulus_to_plot = np.abs(self.values)

            # Only keep positive values
            valid_indices = np.where(modulus_to_plot > 0)

            frequencies_to_plot = self.freqs[valid_indices]
            modulus_to_plot = modulus_to_plot[valid_indices]
            phase_to_plot = np.angle(self.values)[valid_indices]
            if unwrap_phase:
                phase_to_plot = np.unwrap(phase_to_plot)
            label = '|H|'

        ax_0.plot(frequencies_to_plot, modulus_to_plot, **kwargs)
        ax_0.set_xlabel('Freq (Hz)')
        ax_0.set_ylabel(label)
        if logx:
            ax_0.set_xscale('log')
        if plot_phase:
            ax[1].plot(frequencies_to_plot, phase_to_plot, **kwargs)
            ax[1].set_ylabel('Phase')
            ax[1].set_xlabel('Freq (Hz)')
            if logx:
                ax[1].set_xscale('log')
        return ax

    #  END of Spectral
