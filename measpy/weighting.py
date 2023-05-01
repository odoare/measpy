# measpy/weighting.py
#
# -------------------------------------
# This file defines the Weighting class
# Namespace : measpy.weighting.Weighting
# --------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

from warnings import WarningMessage
import numpy as np
import scipy.io.wavfile as wav
import csv

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
        out = cls([], [], 'Weigting')
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
