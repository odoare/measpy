# measpy/signalgroup.py
#
# --------------------------------------------------
# measpy package: module defining Signal_group class
# --------------------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

import os
import copy
import csv
from matplotlib.pyplot import title
from .signal import Signal
from ._version import get_versions
__version__ = get_versions()['version']
from ._tools import (csv_to_dict,
                        convl1)

known_group_len = {
    'Stereo' : 2,
    'Binau' : 2,
    'XY' : 2,
    'AmbiA' : 4,
    'AmbiB' : 4,
    '5.1' : 6
    }

known_functions = ['smooth',
                    'rms_smooth',
                    'iir',
                    'as_volts',
                    'as_raw',
                    'hilbert',
                    'hilbert_ana',
                    'normalize',
                    'diff',
                    'real',
                    'imag',
                    'angle',
                    'delay',
                    'dB',
                    'dB_SPL',
                    'dB_SVL',
                    'resample',
                    'cut',
                    'fade',
                    'add_silence',
                    'unit_to',
                    'unit_to_std',
                    'window']

#known_properties = ['length', 'dur']

class SignalGroup:
    """ Signal group
    """
    def __init__(self,**kwargs):

        # if 'sigs' not in kwargs:
        #     raise Exception('No signal list provided')
        if 'group_type' in kwargs:
            if kwargs['group_type'] in known_group_len:
                if 'sigs' in kwargs:
                    if known_group_len[kwargs['group_type']]!=len(kwargs['sigs']):
                        raise ValueError('Number of signals in siglist does not correspond to group type "'+kwargs['group_type']+'"')
            self.group_type = kwargs.pop('group_type')
        else:
            self.group_type=None

        if 'sigs' in kwargs:
            self.sigs = kwargs.pop('sigs')
        else:
            if 'group_type' in kwargs:
                if self.group_type in known_group_len:
                    self.sigs = list(Signal() for i in range(known_group_len[self.group_type]))
            elif 'nsigs' in kwargs:
                self.sigs = list(Signal() for i in range(kwargs.pop('nsigs')))
            else:
                self.sigs=[]

        if 'desc' in kwargs:
            self.desc = kwargs.pop('desc')
        else:
            self.desc = 'Signal group'

        for key,value in kwargs.items():
            self.__dict__[key] = value

        for f in known_functions:
            setattr(self,f, lambda f=f, **kwargs : self.apply_function(f,**kwargs))

    def apply_function(self,func,**kwargs):
        """ Applies a Signal function to all signals of a SignalGroup instance
        """
        out = copy.deepcopy(self)
        for i,s in enumerate(self.sigs):
            fun = getattr(Signal,func)
            out.sigs[i] = fun(s,**kwargs)
        return out

    def plot(self,**kwargs):
        """ Plots all signals of a SignalGroup instance
        """
        for s in self.sigs:
            if 'a' in locals():
                a = s.plot(ax=a, **kwargs)
                title (self.desc)
            else:
                a = s.plot(**kwargs)

    # -----------------------
    def to_dir(self,dirname):
        """ Stores the parameters and signals in a directory

            :param dirname: Name of the directory, a (1), (2)...
                            is added to the name if directory exists
            :type dirname: str
            :return: Actual name to the saved folder (if name conflit is detected)
            :rtype: str                            
        """
        if os.path.exists(dirname):
            i = 1
            while os.path.exists(dirname+'('+str(i)+')'):
                i+=1
            dirname = dirname+'('+str(i)+')'
        os.mkdir(dirname)
        self._params_to_csv(dirname+"/params.csv")
        if not isinstance(self.sigs,type(None)):
        # type(self.sigs)!=type(None):
            for i,s in enumerate(self.sigs):
                s.to_csvwav(dirname+"/sigs_"+str(i))
        self._write_readme(dirname+"/README")
        return dirname

    # ------------------------------
    def _to_dict(self,withsig=True):
        """ Converts a SignalGroup object to a dict

            :param withsig: Optionnally removes the data arrays, defaults to True
            :type withsig: bool
            
        """
        group = copy.deepcopy(self.__dict__)
        if not withsig:
            del group['sigs']
            for f in known_functions:
                del group[f]
        return group

    # --------------------------------
    def _params_to_csv(self,filename):
        """ Writes all the SignalGroup object parameters to a csv file """
        dd = self._to_dict(withsig=False)
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for key in dd:
                if type(dd[key])==list:
                    writer.writerow([key]+dd[key])
                else:
                    writer.writerow([key,str(dd[key])])

    # -------------------------------
    def _write_readme(self,filename):
        with open(filename, 'w') as f:
            f.write('Signal group directory\n\n')
            f.write('Created with measpy version '+__version__)
            f.write('\n')
            f.write('https://github.com/odoare/measpy')

    @classmethod
    def from_dir(cls,dirname):
        """ Load a SignalGroup object from a directory

            :param dirname: Name of the directory
            :type dirname: str                
        """
        group_dict = csv_to_dict(dirname+'/params.csv')
        self = cls._from_dict(group_dict)
        more_files = True
        file_ind = 0
        self.sigs=[]
        while more_files:
            try:
                self.sigs.append(Signal.from_csvwav(dirname+'/sigs_'+str(file_ind)))
            except FileNotFoundError:
                more_files = False
            file_ind += 1
        return self

    @classmethod
    def _from_dict(cls, task_dict):
        self=cls()
        for k in task_dict.keys():
            if k=='desc':
                self.desc = convl1(str,task_dict['desc'])
            elif k=='group_type':
                self.group_type = convl1(str,task_dict['group_type'])
            else:
                self.__dict__[k]=task_dict[k]
        return self

    def __getitem__(self,i):
        """ We redefine __getitem__ so that a Signal_group instance
        can be accessed as if it is a list a Signal instances
        """
        return self.sigs[i]

    def __setitem__(self,i,value):
        """ We redefine __setitem__ so that a Signal_group instance
        can be accessed as if it is a list a Signal instances
        """
        self.sigs[i] = value

    def __len__(self):
        return len(self.sigs)

    @property
    def nsigs(self):
        return len(self.sigs)
    @nsigs.setter
    def nsigs(self,values):
        raise AttributeError('Property nsigs cannot be set.')

    @property
    def length(self):
        return [s.length for s in self.sigs]

    @property
    def dur(self):
        return [s.dur for s in self.sigs]

    @property
    def fs(self):
        return [s.fs for s in self.sigs]

    @property
    def unit(self):
        return [s.unit for s in self.sigs]
    
    @property
    def dbfs(self):
        return [s.dbfs for s in self.sigs]

    def append(self,sig):
        self.sigs.append(sig)

    def __repr__(self):
        outstring = 'measpy.SignalGroup(\n'
        outstring += "  desc='"+str(self.desc)+"',\n"
        outstring += "  grouptype='"+str(self.group_type)+"',\n"
        outstring += '  sigs='
        outstring += self.sigs.__repr__()
        outstring += ')'
        return outstring

    # def to_csvwav(self,filename):
    #     """Saves the signal into a pair of files:

    #     * A CSV file with the signal parameters
    #     * A WAV file with the raw data

    #     If the str parameter filename='file', the created files are file.csv and file.wav

    #     :param filename: string for the base file name
    #     :type filename: str
    #     """
    #     with open(filename+'.csv', 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         for arg in self.__dict__.keys():
    #             if arg != 'sigs':
    #                 writer.writerow([arg, self.__dict__[arg]])
    #     siglist_to_wav(filename+'.wav', int(round(self.fs)), self.raw)
