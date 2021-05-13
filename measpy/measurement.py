# measurement.py
# 
# A class for measurement management with data acquisition devices
#
# OD - 2021

# TODO :
# - tbefore, tafter
# - synchronisation

import measpy.signal as ms
from measpy.signal import Signal, ur
import numpy as np
import matplotlib.pyplot as plt

from copy import copy

import scipy.io.wavfile as wav
import csv
import pickle
import json

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

class Measurement:
    """ The class Measurement allows to simply define and perform
        a measurement
    """
    def __init__(self, **params):
        """ Optionnal parameters (param=default)
                fs=44100
                dur=2
                in_map=[1,2]
                in_cal=[1.0,1.0]
                in_dbfs=[1.0,1.0]
                in_unit=['V','V']
                in_desc=['In1','In2']
                out_sig='noise'
                extrat=[0.0,0.0]

            If out_sig != None:
                out_map=[1]
                out_amp=1.0
                out_dbfs=[1.0]
                out_desc=['Out1']
                out_sig_freqs=[20.0,20000.0]
                fades=[0,0]
                ioSync=0
            
            device_type=''
            device=''

            (device and device_type depend on the type of data acquisition device)
        """
        self.fs = params.setdefault("fs",44100)
        self.dur = params.setdefault("dur",2.0)
        self.in_map = params.setdefault("in_map",[1,2])
        self.in_cal = params.setdefault("in_cal",[1.0,1.0])
        self.in_dbfs = params.setdefault("in_dbfs",[1.0,1.0])
        self.in_unit = params.setdefault("in_unit",['V','V'])
        self.in_desc = params.setdefault("in_desc",['In1','In2'])
        self.out_sig = params.setdefault("out_sig",'noise')
        self.extrat = params.setdefault("extrat",[0.0,0.0])
        if self.out_sig!=None:
            self.out_map = params.setdefault("out_map",[1])
            self.out_amp =  params.setdefault("out_amp",1.0)
            self.out_dbfs = params.setdefault("out_dbfs",[1.0])
            self.out_desc =  params.setdefault("out_desc",['Out1'])
            self.out_sig_freqs =  params.setdefault("out_sig_freqs",[20.0,20000.0])
            self.io_sync = params.setdefault("io_sync",0)
            self.out_sig_fades = params.setdefault("out_sig_fades",[0,0])
        self.device_type = params.setdefault("device_type",'')
        self.device = params.setdefault("device",'')
        self.data = {}
        for n in range(len(self.out_desc)):
            self.data[self.out_desc[n]]=Signal(desc=self.out_desc[n],
                                                fs=self.fs,
                                                unit='V',
                                                cal=1.0,
                                                dbfs=self.out_dbfs[n])
        for n in range(len(self.in_desc)):
            self.data[self.in_desc[n]]=Signal(desc=self.in_desc[n],
                                                fs=self.fs,
                                                unit=self.in_unit[n],
                                                cal=self.in_cal[n],
                                                dbfs=self.in_dbfs[n])
        self.datakeys = list(self.data.keys())
        self.create_output()
        
    def create_output(self):
        if self.out_sig=='noise': # White noise output signal
            self.data[self.out_desc[0]] = self.data[self.out_desc[0]].similar(
                ms._noise(self.fs,self.dur,self.out_amp,self.out_sig_freqs)
            ).fade(self.out_sig_fades).add_silence(self.extrat)

            if self.out_map==0:
                self._out_map=[1]

        elif self.out_sig=='logsweep': # Logarithmic sweep output signal
            self.data[self.out_desc[0]] = self.data[self.out_desc[0]].similar(
                ms._log_sweep(self.fs,self.dur,self.out_amp,self.out_sig_freqs)
            ).fade(self.out_sig_fades).add_silence(self.extrat)

            if self.out_map==0:
                self.out_map=[1]

        elif self.out_sig.upper().endswith('.WAV'): # Wave file output signal

            rate, x = wav.read(self.out_sig)

            if len(x.shape)==1:
                nchan = 1
            else:
                nchan = x.shape[1]

            if (rate!=self.fs):
                print('Warning: ')
                print('  Rate of input file='+str(rate)+'Hz.')
                print("  It's different than fs="+str(self.fs)+"Hz of current measurement object.")
                print('  Changing to: '+str(rate))
                self.fs=rate
            if (x.shape[0]/self.fs!=self.dur):
                print("Warning:")
                print("  Duration of input file="+str(x.shape[0]/self.fs)+"s.")
                print("  It's different than dur="+str(self.dur)+"s of current measurement object.")
                print('  Changing to: '+str(x.shape[0]/self.fs))
                self.dur = x.shape[0]/self.fs
            if self.out_map==0:
                self.out_map=list(range(1,x.shape[1]+1))
            elif (nchan!=len(self.out_map)):
                print("Warning:")
                print("Size of out_map and number of channels do not correspond.")
                if (x.shape[1]<len(self.out_map)):
                    print("  Truncating current out_map...")
                    self.out_map=self.out_map[0:nchan]
                    self.out_desc=self.out_desc[0:nchan]
                    self.out_dbfs=self.out_dbfs[0:nchan]
                else:
                    print("  Truncating channels of the output signal...")
                    x=x[:,0:len(self.out_map)]
            if x.dtype == 'int16':
                for ii in range(len(self.out_map)):
                    self.data[self.out_desc[ii]].raw=np.array(x[:,ii],dtype=float)/32768
            elif x.dtype == 'int32':
                for ii in range(len(self.out_map)):
                    self.data[self.out_desc[ii]].raw=np.array(x[:,ii],dtype=float)/2147483648
            else:
                for ii in range(len(self.out_map)):
                    self.data[self.out_desc[ii]].raw=np.array(x[:,ii],dtype=float)

    def show(self):
        """ Pretty prints the measurement properties """
        print("Measurement with the following properties:")
        print("| Type of device: device_type="+self.device_type)
        print("| Device: device="+self.device)
        print("| Sampling frequency (Hz): fs="+str(self.fs))
        print("| Duration (s): dur="+str(self.dur))
        st = str(self.in_map)
        print("| Input map: in_map="+st)
        st = str(self.in_cal)
        print("| Input calibrations (V/Unit): in_cal="+st)
        st = str(self.in_dbfs)
        print("| Input 0dBFS (V): in_dbfs="+st)
        st = "', '".join(self.in_unit)
        print('| Input units: in_unit='+"['"+st+"']")
        st = "', '".join(self.in_desc)
        print('| Input units: in_desc='+"['"+st+"']")        
        print('| Extra time before and after: extrat='+str(self.extrat))
        if self.out_sig!=None:
            st = str(self.out_map)
            print('| Output map: out_map='+st)
            print('| Output amp: out_amp='+str(self.out_amp))
            print("| Output signal type: out_sig='"+self.out_sig+"'")
            print('| Min and max frequency of generated output: out_sig_freqs='+str(self.out_sig_freqs))
            print('| Synchonisation: ioSync='+str(self.io_sync))
            st = str(self.out_dbfs)
            print("| Output 0dBFS (V): out_dbfs="+st)
        try:
            print("| Measurement date: date='"+self.date+"'")
            print("| Measurement time: time='"+self.time+"'")
        except:
            print("| No measurement date or time, the measurement hasn't been performed")
        print("[ Contents of the dictionnary data (keys):")
        for key in self.data:
            print("| "+key)

    def __repr__(self):
        out = "measpy.Measurement("
        out += "fs="+str(self.fs)
        out += ", dur="+str(self.dur)
        out += ", device_type='"+str(self.device_type)+"'"
        out += ", device='"+str(self.device)+"'"
        out += ', in_desc='+str(self.in_desc)
        out += ', in_map='+str(self.in_map)
        out += ', in_cal='+str(self.in_cal)
        out += ', in_unit='+str(self.in_unit)
        out += ', in_dbfs='+str(self.in_dbfs)
        out += ', extrat='+str(self.extrat)
        try:
            out += ", date='"+self.date+"'"
            out += ", time='"+self.time+"'"
        except:
            pass
        if self.out_sig!=None:
            out += ', out_desc='+str(self.out_desc)
            out += ', out_map='+str(self.out_map)
            out += ', out_dbfs='+str(self.out_dbfs)
            out += ", out_sig='"+str(self.out_sig)+"'"
            out += ', out_sig_freqs='+str(self.out_sig_freqs)
            out += ', out_sig_fades='+str(self.out_sig_fades)
            out += ", out_amp="+str(self.out_amp)
            out += ", io_sync="+str(self.io_sync)+")"
        return out

    def to_dict(self,withdata=True):
        """ Converts a Measurement object to a dict
            Optionnally removes the data arrays
        """
        self.data_keys = list(self.data.keys())
        mesu = copy(self.__dict__)
        if not(withdata):
            del mesu['data']
        return mesu

    def from_dict(self,mesu):
        """ Converts a dict to a Measurement object,
            generally loaded from a file.
        """

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

        self.fs=convl1(float,mesu['fs'])
        self.dur=convl1(float,mesu['dur'])
        self.in_map=convl(int,mesu['in_map'])
        self.in_cal=convl(float,mesu['in_cal'])
        self.in_dbfs=convl(float,mesu['in_dbfs'])
        self.in_unit=convl(str,mesu['in_unit'])
        self.in_desc=convl(str,mesu['in_desc'])
        self.extrat=convl(float,mesu['extrat'])
        try:
            self.data=mesu['data']
        except:
            pass
        try:
            self.date=convl1(str,mesu['date'])
            self.time=convl1(str,mesu['time'])
        except:
            pass
        if self.out_sig!=None:
            self.out_map=convl(int,mesu['out_map'])
            self.out_amp=convl1(float,mesu['out_amp'])
            self.out_desc=convl(str,mesu['out_desc'])
            self.out_sig=convl1(str,mesu['out_sig'])
            self.out_sig_freqs=convl(float,mesu['out_sig_freqs'])
            self.out_sig_fades=convl(float,mesu['out_sig_fades'])
            self.io_sync=convl1(int,mesu['io_sync'])
            self.out_dbfs=convl(float,mesu['out_dbfs'])
        self.device=convl1(str,mesu['device'])
        self.device_type=convl1(str,mesu['device_type'])
        self.data_keys=convl(str,mesu['data_keys'])

    def to_pickle(self,filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls,filename):
        with open(filename, 'rb') as handle:
            mesu = pickle.load(handle)
        M = cls()
        M.from_dict(mesu)
        return M

    def data_to_wav(self,filename):
        n = 0
        for key in self.data.keys():
            if n==0:
                out = self.data[key].raw[:,None]
                n += 1
            else:
                out = np.block([out,self.data[key].raw[:,None]])
                n += 1
        wav.write(filename,int(round(self.fs)),out)

    def data_from_wav(self,filename):
        _, dat = wav.read(filename)
        n = 0
        for key in self.data_keys:
            self.data[key].raw = dat[:,n]
            n += 1

    def params_to_csv(self,filename):
        """ Writes all the Measurement object parameters to a csv file """
        dd = self.to_dict(withdata=False)
        #data_keys = list(self.data.keys())
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            for key in dd:
                if type(dd[key])==list:
                    writer.writerow([key]+dd[key])
                else:
                    writer.writerow([key,str(dd[key])])
            #writer.writerow(['data_keys']+data_keys)
                    
    def csv_to_params(self,filename):
        """ Load measurement parameters from a csv file """
        self.from_dict(csv_to_dict(filename))

    def params_to_json(self,filename):
        """ Writes all the Measurement object parameters to a json file """
        with open(filename, mode='w', encoding='utf-8') as f:
            json.dump(self.to_dict(withdata=False), f, indent=2)

    def json_to_params(self,filename):
        """ Load measurement parameters from a json file """
        with open(filename, encoding='utf-8') as f:
            self.from_dict(json.load(f))

    def to_csvwav(self,filebase):
        """ Saves a Measurement object to a set of files
                    filebase : string from which two file names are created
                    filebase+'.csv' : All measurement parameters
                    filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        self.params_to_csv(filebase+'.csv')
        try:
            self.data_to_wav(filebase+'.wav')
        except:
            print('data_to_wav failed (no data?)')

    @classmethod
    def from_csvwav(cls,filebase):
        """ Load a measurement object from a set of files
                    filebase : string from which two file names are created
                    filebase+'.csv' : All measurement parameters
                    filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        M=cls()
        M.csv_to_params(filebase+'.csv')
        try:
            M.data_from_wav(filebase+'.wav')
        except:
            print('data_from_wav failed (file not present?)')
        return M

    def to_jsonwav(self,filebase):
        """ Saves a Measurement object to a set of files
                    filebase : string from which two file names are created
                    filebase+'.json' : All measurement parameters
                    filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        self.params_to_json(filebase+'.json')
        try:
            self.data_to_wav(filebase+'.wav')
        except:
            print('data_to_wav failed (no data?)')

    @classmethod
    def from_jsonwav(cls,filebase):
        """ Load a measurement object from a set of files
                    filebase : string from which two file names are created
                    filebase+'.json' : All measurement parameters
                    filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        M=cls()
        M.json_to_params(filebase+'.json')
        try:
            M.data_from_wav(filebase+'.wav')
        except:
            print('data_from_wav failed (file not present?)')
        return M

    def plot_with_cal(self):
        for ii in range(self.y.shape[1]):
            plt.plot(self.t,self.y[:,ii]/self.in_cal[ii]*self.in_dbfs[ii])
        plt.plot(self.t,np.ones_like(self.t),':',color='grey')
        plt.plot(self.t,-np.ones_like(self.t),':',color='grey')
        plt.xlabel('Time(s)')
        legende = []
        for ii in range(self.y.shape[1]):
            legende+=[self.in_desc[ii]+'('+self.in_unit[ii]+')']
        legende+=['limits']
        plt.legend(legende)
        plt.title('Measurement date: '+str(self.date)+"   "+str(self.time))
        plt.grid('on',color='grey',linestyle=':')
    
    def tfe(self,nperseg=2**16,noverlap=None,plotH=False):
        """ Helper function that calculates the transfer function between
            the output channel x and all the input channels y. Works only
            if x has only one channel.
            If out_sig='logsweep', the method of Farina is used, Welch's
            method is used otherwise.
        """
        if (self.out_sig=='noise') or (self.out_sig.upper().endswith('.WAV')):
            if self.x.shape[1]>1:
                print("tfe : This basic helper function works only if out_sig has only one channel")
                return None, None
            Hout = np.zeros((1+int(nperseg/2),self.y.shape[1]),dtype=complex)
            for ii in range(self.y.shape[1]):
                freqs, Hout[:,ii] =  ms.tfe_welch(self.x[:,0],
                                    self.y[:,ii],
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    fs=self.fs)
        elif self.out_sig=='logsweep':
            #H = np.zeros_like(self.y,dtype=complex)
            freqs, Hout = ms.tfe_farina(self.y[:,0],
                                self.fs,
                                self.out_sig_freqs)
            Hout = Hout[:,None]
            for ii in range(self.y.shape[1]-1):
                freqs, H =  ms.tfe_farina(self.y[:,ii+1],
                                    self.fs,
                                    self.out_sig_freqs)
                Hout = np.block([Hout,H[:,None]])
        else:
            print("tfe : This basic helper function works only if ouSig='noise' or 'logsweep'")
            return None, None
        if plotH:
            ms.plot_tfe(freqs,Hout)
        return freqs, Hout
    
    def tfe_xy(self,x,y,plotH=False,**kwargs):
        """ Compute transfert function between x and y, where x and y are
            strings representing keys of the dictionnary of the data property
            of the Measurement object. Welch's method is used. Data is calibrated.
        """
        out = self.data[y].tfe(self.data[x],**kwargs)

        # freqs, Hout =  ms.tfe_welch(self.data[x].values_in_unit,
        #                             self.data[y].values_in_unit,
        #                             nperseg=nperseg,
        #                             noverlap=noverlap,
        #                             fs=self.fs)

        if plotH:
            out.plot()
        return out

    @property
    def x(self):
        return np.array([self.data[n].values for n in self.out_desc]).T
    
    @property
    def y(self):
        return np.array([self.data[n].values for n in self.in_desc]).T

    @property
    def x_raw(self):
        return np.array([self.data[n].raw for n in self.out_desc]).T
    
    @property
    def y_raw(self):
        return np.array([self.data[n].raw for n in self.in_desc]).T

    @property
    def t(self):
        return ms._create_time(self.fs,dur=self.dur+self.extrat[0]+self.extrat[1])
