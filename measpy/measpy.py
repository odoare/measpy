# measpy.py
# 
# A class for measurement management with data acquisition devices
#
# OD - 2021

# TODO :
# - tbefore, tafter
# - synchronisation

import measpy.measpysignal as ms
from measpy.measpysignal import Signal
import numpy as np
import matplotlib.pyplot as plt

from copy import copy

from scipy.io.wavfile import write, read
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

def load_measurement_from_csvwav(basefilename):
    """ Returns a Measurement object from a set of csv+wav files
        See help of method from_csvwav
    """
    M=Measurement()
    M.from_csvwav(basefilename)
    return M

def load_measurement_from_jsonwav(basefilename):
    """ Returns a Measurement object from a set of json+wav files
        See help of method from_jsonwav
    """
    M=Measurement()
    M.from_jsonwav(basefilename)
    return M

def load_measurement_from_pickle(filename):
    """ Returns a Measurement object from a pickle file """
    M=Measurement()
    M.from_pickle(filename)
    return M

class Measurement:
    """ The class Measurement allows to simply define and perform
        a measurement
    """
    def __init__(self, **params):
        """ Optionnal parameters (param=default)
                fs=44100
                dur=2
                in_map=[1,2]
                in_cal=[1,1]
                in_dbfs=[1.0,1.0]
                in_unit=['V','V']
                in_desc=['Input channel 1','Input channel 2']
                out_sig='noise'

            If out_sig != None:
                out_map=[1]
                out_amp=1.0
                out_dbfs=[1.0]
                min_freq=20
                max_freq=20000
                ioSync=0
                fades=[100,100]
            
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
            self.data[self.out_desc[n]]=Signal(self.out_desc[n],self.fs,'V',1.0,self.out_dbfs[n])
        for n in range(len(self.in_desc)):
            self.data[self.in_desc[n]]=Signal(self.in_desc[n],self.fs,self.in_unit[n],self.in_cal[n],self.in_dbfs[n])
        self.datakeys = list(self.data.keys())
        self.create_output()
        
    def create_output(self):
        if self.out_sig=='noise': # White noise output signal
            _, out = ms.create_noise(self.fs,
                                                            self.dur,
                                                            self.out_amp,
                                                            self.out_sig_freqs,
                                                            self.out_sig_fades)
            self.data[self.out_desc[0]].values = np.hstack(
                (np.zeros(int(np.round(self.extrat[0]*self.fs))),
                out,
                np.zeros(int(np.round(self.extrat[1]*self.fs))) ))
            if self.out_map==0:
                self._out_map=[1]

        elif self.out_sig=='logsweep': # Logarithmic sweep output signal
            _, self.data[self.out_desc[0]].values = ms.create_log_sweep(self.fs,
                                                            self.dur,
                                                            self.out_amp,
                                                            self.out_sig_freqs,
                                                            self.out_sig_fades)
            if self.out_map==0:
                self.out_map=[1]

        elif self.out_sig.upper().endswith('.WAV'): # Wave file output signal

            rate, x = read(self.out_sig)

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
                print("  Size of out_map and number of channels do not correspond.")
                if (x.shape[1]<len(self.out_map)):
                    self.out_map=self.out_map[0:nchan]
                    self.out_desc=self.out_desc[0:nchan]
                    self.out_dbfs=self.out_dbfs[0:nchan]
                    print("  Truncating current out_map...")
                else:
                    x=x[:,0:len(self.out_map)]
                    print("  Truncating channels of the output signal...")
            if x.dtype == 'int16':
                for ii in range(len(self.out_map)):
                    self.data[self.out_desc[ii]].values=np.array(x[:,ii],dtype=float)/32768
            elif x.dtype == 'int32':
                for ii in range(len(self.out_map)):
                    self.data[self.out_desc[ii]].values=np.array(x[:,ii],dtype=float)/2147483648
            else:
                for ii in range(len(self.out_map)):
                    self.data[self.out_desc[ii]].values=np.array(x[:,ii],dtype=float)

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

    def from_pickle(self,filename):
        with open(filename, 'rb') as handle:
            mesu = pickle.load(handle)
        self.from_dict(mesu)

    def data_to_wav(self,filename):
        n = 0
        for key in self.data.keys():
            if n==0:
                out = self.data[key].values[:,None]
                n += 1
            else:
                out = np.block([out,self.data[key].values[:,None]])
                n += 1
        write(filename,int(round(self.fs)),out)

    def data_from_wav(self,filename):
        _, dat = read(filename)
        n = 0
        for key in self.data_keys:
            self.data[key].values = dat[:,n]
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

    def from_csvwav(self,filebase):
        """ Load a measurement object from a set of files
                    filebase : string from which two file names are created
                    filebase+'.csv' : All measurement parameters
                    filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        self.csv_to_params(filebase+'.csv')
        try:
            self.data_from_wav(filebase+'.wav')
        except:
            print('data_from_wav failed (file not present?)')

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

    def from_jsonwav(self,filebase):
        """ Load a measurement object from a set of files
                    filebase : string from which two file names are created
                    filebase+'.json' : All measurement parameters
                    filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        self.json_to_params(filebase+'.json')
        try:
            self.data_from_wav(filebase+'.wav')
        except:
            print('data_from_wav failed (file not present?)')

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
                                self.dur,
                                self.fs,
                                self.out_sig_freqs)
            Hout = Hout[:,None]
            for ii in range(self.y.shape[1]-1):
                freqs, H =  ms.tfe_farina(self.y[:,ii+1],
                                    self.dur,
                                    self.fs,
                                    self.out_sig_freqs)
                Hout = np.block([Hout,H[:,None]])
        else:
            print("tfe : This basic helper function works only if ouSig='noise' or 'logsweep'")
            return None, None
        if plotH:
            ms.plot_tfe(freqs,Hout)
        return freqs, Hout
    
    def tfe_xy(self,x,y,nperseg=2**16,noverlap=None,plotH=False):
        """ Compute transfert function between x and y, where x and y are
            strings representing keys of the dictionnary of the data property
            of the Measurement object. Welch's method is used. Data is calibrated.
        """
        freqs, Hout =  ms.tfe_welch(self.data[x].values_in_unit,
                                    self.data[y].values_in_unit,
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    fs=self.fs)

        if plotH:
            ms.plot_tfe(freqs,Hout)
        return freqs, Hout

    @property
    def x(self):
        return np.array([self.data[n].values_in_volts for n in self.out_desc]).T
    
    @property
    def y(self):
        return np.array([self.data[n].values_in_unit for n in self.in_desc]).T

    @property
    def t(self):
        return ms.create_time(self.fs,dur=self.dur+self.extrat[0]+self.extrat[1])
