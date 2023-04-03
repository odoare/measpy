# measurement.py
# 
# A class for measurement management with data acquisition devices
#
# OD - 2021

import measpy.signal as ms
from measpy.signal import Signal

from measpy._tools import csv_to_dict, convl, convl1

import numpy as np
import matplotlib.pyplot as plt

from copy import copy

import scipy.io.wavfile as wav
import csv
import pickle
import json

from unyt import Unit

class Measurement:
    """ The class Measurement allows to simply define and perform
        a measurement.

        Initialization parameters:

        :param fs: Sampling frequency, defaults to 44100.
        :type fs: int
        :param dur: Duration in seconds, defaults to 2.0.
        :type dur: float
        :param in_map: Map of the inputs used on the device, defaults to [1,2].
        :type in_map: list(int)
        :param in_cal: Calibrations of the incoming signals, in volts/units, defaults to a list of 1.0.
        :type in_cal: list(str)
        :param in_unit: Measurement units, defaults to a list of 'V'.
        :type in_unit: list(str)
        :param in_dbfs: Input voltage for a unitary value in the resulting signal, defaults to 1.0.
        :type in_unit: list(str)
        :param in_name: Short name of the measurement signals (used as keys for the data dictionnary), defaults to 'In'+str(n)
        :type in_name: list(str)
        :param in_desc: Description of the measurement signals, defaults to 'Thi is input '+str(n)
        :type in_unit: list(str)
        :param out_sig: Output signal type ('noise', 'logsweep', 'file.wav' or None) where file.wav is a wave file string.
        :type out_sig: str
        :param extrat: (not really useful yet) Additionnal extra time before and after (additionnal time), in sample numbers.
        :type extrat: tuple(int,int)

        :param out_map: Map of the outputs used on the device, defaults to [1]
        :type out_map: list(int)
        :param out_amp: Output amplitude, defaults to 1.0
        :type out_amp: float
        :param out_dbfs: Output voltage for a unitary value in the sent signal, defaults to 1.0.
        :type out_sig: float
        :param extrat: Additionnal extra time before and after (additionnal time), in sample numbers.
        :type extrat: tuple(int,int)
        :param out_sig_freqs: Minimum and maximum frequencies for the generated output signals (used if out_sig is 'noise' or 'logsweep')
        :type out_sig_freqs: tuple(float,float)
        :param fades: Fades in/out at the begining and end of the output signal, defaults to (0,0)
        :type fades: tuple(int,int)
        :param io_sync: (not implemented yet) Specifies if in/out synchronization is done, and which type, defaults to 0 (no synchronization).
        :type io_sync: int

        :param device_type: Type of device 'audio', 'ni', 'pico' or '', or None, defaults to ''. It can be eventually reactualized when running the measurement.
        :type device_type: str or None
        :param in_device: Input device, defaults to ''
        :type in_device: str
        :param out_device: Output device, defaults to ''
        :type out_device: str

    """
    def __init__(self, **params):
        # params checking
        if 'out_sig' in params:
            noise=params['out_sig']!='noise'
            logsweep=params['out_sig']!='logsweep'
            sine=params['out_sig']!='sine'
            wa=not str(params['out_sig']).upper().endswith('.WAV')
            non=params['out_sig']!=None
            ar=type(params['out_sig'])!=np.ndarray
            if noise&logsweep&sine&wa&non&ar:
                raise Exception("out_sig must but be numpy.ndarray, 'noise', 'sweep', 'sine', '*.wav' or None")

        self.fs = params.setdefault("fs",44100)
        self.dur = params.setdefault("dur",2.0)
        self.device_type = params.setdefault("device_type",'')

        if 'in_name' in params:
            self.in_map = params.setdefault("in_map",list(range(1,len(params['in_name'])+1)))
        else:
            self.in_map = params.setdefault("in_map",[1,2])
        self.in_name = params.setdefault("in_name",list('In'+str(b) for b in self.in_map))

        self.in_map = params.setdefault("in_map",[1,2])
        self.in_cal = params.setdefault("in_cal",list(1.0 for b in self.in_map))
        self.in_dbfs = params.setdefault("in_dbfs",list(1.0 for b in self.in_map))
        self.in_unit = params.setdefault("in_unit",list('V' for b in self.in_map))
        self.in_name = params.setdefault("in_name",list('In'+str(b) for b in self.in_map))
        self.in_desc = params.setdefault("in_desc",list('This is input '+str(b) for b in self.in_map))
        self.out_sig = params.setdefault("out_sig",'noise')                
        self.extrat = params.setdefault("extrat",[0.0,0.0])
        self.in_device = params.setdefault("in_device",'')
        if self.out_sig!=None:
            self.out_map = params.setdefault("out_map",[1])
            self.out_amp =  params.setdefault("out_amp",1.0)
            self.out_dbfs = params.setdefault("out_dbfs",list(1.0 for b in self.out_map))
            self.out_name = params.setdefault("out_name",list('Out'+str(b) for b in self.out_map))
            self.out_desc =  params.setdefault("out_desc",list('This is output '+str(b) for b in self.out_map))
            self.out_sig_freqs =  params.setdefault("out_sig_freqs",[20.0,20000.0])
            self.io_sync = params.setdefault("io_sync",0)
            self.out_sig_fades = params.setdefault("out_sig_fades",[0,0])
            self.out_device = params.setdefault("out_device",'')
        self.data = {}
        if self.out_sig!=None:
            for n in range(len(self.out_name)):
                self.data[self.out_name[n]]=Signal(desc=self.out_desc[n],
                                                    fs=self.fs,
                                                    unit='V',
                                                    cal=1.0,
                                                    dbfs=self.out_dbfs[n])
            self.create_output()
        for n in range(len(self.in_name)):
            self.data[self.in_name[n]]=Signal(desc=self.in_desc[n],
                                                fs=self.fs,
                                                unit=self.in_unit[n],
                                                cal=self.in_cal[n],
                                                dbfs=self.in_dbfs[n])
        self.data_keys = list(self.data.keys())
        if self.device_type=='pico':
            self.in_range = params.setdefault("in_range",list('10V' for b in self.in_map))
            self.upsampling_factor = params.setdefault("upsampling_factor",1)
            self.in_coupling = params.setdefault("in_coupling",list('dc' for b in self.in_map))
        if self.device_type=='ni':
            self.in_range = params.setdefault("in_range",None)
            self.out_range = params.setdefault("out_range",None)

    def create_output(self):
        """ Creates the output signals, if out_sig is 'noise',
            'logsweep' or a string ending with 'wav'.
            If 'out_sig' is None, nothing is created.
        """
        if self.out_sig=='noise': # White noise output signal
            self.data[self.out_name[0]] = self.data[self.out_name[0]].similar(
                volts=ms._noise(self.fs,self.dur,self.out_amp,self.out_sig_freqs)
            ).fade(self.out_sig_fades).add_silence(self.extrat)

            if self.out_map==0:
                self._out_map=[1]

        elif self.out_sig=='logsweep': # Logarithmic sweep output signal
            self.data[self.out_name[0]] = self.data[self.out_name[0]].similar(
                volts=ms._log_sweep(self.fs,self.dur,self.out_amp,self.out_sig_freqs)
            ).fade(self.out_sig_fades).add_silence(self.extrat)

            if self.out_map==0:
                self.out_map=[1]

        elif self.out_sig=='sine':  # Sinusoidal output signal
            self.data[self.out_name[0]] = self.data[self.out_name[0]].similar(
                volts=ms._sine(self.fs,self.dur,self.out_amp,self.out_sig_freqs[0])
            ).fade(self.out_sig_fades).add_silence(self.extrat)
        
            if self.out_map==0:
                self.out_map=[1]

        elif self.out_sig.upper().endswith('.WAV'): # Wave file output signal

            rate, x = wav.read(self.out_sig)

            if len(x.shape)==1:
                nchan = 1
                x=x[:,None]
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
                    self.out_name=self.out_name[0:nchan]
                    self.out_dbfs=self.out_dbfs[0:nchan]
                else:
                    print("  Truncating channels of the output signal...")
                    x=x[:,0:len(self.out_map)]

            if x.dtype == 'int16':
                vmax=32768
            elif x.dtype == 'int32':
                vmax=2147483648
            else:
                vmax=1.0
                
            for ii in range(len(self.out_map)):
                self.data[self.out_name[ii]]=self.data[self.out_name[ii]].similar(
                    volts=np.array(x[:,ii],dtype=float)/vmax
                ).fade(self.out_sig_fades).add_silence(self.extrat)
        elif type(self.outsig)==np.ndarray:
            pass

    def show(self):
        """ Pretty prints the measurement properties """
        print("Measurement with the following properties:")
        print("| Type of device: device_type="+self.device_type)
        print("| Input device: device="+str(self.in_device))
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
        st = "', '".join(self.in_name)
        print('| Input names: in_name='+"['"+st+"']")        
        st = "', '".join(self.in_desc)
        print('| Input descriptions: in_desc='+"['"+st+"']")        
        print('| Extra time before and after: extrat='+str(self.extrat))
        if self.out_sig!=None:
            print("| Output device: device="+str(self.out_device))
            st = str(self.out_map)
            print('| Output map: out_map='+st)
            st = "', '".join(self.out_name)
            print('| Output names: out_name='+"['"+st+"']")              
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

        if self.device_type=='pico':
            print('| Input ranges: in_range='+str(self.in_range))
            print("| Upsampling factor: upsampling_factor='"+str(self.upsampling_factor)+"'")
            print("| Input coupling: in_coupling="+str(self.in_coupling))
        print("[ Contents of the dictionnary data (keys):")
        for key in self.data:
            print("| "+key)

    def __repr__(self):
        out = "measpy.Measurement("
        out += "fs="+str(self.fs)
        out += ", dur="+str(self.dur)
        out += ", device_type='"+str(self.device_type)+"'"
        out += ", in_device='"+str(self.in_device)+"'"
        out += ', in_name='+str(self.in_name)
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
            out += ", out_device='"+str(self.out_device)+"'"
            out += ', out_name='+str(self.out_name)
            out += ', out_desc='+str(self.out_desc)
            out += ', out_map='+str(self.out_map)
            out += ', out_dbfs='+str(self.out_dbfs)
            out += ", out_sig='"+str(self.out_sig)+"'"
            out += ', out_sig_freqs='+str(self.out_sig_freqs)
            out += ', out_sig_fades='+str(self.out_sig_fades)
            out += ", out_amp="+str(self.out_amp)
            out += ", io_sync="+str(self.io_sync)
        if self.device_type=='pico':
            out += ", in_range="+str(self.in_range)
            out += ", upsampling_factor="+str(self.upsampling_factor)
            out += ", in_coupling="+str(self.in_coupling)
        out +=")"
        
        return out

    def _to_dict(self,withdata=True):
        """ Converts a Measurement object to a dict

            :param withdata: Optionnally removes the data arrays, defaults to True
            :type withdata: bool
            
        """
        self.data_keys = list(self.data.keys())
        mesu = copy(self.__dict__)
        if not(withdata):
            del mesu['data']
        return mesu

    def _from_dict(self,meas):
        """ Update Measurement properties from a dict,
            generally loaded from a file.

            :param meas: dictionnary whose contents should be compatible
            with Measurement class
        """
        self.fs=convl1(float,meas['fs'])
        self.dur=convl1(float,meas['dur'])
        self.in_map=convl(int,meas['in_map'])
        self.in_cal=convl(float,meas['in_cal'])
        self.in_dbfs=convl(float,meas['in_dbfs'])
        self.in_unit=convl(str,meas['in_unit'])
        self.in_name=convl(str,meas['in_name'])
        self.in_desc=convl(str,meas['in_desc'])
        self.extrat=convl(float,meas['extrat'])
        self.out_sig=convl1(str,meas['out_sig'])
        if self.out_sig=='None':
            self.out_sig=None
        try:
            self.data=meas['data']
        except:
            pass
        try:
            self.date=convl1(str,meas['date'])
            self.time=convl1(str,meas['time'])
        except:
            pass
        if self.out_sig!=None:
            self.out_map=convl(int,meas['out_map'])
            self.out_amp=convl1(float,meas['out_amp'])
            self.out_name=convl(str,meas['out_name'])
            self.out_desc=convl(str,meas['out_desc'])
            self.out_sig_freqs=convl(float,meas['out_sig_freqs'])
            self.out_sig_fades=convl(float,meas['out_sig_fades'])
            self.io_sync=convl1(int,meas['io_sync'])
            self.out_dbfs=convl(float,meas['out_dbfs'])
            self.out_device=convl1(str,meas['out_device'])            
            self.out_device=convl1(str,meas['out_device'])
        self.in_device=convl1(str,meas['in_device'])
        self.device_type=convl1(str,meas['device_type'])
        self.data_keys=convl(str,meas['data_keys'])
        if self.device_type=='pico':
            self.in_range=convl(str,meas['in_range'])
            self.upsampling_factor=convl1(int,meas['upsampling_factor'])            
            self.in_range=convl(str,meas['in_coupling'])
        
        # print('In _from_dict')
        # print(self.data)

    def to_pickle(self,filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self._to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls,filename):
        """ Load a Measurement class object from a pickle file

            :param filename: Filename
            :type filename: str

            :return: The loaded measurement object
            :rtype: Measurement

        """
        with open(filename, 'rb') as handle:
            mesu = pickle.load(handle)
        M = cls()
        M._from_dict(mesu)
        for key in M.datakeys:
            M.data[key].unit=Unit(str(M.data[key].unit))
        return M

    def _data_to_array_old(self,includetime=False,datatype='raw'):
        n = 0
        if includetime:
            out = self.data[list(self.data.keys())[0]].time[:,None]
            n += 1
        for key in self.data.keys():
            if n==0:
                if datatype=='raw':
                    out = self.data[key].raw[:,None]
                elif datatype=='volts':
                    out = self.data[key].volts[:,None]
                elif datatype=='values':
                    out = self.data[key].values[:,None]
                n += 1
            else:
                if datatype=='raw':
                    out = np.block([out,self.data[key].raw[:,None]])
                elif datatype=='volts':
                    out = np.block([out,self.data[key].volts[:,None]])
                elif datatype=='values':
                    out = np.block([out,self.data[key].values[:,None]])
                n += 1
        return out

    def _data_to_array(self,includetime=False,datatype='raw'):
        if self.out_sig!=None:
            if datatype=='raw':
                outdata = np.concatenate((self.x_raw,self.y_raw),1)
            elif datatype=='volts':
                outdata = np.concatenate((self.x_volts,self.y_volts),1)
            elif datatype=='values':
                outdata = np.concatenate((self.x,self.y),1)
        else:
            if datatype=='raw':
                outdata = self.y_raw
            elif datatype=='volts':
                outdata = self.y_volts
            elif datatype=='values':
                outdata = self.y
        if includetime:
            outdata = np.concatenate((self.t[:,None],outdata),1)
        return outdata

    def _data_to_wav(self,filename):
        """ Save all data in the measurement as a unique wav file
            It is not the recommended usage to use this method, but
            use to_csvwav or to_jsonwav to save the data, and its
            descriptions.

            :param filename: WAV file name
            :type filename: str
        """
        wav.write(filename,int(round(self.fs)),self._data_to_array())

    def _data_to_txt(self,filename,includetime=True,datatype='raw'):
        """ Save all data in the measurement as a unique txt file
            It is not the recommended usage to use this method, but
            use to_csvtxt or to_jsontxt to save the data, and its
            descriptions.

            :param filename: TXT file name
            :type filename: str
            :param datatype: Type of data ('raw', 'volts' or 'values')
            :type datatype: str 
        """
        np.savetxt(filename,self._data_to_array(includetime=includetime,datatype=datatype))


    def _data_from_wav(self,filename):
        _, dat = wav.read(filename)
        n = 0
        if len(self.data_keys)==1:
            for key in self.data_keys:
                self.data[key].raw = dat
        else:
            for key in self.data_keys:
                self.data[key].raw = dat[:,n]
                n += 1        #print(self.data)
        for key in self.in_name:
            pos = self.in_name.index(key)
            self.data[key].unit = Unit(self.in_unit[pos])
            self.data[key].cal = self.in_cal[pos]
            self.data[key].dbfs = self.in_dbfs[pos]
            self.data[key].desc = self.in_desc[pos]
            self.data[key].fs = self.fs
        if self.out_sig!=None:
            for key in self.out_name:
                pos = self.out_name.index(key)
                self.data[key].unit = Unit('V')
                self.data[key].dbfs = self.out_dbfs[pos]
                self.data[key].desc = self.out_desc[pos]
                self.data[key].fs = self.fs

    def _params_to_csv(self,filename):
        """ Writes all the Measurement object parameters to a csv file """
        dd = self._to_dict(withdata=False)
        #data_keys = list(self.data.keys())
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for key in dd:
                if type(dd[key])==list:
                    writer.writerow([key]+dd[key])
                else:
                    writer.writerow([key,str(dd[key])])
            #writer.writerow(['data_keys']+data_keys)
                    
    def _csv_to_params(self,filename):
        """ Load measurement parameters from a csv file """
        self._from_dict(csv_to_dict(filename))

    def _params_to_json(self,filename):
        """ Writes all the Measurement object parameters to a json file """
        with open(filename, mode='w', encoding='utf-8') as f:
            json.dump(self._to_dict(withdata=False), f, indent=2)

    def _json_to_params(self,filename):
        """ Load measurement parameters from a json file """
        with open(filename, encoding='utf-8') as f:
            self._from_dict(json.load(f))

    def to_csvwav(self,filebase):
        """ Saves a Measurement object to a set of files

            * filebase : string from which two file names are created
            * filebase+'.csv' : All measurement parameters
            * filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        self._params_to_csv(filebase+'.csv')
        try:
            self._data_to_wav(filebase+'.wav')
        except:
            print('data_to_wav failed (no data?)')

    @classmethod
    def from_csvwav(cls,filebase):
        """ Load a measurement object from a set of files

            * filebase : string from which two file names are created
            * filebase+'.csv' : All measurement parameters
            * filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        M=cls()
        M._csv_to_params(filebase+'.csv')
        if M.out_sig!=None:
            M=cls(out_name=M.out_name,in_name=M.in_name)
        else:
            M=cls(in_name=M.in_name)
        M._csv_to_params(filebase+'.csv')
        #try:
        M._data_from_wav(filebase+'.wav')
        #except:
        #    print(filebase+'.wav')
        #    print('data_from_wav failed (file not present?)')
        return M

    def to_jsonwav(self,filebase):
        """ Saves a Measurement object to a set of files

            * filebase : string from which two file names are created
            * filebase+'.json' : All measurement parameters
            * filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        self._params_to_json(filebase+'.json')
        try:
            self._data_to_wav(filebase+'.wav')
        except:
            print('data_to_wav failed (no data?)')

    @classmethod
    def from_jsonwav(cls,filebase):
        """ Load a measurement object from a set of files
        
            * filebase : string from which two file names are created
            * filebase+'.json' : All measurement parameters
            * filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
        """
        M=cls()
        M._json_to_params(filebase+'.json')
        if M.out_sig!=None:
            M=cls(out_name=M.out_name,in_name=M.in_name)
        else:
            M=cls(in_name=M.in_name)
        M._json_to_params(filebase+'.json')
        # try:
        M._data_from_wav(filebase+'.wav')
        # except:
        # print('data_from_wav failed (file not present?)')
        return M

    def to_csvtxt(self,filebase,includetime=True,datatype='values'):
        """ Saves a Measurement object to a set of files

            * filebase : string from which two file names are created
            * filebase+'.csv' : All measurement parameters
            * filebase+'.txt' : all input and out channels + time (as readable textfile)
        """
        self._params_to_csv(filebase+'.csv')
        # try:
        self._data_to_txt(filebase+'.txt',includetime=includetime,datatype=datatype)
        # except:
        #     print('data_to_wav failed (no data?)')

    def plot(self,ytype='values',limit=None):

        for ii in range(self.y.shape[1]):
            if ytype=='values':
                plt.plot(self.t,self.y[:,ii])
            if ytype=='volts':
                plt.plot(self.t,self.y_volts[:,ii])
            if ytype=='raw':
                plt.plot(self.t,self.y_raw[:,ii])
        if limit!=None:
            plt.plot(self.t,limit*np.ones_like(self.t),':',color='grey')
            plt.plot(self.t,-1*limit*np.ones_like(self.t),':',color='grey')
        plt.xlabel('Time(s)')
        legende = []
        for ii in range(self.y.shape[1]):
            if ytype=='values':
                legende+=[self.in_name[ii]+'('+self.in_unit[ii]+')']
            if ytype=='volts':
                legende+=[self.in_name[ii]+'(volts)']
            if ytype=='raw':
                legende+=[self.in_name[ii]+'(-)']            
        legende+=['limits']
        plt.legend(legende)
        plt.title('Measurement date: '+str(self.date)+"   "+str(self.time))
        #plt.grid('on',color='grey',linestyle=':')
        plt.grid('on')

    def tfe(self,nperseg=2**16,noverlap=None,plotH=False):
        """ Helper function that calculates the transfer function between
            the output channel x and all the input channels y. Works only
            if x has only one channel.
            If out_sig='logsweep', the method of Farina is used, Welch's
            method is used otherwise.

            Retruns: a dict of Spectral class objects
        """
        if (self.out_sig=='noise') or (self.out_sig.upper().endswith('.WAV')):
            if self.x.shape[1]>1:
                print("tfe : This basic helper function works only if out_sig has only one channel")
                return None
            out={}
            for key in self.in_name:
                out[key]=self.data[key].tfe_welch(self.data[self.out_name[0]],nperseg=nperseg,noverlap=noverlap)                
        elif self.out_sig=='logsweep':
            if self.x.shape[1]>1:
                print("tfe : This basic helper function works only if out_sig has only one channel")
                return None
            out = {}
            for key in self.in_name:
                out[key]=self.data[key].tfe_farina(self.out_sig_freqs)
        else:
            print("tfe : This basic helper function works only if ouSig='noise' or 'logsweep'")
            return None, None
        if plotH:
            for key in self.in_name:
                out[key].plot()
        return out

    @property
    def x(self):
        """ The output data values as a 2D-array converted in the correponding
            units using the calibration and dbfs properties
        """
        return np.array([self.data[n].values for n in self.out_name]).T
    
    @property
    def y(self):
        """ The input data values as a 2D-array converted in the correponding
            units using the calibration and dbfs properties
        """
        return np.array([self.data[n].values for n in self.in_name]).T

    @property
    def x_volts(self):
        """ The output data values as a 2D-array converted in the correponding
            units using the calibration and dbfs properties
        """
        return np.array([self.data[n].volts for n in self.out_name]).T
    
    @property
    def y_volts(self):
        """ The input data values as a 2D-array converted in the correponding
            units using the calibration and dbfs properties
        """
        return np.array([self.data[n].volts for n in self.in_name]).T

    @property
    def x_raw(self):
        """ Raw output values as a 2D-array (no conversion applied) """
        return np.array([self.data[n].raw for n in self.out_name]).T
    
    @property
    def y_raw(self):
        """ Raw input values as a 2D-array (no conversion applied) """
        return np.array([self.data[n].raw for n in self.in_name]).T

    @property
    def t(self):
        """ Time array """
        return ms.create_time(self.fs,dur=self.dur+self.extrat[0]+self.extrat[1])

    # Old tfe function (deprecated)
    def tfeb(self,nperseg=2**16,noverlap=None,plotH=False):
        """ DEPRECATED
            Helper function that calculates the transfer function between
            the output channel x and all the input channels y. Works only
            if x has only one channel.
            If out_sig='logsweep', the method of Farina is used, Welch's
            method is used otherwise.

            Returns : 1D f array and 2D H array 
        """
        print('Warning: DEPRECATED')
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
        """ DEPRECATED
            Compute transfert function between x and y, where x and y are
            strings representing keys of the dictionnary of the data property
            of the Measurement object. Welch's method is used. Data is calibrated.
        """
        print('Warning: DEPRECATED')
        out = self.data[y].tfe_welch(self.data[x],**kwargs)

        if plotH:
            out.plot()
        return out
        
