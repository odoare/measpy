# measpy/measurement.py
#
# ------------------------------------------------------
# Measuremnt class definition for data acquisition tasks
# ------------------------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

from .signal import Signal

from ._tools import (csv_to_dict, 
                     convl, 
                     convl1,  
                     calc_dur_siglist)

import numpy as np
import matplotlib.pyplot as plt

from copy import copy

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from copy import copy,deepcopy
import csv
import os

class Measurement:
    # ---------------------------
    def __init__(self, **params):
        # Check out_sig contents
        if 'out_sig' in params:
            non=params['out_sig']!=None
            sig=type(params['out_sig'])!=list
            if non&sig:
                raise Exception("out_sig must but be a list of measpy.signal.Signal or None")
            else:
                #print(list((type(s)==Signal for s in params['out_sig'])))
                if all((type(s)==Signal for s in params['out_sig'])):
                    # print('These are all signals')
                    if all(s.fs==params['out_sig'][0].fs for s in params['out_sig']):
                        # print('Same fs for all signals')
                        self.out_sig = params['out_sig']
                    else:
                        raise Exception("Signals in out_sig list have different sampling frequencies")                     
                else:
                    raise Exception("Some elements of out_sig list are not Signals")
        else:
            self.out_sig = None

        # Check in_sig contents
        if 'in_sig' in params:
            non=params['in_sig']!=None
            sig=type(params['in_sig'])!=list
            if non&sig:
                raise Exception("in_sig must but be a list of measpy.signal.Signal or None")
            else:
                if all((type(s)==Signal for s in params['in_sig'])):
                    # print('These are all signals')
                    if all(s.fs==params['in_sig'][0].fs for s in params['in_sig']):
                        # print('Same fs for all signals')
                        self.in_sig = params['in_sig']
                    else:
                        raise Exception("Signals in in_sig list have different sampling frequencies")                     
                else:
                    raise Exception("Some elements of in_sig list are not Signals")
        else:
            self.in_sig = None

        #Check sampling frequencies
        if type(self.out_sig)==type(None):
            if type(self.in_sig)==type(None):
                #raise Exception("This is a task with no input nor output ?")
                print("This is a task with no input nor output ?")
            else:
                # print("This is a task with no output.")
                if 'fs' in params:
                    if params['fs']!=self.in_sig[0].fs:
                        print('Selected sampling frequency '+str(params['fs'])+'Hz is different to that given in signals in in_sig list')
                        print('Sampling frequencies of all input signals are set to the selected value: '+str(params['fs'])+'Hz.')
                        self.fs = params['fs']
                        for s in self.in_sig:
                            s.fs = params['fs']
                    else:
                        self.fs = params['fs']
                else:
                    self.fs = self.in_sig[0].fs
                    print('Task frequency is: ', str(self.fs) )
        else:
            if 'fs' in params:
                if params['fs']!=self.out_sig[0].fs:
                    print('Selected sampling frequency '+str(params['fs'])+'Hz is different to that given in signals in out_sig list')
                    print("Task's sampling frequency is set to the selected value in outpu signals: "+str(self.out_sig[0].fs)+"Hz.")
            self.fs = self.out_sig[0].fs
            if type(self.in_sig)==type(None):
                print ("This is a task with no input.")
            else:
                if self.fs!=self.in_sig[0].fs:
                    print('Selected sampling frequency '+str(self.fs)+'Hz is different to that given in signals on in_sig list')
                    print('Sampling frequencies of all input signals are set to the selected value: '+str(self.fs)+'Hz.')
                    for s in self.in_sig:
                        s.fs = self.fs

        # Check list lengths
        if type(self.out_sig)!=type(None):
            if 'out_map' in params:
                if len(params['out_map'])!=len(self.out_sig):
                    raise Exception('Lengths of out_map and out_sig do not correspond.')
                self.out_map = params['out_map']
            else:
                self.out_map = list(range(1,len(self.out_sig)+1))
                print("out_map not given, it is set to default value of: "+str(self.out_map))
        if type(self.in_sig)!=type(None):
            if 'in_map' in params:
                if len(params['in_map'])!=len(self.in_sig):
                    raise Exception('Lengths of in_map and in_sig do not correspond.')
                self.in_map = params['in_map']
            else:
                self.in_map = list(range(1,len(self.in_sig)+1))
                print("in_map not given, it is set to default value of: "+str(self.in_map))
                print(self.in_map)

        self.in_device = params.setdefault("in_device",'')
        self.out_device = params.setdefault("out_device",'')

        # Check durations
        if 'dur' in params:
            if type(self.out_sig)==type(None):
                self.dur = params['dur']
            else:
                dursigs = calc_dur_siglist(self.out_sig)
                if params['dur']!=dursigs:
                    print('Selected duration is different thant duration of combined output signals.')
                    print('It is changed to match.')
                self.dur = dursigs
        else:
            if type(self.out_sig)==type(None):
                #raise Exception('No duration nor out_sig given. Impossible to determine task duration')
                print('No duration nor out_sig given. Impossible to determine task duration')
            else:
                self.dur = calc_dur_siglist(self.out_sig)
                print("Duration of the task set to: "+str(self.dur)+" s.")

        if 'device_type' not in params:
            self.device_type = ''
            print('No device_type given, it is set to empty string and will be updated when performing the task.')
        else:
            self.device_type = params['device_type']

        # Fix specific properties
        if self.device_type=='pico':
            self.in_range = params.setdefault("in_range",list('10V' for b in self.in_map))
            self.upsampling_factor = params.setdefault("upsampling_factor",1)
            self.in_coupling = params.setdefault("in_coupling",list('dc' for b in self.in_map))
            self.sig_gen = params.setdefault("sig_gen",False)
            if self.sig_gen != None:
                self.offset = params.setdefault("offset",0.0)
                self.wave = params.setdefault("wave",0)
                self.amp = params.setdefault("amp",1.0)
                self.freq_start = params.setdefault("freq_start",20)
                self.freq_stop = params.setdefault("freq_start",20_000)
                self.freq_change = params.setdefault("freq_change",10)
                self.freq_int = params.setdefault("freq_int",0.01)
                self.sweep_dir = params.setdefault("sweep_dir",0)
                self.sweep_number = params.setdefault("sweep_number",100)
        if self.device_type=='ni':
            self.in_range = params.setdefault("in_range",None)
            self.out_range = params.setdefault("out_range",None)
            self.in_iepe = params.setdefault("in_iepe",list(False for b in self.in_map))
        if type(self.out_sig)!=type(None):
            self.io_sync = params.setdefault('io_sync',0)
        elif 'io_sync' in params:
                print('No output signals given. io_sync param ignored')
        self.desc = params.setdefault('desc','No description')

    # -----------------
    def __repr__(self):
        out = "measpy.Measurement("
        out += "fs="+str(self.fs)
        out += ",\n dur="+str(self.dur)
        out += ",\n device_type='"+str(self.device_type)+"'"
        if self.in_sig!=None:
            out += ',\n in_device='+str(self.in_device)
            out += ',\n in_map='+str(self.in_map)
        try:
            out += ",\n date='"+self.date+"'"
            out += ",\n time='"+self.time+"'"
        except:
            pass
        if self.out_sig!=None:
            out += ",\n out_device='"+str(self.out_device)+"'"
            out += ',\n out_map='+str(self.out_map)
            out += ',\n out_sig=list of '+str(len(self.out_sig))+' measpy.signal.Signal'
            out += ",\n io_sync="+str(self.io_sync)
        if self.device_type=='pico':
            out += ",\n in_range="+str(self.in_range)
            out += ",\n upsampling_factor="+str(self.upsampling_factor)
            out += ",\n in_coupling="+str(self.in_coupling)
        if self.device_type=='ni':
            out += ",\n in_range="+str(self.in_range)
            out += ",\n out_range="+str(self.out_range)
            out += ",\n in_iepe="+str(self.in_iepe)            
        out += ',\n in_sig=list of '+str(len(self.in_sig))+' measpy.signal.Signal'
        out +=")"
        
        return out
    
    # ------------------------------
    def _to_dict(self,withsig=True):
        """ Converts a Measurement object to a dict

            :param withdata: Optionnally removes the data arrays, defaults to True
            :type withdata: bool
            
        """
        mesu = copy(self.__dict__)
        if not(withsig):
            del mesu['in_sig']
            del mesu['out_sig']
        return mesu

    # -----------------------
    def to_dir(self,dirname):
        """ Stores the parameters and signals in a directory

            :param dirname: Name of the directory, a (1), (2)... is added to the name if directory exists
            :type dirname: str
            :return dirname: Actual name to the saved folder (if name conflit is detected)
            :rtype dirname: str                            
        """
        if os.path.exists(dirname):
            i = 1
            while os.path.exists(dirname+'('+str(i)+')'):
                i+=1
            dirname = dirname+'('+str(i)+')'
        os.mkdir(dirname)
        self._params_to_csv(dirname+"/params.csv")
        if type(self.in_sig)!=type(None):
            for i,s in enumerate(self.in_sig):
                s.to_csvwav(dirname+"/in_sig_"+str(i))
        if type(self.out_sig)!=type(None):
            for i,s in enumerate(self.out_sig):
                s.to_csvwav(dirname+"/out_sig_"+str(i))
        self._write_readme(dirname+"/README")
        return dirname
    
    # ------------------------
    @classmethod
    def from_dir(cls,dirname):
        """ Load a measurement object from a directory

            :param dirname: Name of the directory
            :type dirname: str                
        """
        self=cls()
        task_dict = csv_to_dict(dirname+'/params.csv')
        self.fs=convl1(float,task_dict['fs'])
        self.dur=convl1(float,task_dict['dur'])
        try:
            self.date=convl1(str,task_dict['date'])
            self.time=convl1(str,task_dict['time'])
        except:
            pass
        self.device_type=convl1(str,task_dict['device_type'])

        if 'in_map' in task_dict:
            self.in_map = convl(int,task_dict['in_map'])
            self.in_device = convl1(str,task_dict['in_device'])
            self.in_sig = list(Signal.from_csvwav(dirname+'/in_sig_'+str(i)) for i in range(len(task_dict['in_map'])) )
        else:
            self.in_sig = None
        if 'out_map' in task_dict:
            self.out_map = convl(int,task_dict['out_map'])
            self.out_device = convl1(str,task_dict['out_device'])
            self.out_sig = list(Signal.from_csvwav(dirname+'/out_sig_'+str(i)) for i in range(len(task_dict['out_map'])) )
            self.io_sync = convl1(int,task_dict['io_sync'])
        else:
            self.out_sig = None

        # Picoscope specifics
        if self.device_type == 'pico':
            try:
                self.in_range = convl(str,task_dict['in_range'])
            except:
                pass
            try:
                self.in_coupling = convl(str,task_dict['in_coupling'])
            except:
                pass
            try:
                self.upsampling_factor = convl1(int,task_dict['upsampling_factor'])
            except:
                pass
            try:
                self.sig_gen = convl1(bool,task_dict['sig_gen'])
            except:
                pass
            try:
                self.offset = convl1(float,task_dict['offset'])
            except:
                pass
            try:
                self.wave = convl1(int,task_dict['wave'])
            except:
                pass
            try:
                self.freq_start = convl1(float,task_dict['freq_start'])
            except:
                pass
            try:
                self.freq_stop = convl1(float,task_dict['freq_stop'])
            except:
                pass
            try:
                self.freq_change = convl1(float,task_dict['freq_change'])
            except:
                pass
            try:
                self.freq_int = convl1(float,task_dict['freq_int'])
            except:
                pass
            try:
                self.sweep_dir = convl1(int,task_dict['sweep_dir'])
            except:
                pass
            try:
                self.sweep_number = convl1(float,task_dict['sweep_number'])
            except:
                pass

        # NI specifics
        if self.device_type == 'ni':
            try:
                self.in_range = convl1(float,task_dict['in_range'])
            except:
                pass
            try:
                self.out_range = convl1(float,task_dict['out_range'])
            except:
                pass
            try:
                self.in_iepe = convl(bool,task_dict['in_iepe'])
            except:
                pass

        return self

    # --------------------------------
    def _params_to_csv(self,filename):
        """ Writes all the Measurement object parameters to a csv file """
        dd = self._to_dict(withsig=False)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for key in dd:
                if type(dd[key])==list:
                    writer.writerow([key]+dd[key])
                else:
                    writer.writerow([key,str(dd[key])])

    # -------------------------------
    def _write_readme(self,filename):
        with open(filename, 'w') as f:
            f.write('Created with measpy version '+__version__)
            f.write('\n')
            f.write('https://github.com/odoare/measpy')

    def sync_prepare(self,out_chan=0,added_time=1):
        """
        Prepare measurement for synchronization

        :param out_chan: The selected output channel for synchronization. It is the index of the selected output signal in the list ``M.out_sig``
        :type out_chan: int
        :param in_chan: The selected input channel for synchronization. It is the index of the selected input signal in the list ``M.in_sig``
        :type in_chan: int
        :param added_time: Duration of silence added before and after the selected output signal
        :type added_time: float
        """        
        osig = self.out_sig[out_chan].add_silence((added_time,added_time)).delay(-added_time)
        self.dur = self.dur+2*added_time
        self.out_sig[out_chan]=osig

    def sync_render(self,out_chan=0,in_chan=0,added_time=1):
        d = self.in_sig[in_chan].timelag(self.out_sig[out_chan])
        dt = 1/self.fs
        # print("delay: "+str(d)+"s")
        self.dur = self.dur-2*added_time
        self.out_sig[out_chan] = self.out_sig[out_chan].cut(dur=(added_time,added_time+self.dur)).delay(added_time)
        for i,s in enumerate(self.in_sig):
            self.in_sig[i] = s.cut(dur=(1+d+dt,1+d+dt+self.dur))
            self.in_sig[i].t0 = self.out_sig[out_chan].t0
        return d

# def peak_sync_prepare(M,out_chan=0,in_chan=0):
#     def inner(*args,**kwargs):
#         osig = M.out_sig[out_chan]
#         osigpeak = osig.similar(
#             values=np.stack((picv(long=2*M.fs),osig.values)),
#             t0=osig.t0-2)
#         M1 = copy(M)
#         M1.out_sig[out_chan]=osigpeak
#         func(M1)
#         isig = M1.in_sig[in_chan]
#         posmax = int( np.argmax(isig.values[int(0.25*M.fs*2):int(0.75*M.fs*2)]) + 0.75*M.fs*2 )
#         print(posmax)
#         for i,s in enumerate(M1.in_sig):
#             s.values = s.values[posmax:posmax+M.fs*M.dur]
#         M=M1
#         del(M1)
#     return inner

    # def peak_sync_prepare(self,out_chan=0):
    #     osig = self.out_sig[out_chan]
    #     osigpeak = osig.similar(
    #         values=np.stack((picv(long=2*M.fs),osig.values)),
    #         t0=osig.t0-2)
    #     del(osig)
    #     M1 = deepcopy(self)
    #     M1.out_sig[out_chan]=osigpeak
    #     return M1

    # def peak_sync_render(self,in_chan=0):
    #     isig = self.in_sig[in_chan]
    #     posmax = int( np.argmax(isig.values[int(0.25*M.fs*2):int(0.75*M.fs*2)]) + 0.75*M.fs*2 )
    #     print(posmax)
    #     for i,s in enumerate(M1.in_sig):
    #         s.values = s.values[posmax:posmax+M.fs*M.dur]
    #     return M1
    #     del(M1)

