# measurement.py
# 
# A class for measurement management with data acquisition devices
#
# OD - 2021

from .signal import Signal

from ._tools import (csv_to_dict, 
                     convl, 
                     convl1,  
                     calc_dur_siglist,
                     picv)

from ._version import VERSION

import numpy as np
import matplotlib.pyplot as plt

from copy import copy,deepcopy

import scipy.io.wavfile as wav
import csv
import pickle
import json

from unyt import Unit

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
        self.out_device = params.setdefault("in_device",'')

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
        if type(self.out_sig)!=type(None):
            self.io_sync = params.setdefault('io_sync',0)
        elif 'io_sync' in params:
                print('No output signals given. io_sync param ignored')
        self.desc = params.setdefault('desc','No description')

    # -----------------
    def __repr__(self):
        out = "measpy.Daqtask("
        out += "fs="+str(self.fs)
        out += ",\n dur="+str(self.dur)
        out += ",\n device_type='"+str(self.device_type)+"'"
        out += ',\n in_map='+str(self.in_map)
        try:
            out += ",\n date='"+self.date+"'"
            out += ",\n time='"+self.time+"'"
        except:
            pass
        if self.out_sig!=None:
            out += ",\n out_device='"+str(self.out_device)+"'"
            out += ',\n out_map='+str(self.out_map)
            #out += ", out_sig='"+str(self.out_sig)+"'"
            out += ',\n out_sig=list of '+str(len(self.out_sig))+' measpy.signal.Signal'
            out += ",\n io_sync="+str(self.io_sync)
        if self.device_type=='pico':
            out += ",\n in_range="+str(self.in_range)
            out += ",\n upsampling_factor="+str(self.upsampling_factor)
            out += ",\n in_coupling="+str(self.in_coupling)
        if self.device_type=='ni':
            out += ",\n in_range="+str(self.in_range)
            out += ",\n out_range="+str(self.out_range)
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
        """ Writes the parameters and signals in a directory"""
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
    
    # ------------------------
    @classmethod
    def from_dir(cls,dirname):
        """ Load a measurement object from a set of files

            * filebase : string from which two file names are created
            * filebase+'.csv' : All measurement parameters
            * filebase+'.wav' : all input and out channels + time (32 bit float WAV at fs)
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
            f.write('Created with measpy version '+VERSION)
            f.write('\n')
            f.write('https://github.com/odoare/measpy')
