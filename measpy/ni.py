# measpy/ni.py
#
# -------------------------------------------------
# Data acquisition with National Instrument devices
# -------------------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

import nidaqmx
import nidaqmx.constants as niconst

from ._tools import siglist_to_array, t_min

import numpy as np

from datetime import datetime
from time import sleep

def _n_to_ain(n):
    return 'ai'+str(n-1)
def _n_to_aon(n):
    return 'ao'+str(n-1)

def _callback(task_handle, every_n_samples_event_type,
                number_of_samples, callback_data):
    print('Every N Samples callback invoked.')

    return 0

def ni_run_measurement(M):
    """
    Runs a measurement defined in the object of
    the class measpy.measurement.Measurement given
    as argument.

    Once the data acquisition process is terminated,
    the measurement object given in argument contains
    a property in_sig consisting of a list of signals.

    :param M: The measurement object that defines the measurement properties
    :type M: measpy.measurement.Measurement

    :return: Nothing, the measurement passed as argument is modified in place.

    """
    system = nidaqmx.system.System.local()
    nsamps = int(round(M.dur*M.fs))

    if M.device_type!='ni':
        print("Warning: deviceType != 'ni'. Changing to 'ni'.")
        M.device_type='ni'
    if M.in_device=='':
        print("Warning: no output device specified, changing to "+system.devices[0].name)
        M.in_device=system.devices[0].name

    if hasattr(M, 'in_range'):
        if M.in_range == None:
            inr = False
        else :
            inr = True
    else:
        inr = False
    if not(inr):
        val = nidaqmx.system.device.Device(M.in_device).ai_voltage_rngs[-1]
        print("Warning: no input range specified, changing to the max value of "+M.in_device+" -> "+str(val))
        M.in_range = list(val for b in M.in_map)

    if hasattr(M, 'out_sig') and M.out_sig!=None:
        outx = siglist_to_array(M.out_sig)
        tmin = t_min(M.out_sig)
        if M.out_device=='':
            print("Warning: no output device specified, changing to "+system.devices[0].name)
            M.out_device=system.devices[0].name
        if hasattr(M, 'out_range'):
            if M.out_range == None:
                outr = False
            else :
                outr = True
        else:
            outr = False
        if not(outr):
            val = nidaqmx.system.device.Device(M.out_device).ao_voltage_rngs[-1]
            print("Warning: no output range specified, changing to the max value of "+M.out_device+" -> "+str(val))
            M.out_range = list(val for b in M.out_map)
    else:
        tmin = 0

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Set up the read tasks
    if M.in_sig!=None:
        intask = nidaqmx.Task(new_task_name="in") # read task
        for i,n in enumerate(M.in_map):
            print(_n_to_ain(n))
            intask.ai_channels.add_ai_voltage_chan(
                physical_channel=M.in_device + "/" + _n_to_ain(n),
                terminal_config=niconst.TerminalConfiguration.DEFAULT,
                min_val=-M.in_range[i], max_val=M.in_range[i],
                units=niconst.VoltageUnits.VOLTS)

        intask.timing.cfg_samp_clk_timing(
            rate=M.fs,
            sample_mode=niconst.AcquisitionType.CONTINUOUS,
            samps_per_chan=nsamps)
        
        for i,iepeval in enumerate(M.in_iepe):
           if iepeval:
                intask.ai_channels[i].ai_excit_val = 0.002
                intask.ai_channels[i].ai_coupling = niconst.Coupling.AC

    # Set up the write tasks
    if M.out_sig!=None:
        outtask = nidaqmx.Task(new_task_name="out") # write task

        # Set up the write tasks, use the sample clock of the Analog input if possible
        for i,n in enumerate(M.out_map):   
            outtask.ao_channels.add_ao_voltage_chan(
                physical_channel=M.out_device + "/" + _n_to_aon(n), 
                min_val=-M.out_range[i], max_val=M.out_range[i],
                units=niconst.VoltageUnits.VOLTS)
      
        if M.in_device.startswith('myDAQ'):
            # If the device is a myDAQ card, we keep most default values
            # The myDAQ devices are set up separately because
            # there is no error messages when setting up properties
            # that are not supported, and the acquisition then fails
            print("This is a NI my DAQ device")
            outtask.timing.cfg_samp_clk_timing(
                rate=M.fs,
                sample_mode=niconst.AcquisitionType.CONTINUOUS,
                samps_per_chan=nsamps)
        else:
            try:
                # We first try to use analog input sample clock as output clock
                outtask.timing.cfg_samp_clk_timing(
                    rate=M.fs,
                    source="/" + M.in_device + "/ai/SampleClock", #"OnboardClock",
                    sample_mode=niconst.AcquisitionType.CONTINUOUS,
                    samps_per_chan=nsamps)
                print('Use of /'+ M.in_device + '/ai/SampleClock as output clock : success !')
            except:
                # If it fails, use defaults
                # Then the in/out are not synchronized
                # There is hence the possibility to use one analog input
                # to do the in/out sync (io_sync=input channel number)
                print("Error when choosing \""+"/" + M.in_device + "/ai/SampleClock\" as clock source, let's try \"OnboardClock\" ")
                outtask.timing.cfg_samp_clk_timing(
                    rate=M.fs,
                    sample_mode=niconst.AcquisitionType.CONTINUOUS,
                    samps_per_chan=nsamps)

        if len(M.out_map)==1:
            outtask.write(outx[:,0], auto_start=False)
        else:
            outtask.write(outx.T, auto_start=False)

        outtask.start() # Start the write task first, waiting for the analog input sample clock

    if M.in_sig!=None:
        y = intask.read(nsamps,timeout=M.dur+10) # Start the read task
        intask.close()
    else:
        sleep(M.dur+10)

    if M.out_sig!=None:
        outtask.close()

    if M.in_sig!=None:
        y=np.array(y).T
        if len(M.in_map)==1:
            M.in_sig[0].raw = y
            M.in_sig[0].t0 = tmin
        else:
            for i,s in enumerate(M.in_sig):
                s.raw = y[:,i]
                s.t0 = tmin

def ni_run_synced_measurement(M,in_chan=0,out_chan=0,added_time=1):
    """
    Before running a measurement, added_time second of silence
    is added at the begining and end of the selected output channel.
    The measurement is then run, and the time lag between
    a selected acquired signal and the output signal is computed
    from cross-correlation calculation.
    All the acquired signals are then re-synced from the time lag value.

    :param M: The measurement object
    :type M: measpy.measurement.Measurement
    :param out_chan: The selected output channel for synchronization. It is the index of the selected output signal in the list ``M.out_sig``
    :type out_chan: int
    :param in_chan: The selected input channel for synchronization. It is the index of the selected input signal in the list ``M.in_sig``
    :type in_chan: int
    :param added_time: Duration of silence added before and after the selected output signal
    :type added_time: float
    :return: Measured delay between i/o sync channels
    :rtype: float
    """
    M.sync_prepare(out_chan=out_chan,added_time=added_time)
    ni_run_measurement(M)
    d = M.sync_render(in_chan=in_chan,out_chan=out_chan,added_time=added_time)
    return d

def ni_get_devices():
    """
    Get the list of NI devices present in the system

    :returns: A list of devices object
    """
    system = nidaqmx.system.System.local()
    print(system.devices)
    return system.devices
