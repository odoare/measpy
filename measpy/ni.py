import nidaqmx
import nidaqmx.constants as niconst

import measpy.signal as ms

import numpy as np
from numpy.matlib import repmat

from datetime import datetime

def n_to_ain(n):
    return 'ai'+str(n-1)
def n_to_aon(n):
    return 'ao'+str(n-1)


def callback(task_handle, every_n_samples_event_type,
                number_of_samples, callback_data):
    print('Every N Samples callback invoked.')

    return 0


def ni_run_measurement(M):
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

    if M.out_sig!=None:
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
            
    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Insert a synchronization peak at the begining of the output signals
    if M.out_sig==None:
        dursync=0
        effsync=False
    elif M.io_sync>0:
        if M.io_sync in M.in_map:
            nout = M.x.shape[1]
            peaks = repmat(ms.picv(M.fs),nout,1).T
            zers = repmat(np.zeros(int(M.fs)),nout,1).T
            outx = np.block([[peaks],[M.x],[zers]])
            effsync = True
            dursync=4
            indsearch=M.in_map.index(M.io_sync)
        else:
            print('io_sync channel not present in in_map, no sync is done')
            outx=M.x
            dursync=0
            effsync=False
    else:
        outx=M.x
        dursync=0
        effsync=False

    nsamps = int(round((dursync+M.dur)*M.fs))

    intask = nidaqmx.Task(new_task_name="in") # read task

    if M.out_sig!=None:
        outtask = nidaqmx.Task(new_task_name="out") # write task

    # Set up the read tasks
    for i,n in enumerate(M.in_map):
        print(n_to_ain(n))
        intask.ai_channels.add_ai_voltage_chan(
            physical_channel=M.in_device + "/" + n_to_ain(n),
            terminal_config=niconst.TerminalConfiguration.DEFAULT,
            min_val=-M.in_range[i], max_val=M.in_range[i],
            units=niconst.VoltageUnits.VOLTS)

    intask.timing.cfg_samp_clk_timing(
        rate=M.fs,
        sample_mode=niconst.AcquisitionType.CONTINUOUS,
        samps_per_chan=nsamps)

    if M.out_sig!=None:
        # Set up the write tasks, use the sample clock of the Analog input if possible
        for i,n in enumerate(M.out_map):   
            outtask.ao_channels.add_ao_voltage_chan(
                physical_channel=M.out_device + "/" + n_to_aon(n), 
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

    y = intask.read(nsamps,timeout=M.dur+10) # Start the read task

    intask.close()

    if M.out_sig!=None:
        outtask.close()

    y=np.array(y).T

    if effsync:
        posmax = int( np.argmax(y[int(0.25*M.fs*2):int(0.75*M.fs*2),indsearch]) + 0.75*M.fs*2 )
        print(posmax)
        y = y[posmax:posmax+M.fs*M.dur,:]

    if len(M.in_map)==1:
        M.data[M.in_name[0]].raw = y
    else:
        n=0
        for s in M.in_name: 
            M.data[s].raw = y[:,n]
            n+=1

def ni_get_devices():
    system = nidaqmx.system.System.local()
    print(system.devices)
    return system.devices
