import nidaqmx
from measpy.measurement import (Signal,
                    Measurement,
                    ms)

import numpy as np
from numpy.matlib import repmat

from datetime import datetime
import nidaqmx.constants as niconst

def n_to_ain(n):
    return 'ai'+str(n-1)
def n_to_aon(n):
    return 'ao'+str(n-1)

def run_ni_measurement(M):
    system = nidaqmx.system.System.local()
    nsamps = int(round(M.dur*M.fs))

    if M.device_type!='ni':
        print("Warning: deviceType != 'ni'. Changing to 'ni'.")
        M.device_type='ni'
    if M.in_device=='':
        print("Warning: no output device specified, changing to "+system.devices[0].name)
        M.in_device=system.devices[0].name
    if M.out_device=='':
        print("Warning: no output device specified, changing to "+system.devices[0].name)
        M.out_device=system.devices[0].name
    now = datetime.now()
    M.date = now.strftime("%Y/%m/%d")
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
    outtask = nidaqmx.Task(new_task_name="out") # write task

    # Set up the read tasks
    for n in M.in_map:
        print(n_to_ain(n))
        intask.ai_channels.add_ai_voltage_chan(
            physical_channel=M.in_device + "/" + n_to_ain(n),
            terminal_config=niconst.TerminalConfiguration.DEFAULT,
            min_val=-10, max_val=10,
            units=niconst.VoltageUnits.VOLTS)

    intask.timing.cfg_samp_clk_timing(
        rate=M.fs,
        source="OnboardClock",
        active_edge=niconst.Edge.FALLING,
        sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)

    # Set up the write tasks, use the sample clock of the Analog input if possible
    for n in M.out_map:   
        outtask.ao_channels.add_ao_voltage_chan(
            physical_channel=M.out_device + "/" + n_to_aon(n), 
            min_val=-10, max_val=10,
            units=niconst.VoltageUnits.VOLTS)

    try:
        outtask.timing.cfg_samp_clk_timing(
            rate=M.fs,
            source="/" + M.in_device + "/ai/SampleClock", #"OnboardClock",
            active_edge=niconst.Edge.FALLING,
            sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)
    except:
        print("Error when choosing \""+"/" + M.in_device + "/ai/SampleClock\" as clock source, let's try \"OnboardClock\" ")
        outtask.timing.cfg_samp_clk_timing(
            rate=M.fs,
            source="OnboardClock",
            active_edge=niconst.Edge.FALLING,
            sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)

    if len(M.out_map)==1:
        outtask.write(outx[:,0], auto_start=False)
    else:
        outtask.write(outx.T, auto_start=False)

    outtask.start() # Start the write task first, waiting for the analog input sample clock
    y = intask.read(nsamps) # Start the read task

    intask.close()
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

Measurement.run_measurement=run_ni_measurement
