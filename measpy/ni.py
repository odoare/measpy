import nidaqmx
from measpy.measurement import (Signal,
                    Measurement,
                    ms)

import numpy as np
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
    if M.device=='':
        print("Warning: no device specified, changing to "+system.devices[0].name)
        M.device=system.devices[0].name
    now = datetime.now()
    M.date = now.strftime("%Y/%m/%d")
    M.time = now.strftime("%H:%M:%S")

    intask = nidaqmx.Task(new_task_name="in") # read task
    outtask = nidaqmx.Task(new_task_name="out") # write task

    # Set up the read tasks
    for n in M.in_map:
        print(n_to_ain(n))
        intask.ai_channels.add_ai_voltage_chan(
            physical_channel=M.device + "/" + n_to_ain(n),
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
            physical_channel=M.device + "/" + n_to_aon(n), 
            min_val=-10, max_val=10,
            units=niconst.VoltageUnits.VOLTS)

    try:
        outtask.timing.cfg_samp_clk_timing(
            rate=M.fs,
            source="/" + M.device + "/ai/SampleClock", #"OnboardClock",
            active_edge=niconst.Edge.FALLING,
            sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)
    except:
        print("Error when choosing \""+"/" + M.device + "/ai/SampleClock\" as clock source, let's try \"OnboardClock\" ")
        outtask.timing.cfg_samp_clk_timing(
            rate=M.fs,
            source="OnboardClock",
            active_edge=niconst.Edge.FALLING,
            sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)

    if len(M.out_map)==1:
        outtask.write(M.x[:,0], auto_start=False)
    else:
        outtask.write(M.x.T, auto_start=False)

    outtask.start() # Start the write task first, waiting for the analog input sample clock
    y = intask.read(nsamps) # Start the read task

    intask.close()
    outtask.close()

    y=np.array(y).T

    if len(M.in_map)==1:
        M.data[M.in_desc[0]].raw = y
    else:
        n=0
        for s in M.in_desc: 
            M.data[s].raw = y[:,n]
            n+=1

Measurement.run_measurement=run_ni_measurement
