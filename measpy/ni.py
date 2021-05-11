import nidaqmx
from measpy.measurement import (Signal,
                    Measurement,
                    ms)

import numpy as np
from datetime import datetime
import nidaqmx.constants as niconst

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

    # Set up the read task
    intask.ai_channels.add_ai_voltage_chan(
        physical_channel=M.device + "/" + 'ai0',
        terminal_config=niconst.TerminalConfiguration.DEFAULT,
        min_val=-10, max_val=10,
        units=niconst.VoltageUnits.VOLTS)

    intask.timing.cfg_samp_clk_timing(
        rate=M.fs,
        source="OnboardClock",
        active_edge=niconst.Edge.FALLING,
        sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)

    # Set up the write task, use the sample clock of the Analog input
    outtask.ao_channels.add_ao_voltage_chan(
        physical_channel=M.device + "/" + 'ao0', 
        min_val=-10, max_val=10,
        units=niconst.VoltageUnits.VOLTS)

    try:
        outtask.timing.cfg_samp_clk_timing(
            rate=M.device,
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

    outtask.write(M.data['Out1'].values, auto_start=False)

    outtask.start() # Start the write task first, waiting for the analog input sample clock
    y = intask.read(nsamps) # Start the read task

    intask.close()
    outtask.close()

    M.data['In1'].raw = np.array(y)

Measurement.run_measurement=run_ni_measurement
