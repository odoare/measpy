#%%

#from measpy import measpyaudio as ma
import measpy.audio as ma
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#plt.style.use('dark_background')

# plt.style.use('classic')
# matplotlib.use('TkAgg')
%matplotlib auto


#%% Define and run a measurement
M1 = ma.Measurement(out_sig='noise',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['In1','In2'],
                    in_cal=[1.0,1.0],
                    in_unit=['Pa','meter/second**2'],
                    in_dbfs=[1.0,1.0],
                    extrat=[0,0],
                    out_sig_fades=[1000,10000],
                    dur=5)
M1.run_measurement()
M1.plot_with_cal()

#%% Save in three different formats
M1.to_jsonwav('j1')
M1.to_csvwav('c1')
M1.to_pickle('1.pck')

#%% Load from file 
M3=ma.load_measurement_from_pickle('1.pck')
plt.plot(M3.t,M3.x)

# %%

M1 = ma.Measurement(out_sig='logsweep',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['V'],
                    in_dbfs=[1.0],
                    extrat=[0.0,0.0],
                    out_sig_fades=[10,10],
                    dur=5)
M1.run_measurement()
M1.plot_with_cal()
# %%






# -*- coding: utf-8 -*-
"""
Gestion des cartes NI en entrée/sortie
"""
#%% Definitions

import nidaqmx
import nidaqmx.constants as niconst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def picv(long):
    return np.hstack((np.zeros(long),1,np.zeros(long-1)))

#%matplotlib auto

plt.style.use('classic')
matplotlib.use('TkAgg')

#%% Measurement properties

SampleRate = 96000
T = 1

dt = 1.0/SampleRate
t = np.linspace(0, T, T*SampleRate)
x = np.sin(1000*t)

x2 = np.hstack( ( picv(int(SampleRate/2)) , x , np.zeros(int(SampleRate)) ) )
nsamps = len(x2)
t2 = np.linspace(0, T, nsamps)

#%% Set up and do acquisition

system = nidaqmx.system.System.local()
devicename =system.devices[0].name
devicename = "Dev2"

intask = nidaqmx.Task(new_task_name="in") # read task
outtask = nidaqmx.Task(new_task_name="out") # write task

# Set up the read task
intask.ai_channels.add_ai_voltage_chan(
    physical_channel=devicename + "/" + 'ai0',
    terminal_config=niconst.TerminalConfiguration.DEFAULT,
    min_val=-10, max_val=10,
    units=niconst.VoltageUnits.VOLTS)

intask.timing.cfg_samp_clk_timing(
    rate=SampleRate,
    source="OnboardClock",
    active_edge=niconst.Edge.FALLING,
    sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)

# Set up the write task, use the sample clock of the Analog input
outtask.ao_channels.add_ao_voltage_chan(
    physical_channel=devicename + "/" + 'ao0', 
    min_val=-10, max_val=10,
    units=niconst.VoltageUnits.VOLTS)

try:
    outtask.timing.cfg_samp_clk_timing(
        rate=SampleRate,
        source="/" + devicename + "/ai/SampleClock", #"OnboardClock",
        active_edge=niconst.Edge.FALLING,
        sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)
except:
    print("Error when choosing \""+"/" + devicename + "/ai/SampleClock\" as clock source, let's try \"OnboardClock\" ")
    outtask.timing.cfg_samp_clk_timing(
        rate=SampleRate,
        source="OnboardClock",
        active_edge=niconst.Edge.FALLING,
        sample_mode=niconst.AcquisitionType.FINITE, samps_per_chan=nsamps)

outtask.write(x2, auto_start=False)

outtask.start() # Start the write task first, waiting for the analog input sample clock
y2 = intask.read(nsamps) # Start the read task

intask.close()
outtask.close()

#%% Analyse and plot data

posmax = int( np.argmax(y2[int(0.25*SampleRate):int(0.75*SampleRate)]) + 0.75*SampleRate )
print(posmax)

y = y2[posmax:posmax+T*SampleRate]

plt.plot(t,x,'o',t,y,'*')
plt.grid()
plt.show()

# %%
