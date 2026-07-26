# Tutorial 3: The Measurement class

The ```measpy.Measurement``` class describes a data acquisition task: which
signals are sent to the outputs of a device, which signals are recorded at its
inputs, at which sampling frequency, for how long, and with which device.

A ```Measurement``` object is basically a container that gathers:

- ```measpy.Signal``` objects for the outputs (```out_sig```) and the inputs (```in_sig```),
- the channel numbers on which they are sent and recorded (```out_map``` and ```in_map```),
- the description of the acquisition device (```device_type```, ```in_device```, ```out_device```),
- the parameters of the task itself (```fs```, ```dur```).

The ```Measurement``` object does not perform the acquisition by itself. It is
given as argument to a function that is specific to the hardware in use:

| Device | Module | Function |
| --- | --- | --- |
| Audio card | ```measpy.audio``` | ```audio_run_measurement``` |
| NI DAQ card | ```measpy.ni``` | ```ni_run_measurement``` |
| Picoscope 2000 | ```measpy.pico``` | ```ps2000_run_measurement``` |
| Picoscope 4000 | ```measpy.pico``` | ```ps4000_run_measurement``` |

These functions modify the measurement **in place**: once the task is done, the
input signals that were empty contain the acquired samples.

**Note:** the code cells of this tutorial need a real acquisition device to be
run, they are therefore not executed in this document.


```python
#This is here in case we want to use the local measpy directory
import sys
sys.path.insert(0, "..")

import measpy as mp
import numpy as np
import matplotlib.pyplot as plt
```

## 1. Preparing the signals

Before creating a measurement, we create the signals that describe what is sent
and what is measured.

The **output signals** contain the data that will be sent to the DAC. They are
usually created with one of the waveform creation methods of the ```Signal```
class (```log_sweep```, ```noise```, ```sine```, ...).

The **input signals** are created empty: only the properties that describe how
the acquired voltage has to be interpreted are given, that is:

- ```unit``` : the physical unit of the measured quantity,
- ```cal``` : the calibration of the sensor, in volts per unit,
- ```dbfs``` : the voltage that corresponds to a sample value of one. This is
  needed for audio cards, whose samples are given in the [-1,1] range. For most
  DAQ cards (NI for instance) the samples are given in volts and ```dbfs```
  stays at its default value of 1.0.

In the example below, we prepare a five seconds logarithmic sweep, and two input
signals: a pressure measured with a microphone of sensitivity 2V/Pa, and an
acceleration measured with an accelerometer of sensitivity 0.1V/(m/s²), both
connected to a sound card whose full scale is 1.4V.


```python
FS = 44100
DUR = 5

# The signal that will be sent to the output
s_out = mp.Signal.log_sweep(fs=FS, dur=DUR, freqs_range=(20, 20000), unit='V')

# The empty signals that will receive the acquired data
s_press = mp.Signal(fs=FS, unit='Pa', cal=2.0, dbfs=1.4, desc='Pressure here')
s_acc = mp.Signal(fs=FS, unit='m/s**2', cal=0.1, dbfs=1.4, desc='Acceleration there')
```

## 2. Creating the Measurement object

The signals are then gathered in a ```Measurement``` object. The main parameters
are:

| Parameter | Description |
| --- | --- |
| ```device_type``` | ```'audio'```, ```'ni'``` or ```'pico'``` |
| ```in_device```, ```out_device``` | Identifiers of the devices (strings) |
| ```fs``` | Sampling frequency |
| ```dur``` | Duration of the task in seconds |
| ```in_sig```, ```out_sig``` | A list of ```Signal```, or one multichannel ```Signal```, or None |
| ```in_map```, ```out_map``` | The channel numbers, starting at 1 |
| ```desc``` | A description of the measurement |

Some rules are applied at creation, and messages are printed when a parameter is
adjusted:

- if ```dur``` is not given, it is deduced from the duration of the output signals,
- if ```dur``` differs from the duration of the output signals, it is rounded up
  to an integer number of repetitions of these signals,
- if ```fs``` differs from the sampling frequency of the signals, the sampling
  frequency of the output signals wins,
- if ```in_map``` or ```out_map``` is not given, channels are numbered from 1,
- a measurement can have no output (```out_sig=None```): it is then a pure
  recording task, and ```dur``` has to be given.


```python
M = mp.Measurement(device_type='audio',
                   fs=FS,
                   out_sig=[s_out],
                   out_map=[1],
                   in_sig=[s_press, s_acc],
                   in_map=[1, 2],
                   dur=DUR,
                   in_device='default',
                   out_device='default',
                   desc='Sweep response of the system')
print(M)
```

## 3. Running the measurement

### With an audio card

The list of the audio devices present in the system is obtained with
```audio_get_devices```. The strings that identify the input and output devices
are then passed to the measurement as ```in_device``` and ```out_device```.
On Linux, ```'default'``` usually corresponds to the main input and output.


```python
from measpy.audio import audio_run_measurement, audio_get_devices

print(audio_get_devices())
```


```python
audio_run_measurement(M)
```

### With a NI DAQ card

The devices are listed with ```ni_get_devices```. Contrary to audio cards, there
is a one to one correspondence between the input volts and the sample values,
so that ```dbfs``` is not needed.

Two additional parameters can be given to a NI measurement:

- ```in_range```, ```out_range``` : the voltage ranges of the channels. If not
  given, the maximum range of the device is selected.
- ```in_iepe``` : a list of booleans that selects the IEPE conditioning of each
  input channel (for ICP microphones and accelerometers).
- ```in_sig_config``` : the terminal configuration, one of ```'DEFAULT'```,
  ```'RSE'```, ```'NRSE'```, ```'DIFF'```, ```'PSEUDO_DIFF'```.


```python
from measpy.ni import ni_run_measurement, ni_get_devices

sysdevs = ni_get_devices().device_names
print(sysdevs)
indev = outdev = sysdevs[0]
```


```python
s_in = mp.Signal(fs=FS, unit='V', desc='Input voltage')

Mni = mp.Measurement(device_type='ni',
                     fs=FS,
                     out_sig=[s_out],
                     out_map=[1],
                     in_sig=[s_in],
                     in_map=[1],
                     dur=DUR,
                     in_device=indev,
                     out_device=outdev,
                     in_range=[10.0],
                     out_range=[10.0],
                     in_iepe=[False])

ni_run_measurement(Mni)
```

## 4. Using the acquired signals

Once the acquisition is done, the signals given in ```in_sig``` are filled with
the measured data. They are regular ```measpy.Signal``` objects: all the methods
presented in the first two tutorials apply (see the Signal and Spectral
tutorials for more details).


```python
# Plot the two input signals on the same graph
a = M.in_sig[0].plot()
M.in_sig[1].plot(ax=a)
```


```python
# Power spectral density of the first input
M.in_sig[0].psd(nperseg=2**14).plot()

# Transfer function between the output and the first input
M.in_sig[0].tfe_welch(M.out_sig[0], nperseg=2**14).plot()

# Same transfer function, using the fact that the output is a logarithmic sweep
# (Farina's method), smoothed on 1/12th octave bands
M.in_sig[0].tfe_farina(freqs_range=(20, 20000)) \
           .nth_oct_smooth_complex(12, freqs_range=(20, 20000)) \
           .plot()
```

## 5. Saving and restoring a measurement

Three storage formats are available.

### A directory of csv and wav files

```to_dir``` creates a directory that contains a ```params.csv``` file with the
measurement parameters, a pair of csv/wav files per signal, and a README file
that indicates the measpy version. If the directory exists, a ```(1)```,
```(2)```... suffix is added and the actual name is returned.


```python
dirname = M.to_dir('my_measurement')
M1 = mp.Measurement.from_dir(dirname)
print(M1)
```

### A single HDF5 file

```to_hdf5``` saves the whole measurement (parameters and data) in one file.
This is the recommended format for large measurements.


```python
M.to_hdf5('my_measurement.h5')
M2 = mp.Measurement.from_hdf5('my_measurement.h5')
```

### Writing to disk during the acquisition

For long acquisitions, the data can be written to an HDF5 file while it is
acquired, instead of being kept in memory. With a NI card, this is done by
giving a file name to ```ni_run_measurement```. The dataset is created before
the task starts, and filled chunk by chunk during the acquisition. At the end,
the data is loaded back into the measurement object.


```python
ni_run_measurement(Mni, filename='long_measurement.h5')

# The data can be reloaded later with
# M3 = mp.Measurement.from_hdf5('long_measurement.h5')
```

## 6. Multichannel input signals

Instead of a list of single channel signals, ```in_sig``` can be one
multichannel ```Signal```. The acquired data is then stored as a 2D numpy array,
one column per channel. This is more efficient for a large number of channels.

The properties (```unit```, ```cal```, ```desc```...) can then be given as
lists, one value per channel.


```python
s_multi = mp.Signal(fs=FS,
                    unit=['Pa', 'm/s**2'],
                    cal=[2.0, 0.1],
                    dbfs=1.4,
                    desc=['Pressure here', 'Acceleration there'])

Mmulti = mp.Measurement(device_type='audio',
                        fs=FS,
                        in_sig=s_multi,
                        in_map=[1, 2],
                        dur=DUR,
                        in_device='default')

audio_run_measurement(Mmulti)

# The channels are then accessed by indexing, or with unpack()
Mmulti.in_sig[0].plot()
sig_list = Mmulti.in_sig.unpack()
```

## 7. What comes next

An acquisition made with a card that has no hardware synchronization between its
inputs and its outputs has an unknown delay between the sent and the recorded
signals. The next tutorial shows how measpy measures and compensates this
delay.
