# measpy - Measurements with Python
(c) 2020-2023 Olivier Doaré
Contact: olivier.doare@ensta-paris.fr

measpy is a set of classes and methods to help data acquisition with DAQ cards or sound cards and analysis of signals. It is mainly acoustics and vibrations oriented.

The base classes are:
- ```Signal```: A class that contain a 1D numpy array, and a few other properties to describe the data as: sampling frequency, calibration and unit.
- ```Spectral```: A spectral data class that contains the complex amplitudes as a 1D numpy array for frequencies up to the Nyquist frequency ot the sampling frequency, and some properties as sampling frequency, unit, description
- ```Measurement``` : A class that describe a data acquisition process, its outputs (Signal objects), its inputs (Signal objects)...
- ```Weighting``` : A weighting class that holds complex values for a list of frequencies, and methods to to smoothing, interpolation, etc.

For now, these daq devices are implemented :
- Audio cards, via the ```sounddevice``` package,
- NI DAQ cards, via the ```nidaqmx``` package.
- Picoscope scopes, via the ```picosdk-python-wrappers``` package.
    
To import the package and perform data acquisition with sound cards:
```python
import measpy as mp
```
This will import the classes ```mp.Measurement```, ```mp.Signal```, ```mp.Spectral``` and ```mp.Weighting```.

To do data acquisition one has to select the module that corresponds to the target device. If it is a soundcard:
```python
from measpy.audio import audio_run_measurement
```
If it is a NI daq card:
```python
from measpy.ni import ni_run_measurement
```

In theses modules, there's also the ```audio_get_devices``` and ```ni_get_devices``` functions to get a liste of devices present in the system. To get the list of devices, do for example:

```python
from measpy.audio import audio_get_devices
l = audio_get_devices()
print(l)
```

## TODO

Things to improve, implement, fix:
- New processing methods have to be implemented (e.g. FIR, convolution, etc.)
- Improve plotting methods
- Other In/Out synchronization methods (for now a method using a peak sync before measurement is implemented)
- More documentation
- More testing scripts
- GUI ?

## Usage example

Consider the following experiment in which we want to record a pressure and an acceleration while we send a white noise at the sound card output (typical sound and vibration transfer function measurement):
- a white noise between 20Hz and 20kHz is sent to output 1 of the soundcard
- a pressure is acquired at input 1 (Unit pascals)
- an acceleration is acquired at input 2 (unit m/s^2)
- the sampling frequency is 44100Hz
- the calibration of signal conditionners are : 1V/pascal, 0.1V/(m/s^2)
- the soundcard input gain is such that for 5V at its inputs, the sample amplitude is 1.0 (input or output). This value is referred to as 0dB full scale (0dBFS)
- the duration of the measurement is 5s
- the soundcard name is 'My card', as given by measpy.audio.audio_get_devices()

To setup and run the measurement, do:
```python
import measpy as mp
from measpy.audio import audio_run_measurement
import matplotlib.pyplot as plt

M1 = mp.Measurement(out_sig='noise',
                    out_map=[1],
                    out_name=['Out1'],
                    out_desc=['Output noise'],
                    out_sig_freqs=[20.0,20000.0]
                    in_map=[1,2],
                    in_name=['Press','Acc']
                    in_desc=['Pressure at point A','Acceleration at point B'],
                    in_cal=[1.0,0.1],
                    in_unit=['Pa','m/s**2'],
                    in_dbfs=[5.0,5.0],
                    out_dbfs=[5.0,]
                    dur=5,
                    in_device='My card',
                    out_device='My card')
audio_run_measurement(M1)
```

To plot the resulting data:
```python
M1.plot()
plt.show()
```

Save Measurement object into a pickle file:
```
M1.to_pickle('file.mck')
```

Load a measurement file into the Measurement object M2:
```python
M2=mp.Measurement.from_pickle('file.mck')
```
Other formats are possible : A combination of a cvs file and wave files, or a json+wave files. See from_csvwav() or from_jsonwav() methods.

Compute transfer functions:
```python
sp = M1.tfe()
```
This compute the transfer function between the output signal and all the input signals as a dict of ```Spectral``` objects. The method that is actually used depends on the output type, as specified in the out_sig property of the measurement object. If a 'noise' or '*wav' type signal is sent, Welch's method is used. If a 'logsweep' type is used, Farina's method is used. This basic helper function works only if there is a unique output.

In general is is preferable to work on individual signals. All the acquired and sent signals are stored into the data property. It is basically a dict of signals, the keys being set by the in_name and out_name arguments when measurement is called. To plot only the measured pressure:
```python
a=M1.data['Press'].plot()
```
If no arguments are given, the ```plot``` method of signal objects creates a new figure, draw the signal with the correct dimension and put the correct labels on all axes. This methods returns an axes object that can be used later to plot new signals on the same figure.

One may want to calculate the power spectral density of the pressure (Welch's method on 2**12 points, 50% overlaping):
```python
PressPSD = M1.data['Press'].psd(nperseg=2**12)
```

```PressPSD``` is now a ```Spectral``` class object. It has its own methods. For instance, to plot the data:
```python
PressPSD.plot()
```

You might want to compute the transfer function between ```M1.data['Acc']``` and ```M1.data['Press']```:
```python
tfap = M1.data['Acc'].tfe_welch(M1.data['Press'],nperseg=2**12)
```

And use this ```Spectral``` object to compute the impulse response:
```python
Gap = tfap.irfft()
```

This could be done in one step:
```python
Gap = M1.data['Acc'].tfe_welch(M1.data['Press'],nperseg=2**12).irfft()
```

To remove frequencies below 20Hz and above 20kHz before computing the impulse:
```python
Gap = M1.data['Acc'].tfe_welch(M1.data['Press'],nperseg=2**12).filterout([20,20000]).irfft()
```

Units are preserved during the operations:
```python
print(Gap.unit)
```
should give something like pascal * second**2 / m

Individual signals can also be saved as a pair of files: a .csv containing the metadata informations of the signal (sampling frequency, calibration informations, name...), and a .wav file, containing the actual data (dimensionless):
```python
M1.data['Press'].to_csvwav('Pressure')
```
This will create Pressure.csv and Pressure.wav, that can be reloaded later with:
```python
press_sig = mp.Signal.from_csvwav('Pressure')
```
