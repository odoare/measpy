# measpy - Measurements with Python

Contact: olivier.doare@ensta-paris.fr

measpy is a set of classes and methods to help the data acquisition and analysis of signals. It is mainly acoustics and vibrations oriented. This package is very recent, it is still incomplete and many bugs may appear.

The base classes are:
- ```Signal```: It is basically a class that contain a 1D numpy array, an a few other properties to describe the data as: sampling frequency, calibration and unit.
- ```Spectral```: A spectral data class that contains the complex amplitudes as a 1D numpy array for frequencies up to the Nyquist frequency, and some properties as sampling frequency, unit, description
- ```Measurement``` : A class that describe a data acquisition process, its outputs (Signal objects), its inputs (Signal objects)...
- ```Weighting``` : Weighting spectral functions (Not yet fully test/functionnal)

For now, these daq devices are implemented :
- Audio cards, via the ```sounddevice``` package,
- NI DAQ cards, via the ```nidaqmx``` package.

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

In theses modules, there's also the ```audio_get_devices``` and ```ni_get_devices``` functions to get a liste of devices present in the system.

## TODO

Things to improve, implement, fix:
- Many processing methods have to be implemented
- Improve plotting methods
- Other In/Out synchronization methods (implemented for now using a peak sync before measurement)
- More documentation
- More testing scripts
- GUI ?

## Usage example

Consider the following experiment:
- a white noise between 20Hz and 20kHz is sent to output 1 of the soundcard
- a pressure is acquired at input 1 (Unit pascals)
- an acceleration is acquired at input 2 (unit m/s^2)
- the sampling frequency is 44100Hz
- the calibration of signal conditionners are : 1V/pascal, 0.1V/(m/s^2)
- the soundcard input is 5V for a unit sample (input or output)
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
This compute the transfer function between the output signal and all the input signals as a dict of ```Spectral``` objects. The method that is actually used depends on the output type. If a 'noise' or '*wav' type signal is sent, Welch's method is used. If a 'logsweep' type is use, Farina's method is used. This basic helper function works only if there is a unique output.

In general is is preferable to work on individual signals. All the acquired and sent signals are stored into the data property. It is basically a dict of signals, the keys being set by the in_name and out_name arguments when measurement is called. To plot only the measured pressure:
```python
M1.data['Press'].plot()
```

Calculate the power spectral density of the pressure (Welch's method on 2**12 points, 50% overlaping):
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
