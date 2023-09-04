# measpy - Measurements with Python

![](https://github.com/odoare/measpy/workflows/docs/badge.svg)

(c) 2021-2023 Olivier Doaré
Contact: olivier.doare@ensta-paris.fr

measpy is a set of classes and methods that
- helps signal processing and analysis using rapid and compact Python scripting, thanks to the functional programming paradigm proposed by this package,
- ease and unify the data acquisition process with various DAQ cards

Documentation: [https://odoare.github.io/measpy](https://odoare.github.io/measpy)

**WARNING:** *major incompatible changes have been made to the measurement class for v0.1, if backward compatibility is needed, the old measurement class system is kept in the pre v0.1 branch. Changes in this branch will only concern eventual bug fixing.*

The base classes defined by ```measpy``` are:
- ```Signal```: This is the core class of the package. It defines a signal through a sampling frequency, a physical unit and a list of samples and a description. Additional properties can be defined in order to take into account calibration or time shifting of the signal with respect to a reference time.
- ```Spectral```: This class represents the signals in the Fourier space. A spectral object contains the complex amplitudes as a 1D numpy array for frequencies up to the Nyquist frequency or the sampling frequency, and some properties as sampling frequency, unit and description.
- ```Measurement``` : A class that describe a data acquisition process, its outputs (Signal objects), its inputs (Signal objects)...
- ```Weighting``` : A weighting class holds complex values for a list of frequencies, and methods to to smoothing, interpolation, etc.

For now, data acquisition with these daq devices are implemented :
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
If it is a Picoscope of the ps2000 series:
```python
from measpy.ps2000 import ps2000_run_measurement
```
If it is a Picoscope of the ps4000 series:
```python
from measpy.ps4000 import ps4000_run_measurement
```

In theses modules, there's also the ```audio_get_devices``` and ```ni_get_devices``` functions to get a list of devices present in the system. To get the list of devices, do for example:

```python
from measpy.audio import audio_get_devices
l = audio_get_devices()
print(l)
```

## Usage example

Consider the following experiment in which we want to record a pressure and an acceleration while we send a white noise at the sound card output (typical sound and vibration transfer function measurement):
- a white noise between 20Hz and 20kHz is sent to output 1 of the soundcard
- a pressure is acquired at input 1 (Unit pascals)
- an acceleration is acquired at input 2 (unit m/s^2)
- the sampling frequency is 44100Hz
- the calibration of signal conditionners are : 1V/pascal, 0.1V/(m/s^2)
- the soundcard input gain is such that for 5V at its inputs, the sample amplitude is 1.0 (input or output). This value is referred to as 0dB full scale (0dBFS). It given in the soundcard specifications or has to be calibrated using known signals.
- the duration of the measurement is 5s
- the soundcard name is 'My card', as given by measpy.audio.audio_get_devices()

First step is to import the modules and prepare the signals. We create a five seconds noise signal for output, and two empty signals with the correct properties for the inputs.
```python
import measpy as mp
from measpy.audio import audio_run_measurement
sout = mp.Signal.noise(fs=44100, freq_min=20, freq_max=20000, dur=5)
sin1 = mp.Signal(desc = 'Pressure', dbfs=5.0, cal=1.0, unit='Pa' )
sin2 = mp.Signal(desc = 'Acceleration', dbfs=5.0, cal=0.1, unit='m*s**(-2)' )
```

We then setup and run the measurement:
```python

M1 = mp.Measurement(out_sig=[sout],
                    out_map=[1],
                    in_sig=[sin1,sin2],
                    in_map=[1,2],
                    dur=5,
                    in_device='My card',
                    out_device='My card',
                    device_type='audio')
audio_run_measurement(M1)
```

The data is stored in the list of ```Signal``` objects ```M1.in_sig```. To plot the pressure:
```python
M1.in_sig[0].plot()
plt.show()
```

The measurement can be saved in a directory:
```
M1.to_dir('my_measurement')
```
The created directory contains the measurement parameters in a file params.csv, individual signals as pairs of csv and wav files, one pair for each signal. The csv contains the signal parameters, the wav file contains the raw data points.

The measurement can then be restored using:
```python
M2=mp.Measurement.from_dir('my_measurement')
```

An example of analysis consists in computing transfer function between sent signal and acquired signals. For instance, the Welch method can be implemented using:
```python
H=M2.in_sig[0].tfe_welch(M2.out_sig[0])
```
The output of ```tfe_welch()``` is an object of the ```measpy.signal.Spectral``` class. This class has also a plotting method:
```python
H.plot()
plt.show()
```

This ```Spectral``` object can then be used to compute an impulse response:
```python
G = H.irfft()
```

## Functional programing paradigm

Most signal processing methods of ```Signal``` and ```Spectral``` classes return a ```Signal``` or ```Spectral``` object. This allows to write signal processing scripts by chaining these methods. For instance the impulse response calculation above can be done in one step:
```python
G = M2.in_sig[0].tfe_welch(M2.out_sig[0]).irfft()
```
We might want to remove frequencies below 20Hz and above 20kHz before computing the impulse. This can be done in the same line of code, in the functional programing way:
```python
G = M2.in_sig[0].tfe_welch(M2.out_sig[0]).filter_out([20,20000]).irfft()
```

## Units

Units are preserved during the operations:
```python
print(Gap.unit)
```
should give something like pascal * second**2 / m

## Documentation

Additionnal documentation and examples can be found in the ./docs and ./examples directories of the project. The main page for web documentation is [https://odoare.github.io/measpy](https://odoare.github.io/measpy).

## Releasing

Releases are published automatically when a tag is pushed to GitHub.

Example:
```bash

   # Set next version number
   export RELEASE=x.x.x

   # Create tags
   git commit --allow-empty -m "Release $RELEASE"
   git tag -a $RELEASE -m "Version $RELEASE"

   # Push
   git push upstream --tags
```
