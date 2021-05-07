# measpy - Measurement with Python

measpy is a set of classes and methods to help the data acquisition and analysis of signals. It is mainly acoustics and vibrations oriented.

The base classes are:
- Signal: It is basically a class that contain a 1D numpy array, an a few other properties to describe the data as: sampling frequency, calibration and unit.
- Measurement : A class that describe a data acquisition process, its outputs (Signal objects), its inputs (Signal objects)...

## Usage example

Example of basic measurement where:
- a white noise between 20Hz and 20kHz is sent to output 1
- a pressure is acquired at input 1
- an acceleration is acquired at input 2
- the sampling frequency is 44100Hz

```python
import measpyaudio as ma
import matplotlib.pyplot as plt
import numpy as np

M1 = ma.Measurement(out_sig='noise',
                    out_map=[1],
                    out_desc=['Output noise'],
                    in_map=[1,2],
                    in_desc=['Pressure','Acceleration'],
                    in_cal=[1.0,1.0],
                    in_unit=['Pa','m.s-2'],
                    dur=5)
M1.run_measurement()
```

To plot the resulting data:
```python
M1.plot_with_cal()
plt.show()
```

Load Measurement object:
```
M1.to_pickle('1.pck')
```
