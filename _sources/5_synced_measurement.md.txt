# Tutorial 5: Synchronized measurements

When a signal is sent to an output of a card and recorded at one of its inputs,
the recorded signal is **late**. The delay comes from several sources:

- the buffers of the output and input streams of the driver,
- the anti-aliasing and reconstruction filters of the converters,
- the propagation time in the system under test itself (the time the sound takes
  to travel from the loudspeaker to the microphone, for instance).

The first two are properties of the acquisition chain, they have nothing to do
with the system being measured, and they must be removed before the measurement
is interpreted. They are often of the order of several milliseconds, which is
already a lot: 5 ms is a full turn of phase at 200 Hz.

This delay matters for anything that uses the phase:

- the phase of a transfer function,
- the impulse response obtained by inverse Fourier transform,
- the complex smoothing of a spectrum (see the previous tutorial),
- the estimation of a harmonic distortion.

This tutorial shows the two ways measpy deals with the problem.

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

## 1. Hardware synchronization

The best solution is to have no delay to correct at all, by driving the outputs
and the inputs of the card from the same sample clock.

With NI cards, measpy tries to do this automatically: when the output task is
configured, the sample clock of the analog input (```/Dev1/ai/SampleClock```) is
used as the clock source of the output. A message is printed when this succeeds:

```
Use of /Dev1/ai/SampleClock as output clock : success !
```

If the device does not support it, measpy falls back to the onboard clock and
prints a message. The inputs and the outputs then drift with respect to each
other, and the delay has to be measured.

Audio cards do not offer this, and the delay between ```sd.playrec``` output and
input is not guaranteed to be constant from one run to the next: it has to be
measured every time.

## 2. Software synchronization by cross-correlation

The general solution implemented in measpy consists in measuring the delay on
the data itself. This is done by the two methods ```sync_prepare``` and
```sync_render``` of the ```Measurement``` class:

1. ```sync_prepare``` adds a duration of silence (one second by default) before
   and after the selected output signal. The task is made longer accordingly, so
   that the whole output signal is recorded even if the input is late.
2. The measurement is run as usual.
3. ```sync_render``` computes the time lag between one selected input signal and
   the output signal by looking for the maximum of their cross-correlation
   (the ```Signal.timelag``` method), then cuts all the input signals so that
   they are aligned with the output signal, and removes the added silences.

The measured delay is returned in seconds.

The three steps are wrapped in one function per device type:

| Device | Function |
| --- | --- |
| Audio card | ```measpy.audio.audio_run_synced_measurement``` |
| NI DAQ card | ```measpy.ni.ni_run_synced_measurement``` |

Both take the measurement object, and optionally ```out_chan``` and
```in_chan```, the indices (in ```M.out_sig``` and ```M.in_sig```) of the signals
used to compute the delay. They return the measured delay.

## 3. A complete example

To measure the delay of an acquisition chain, the simplest setup is a **loopback
cable**: connect the first output of the card directly to its first input. What
is then measured is the latency of the card alone.

The measurement below sends a five second noise, records it back, and re-syncs
the input.


```python
from measpy.ni import ni_run_synced_measurement, ni_get_devices

sysdevs = ni_get_devices().device_names
print(sysdevs)
indev = outdev = sysdevs[0]
```


```python
FS = 44100
DUR = 5

# The signal that is sent
s_out = mp.Signal.noise(fs=FS, dur=DUR, freqs_range=(20, 20000), unit='V')

# The empty signal that will receive the loopback
s_in = mp.Signal(fs=FS, unit='V', desc='Input voltage')

M = mp.Measurement(device_type='ni',
                   fs=FS,
                   out_sig=[s_out],
                   out_map=[1],
                   in_sig=[s_in],
                   in_map=[1],
                   dur=DUR,
                   in_device=indev,
                   out_device=outdev)

# Run the measurement, and re-sync the acquired signals
d = ni_run_synced_measurement(M)

print(f'Measured delay: {d*1000:.3f} ms ({round(d*FS)} samples)')
```

The same measurement with an audio card only differs by the device type and by
the function that is called. Note the ```dbfs``` property of the input signal,
which converts the [-1,1] samples of the card into volts.


```python
from measpy.audio import audio_run_synced_measurement, audio_get_devices

print(audio_get_devices())
```


```python
s_in_audio = mp.Signal(fs=FS, unit='V', dbfs=1.4, desc='Input voltage')

Ma = mp.Measurement(device_type='audio',
                    fs=FS,
                    out_sig=[s_out],
                    out_map=[1],
                    in_sig=[s_in_audio],
                    in_map=[1],
                    dur=DUR,
                    in_device='default',
                    out_device='default')

d = audio_run_synced_measurement(Ma)
print(f'Measured delay: {d*1000:.3f} ms ({round(d*FS)} samples)')
```

## 4. Checking the result

After a successful synchronization on a loopback cable, the transfer function
between the output and the input is that of the card itself: a flat magnitude
close to 0 dB, and a phase close to zero over the whole bandwidth. A residual
phase that grows linearly with the frequency means that a delay is left, of
value $\tau = -\frac{1}{2\pi}\frac{d\varphi}{df}$ (which is what the
```group_delay``` method computes).


```python
# The two signals are now aligned in time
a = M.in_sig[0].plot()
M.out_sig[0].plot(ax=a)
```


```python
# Transfer function of the acquisition chain
G = M.in_sig[0].tfe_welch(M.out_sig[0], nperseg=2**14)
G.nth_oct_smooth_complex(24, freqs_range=(20, 20000)).plot()

# The remaining delay, in seconds, should be a small fraction of a sample
print(f'Residual time lag: {M.in_sig[0].timelag(M.out_sig[0])*1e6:.2f} us')
```

## 5. Using an extra input channel as the synchronization reference

The loopback cable is also the way to synchronize a measurement whose system
under test has its own delay, and whose input signals cannot be used as a
reference. In that case:

- one output channel drives both the system under test and, through a cable, one
  of the inputs of the card,
- this extra input is used as the synchronization channel (```in_chan```),
- the other inputs measure the system.

After the synchronization, the remaining delay in the measured signals is the
one of the system itself (propagation time), which is usually what is being
measured.


```python
# Channel 1: loopback cable, used for the synchronization
# Channel 2: the microphone that measures the system
s_loop = mp.Signal(fs=FS, unit='V', desc='Loopback')
s_mic = mp.Signal(fs=FS, unit='Pa', cal=2.0, desc='Microphone')

M2 = mp.Measurement(device_type='ni',
                    fs=FS,
                    out_sig=[s_out],
                    out_map=[1],
                    in_sig=[s_loop, s_mic],
                    in_map=[1, 2],
                    dur=DUR,
                    in_device=indev,
                    out_device=outdev)

# in_chan=0 selects s_loop as the synchronization reference.
# All the input signals, including the microphone, are shifted by the
# same amount.
d = ni_run_synced_measurement(M2, in_chan=0, out_chan=0)
print(f'Latency of the acquisition chain: {d*1000:.3f} ms')

# What is left between the output and the microphone is the propagation
# time in the system under test
tof = M2.in_sig[1].timelag(M2.out_sig[0])
print(f'Time of flight: {tof*1000:.3f} ms, i.e. {tof*343:.3f} m of air')
```

## 6. Doing it by hand

The wrapper functions are only three lines, and it is sometimes useful to run
the steps separately, for instance to run several measurements with the same
added silence, or to keep the non-synchronized data:

```python
from measpy.ni import ni_run_measurement

M.sync_prepare(out_chan=0, added_samples=2*FS)  # 2 seconds of silence
ni_run_measurement(M)
d = M.sync_render(out_chan=0, in_chan=0, added_samples=2*FS)
```

```added_samples``` is the number of samples of silence added before and after
the output signal, it defaults to ```M.fs``` (one second). **The same value must
be given to both methods.** Increase it if the latency of the system is larger
than one second, decrease it to shorten the measurement.

The delay itself can also be obtained on any pair of signals with the
```timelag``` method, which returns the time at which their cross-correlation is
maximum:

```python
d = M.in_sig[0].timelag(M.out_sig[0])
```

and a signal can be shifted in time without touching its samples, by changing
its ```t0``` property:

```python
shifted = M.in_sig[0].delay(-d)
```

## 7. Practical remarks

- The cross-correlation needs the input and the output to be **correlated**. A
  noise or a sweep works well. A pure sine does not: the cross-correlation is
  periodic and the delay is only known modulo the period.
- The delay is estimated to the nearest sample. At 44100 Hz, one sample is
  23 microseconds, which is a phase error of 0.16 degrees at 20 Hz but 165
  degrees at 20 kHz. For a high frequency phase measurement, use a high
  sampling frequency.
- If the measured delay is close to the added silence duration, the input signal
  is truncated: increase ```added_samples```.
- The synchronization is done once for a given card, a given sampling frequency
  and a given buffer size. If none of these change, the delay measured with a
  loopback can be reused, but it costs little to measure it every time.
