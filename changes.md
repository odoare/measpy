# Changelog

## v0.4

### New features

1. **Real-time plotting during acquisition** *(new `measpy/_plot_tools.py`)* : plot data as it is being acquired, fed from the acquisition queue: `basic_plot`, `basic_plot_with_fft` (live time signal + spectrum), a `BlitManager` for fast redraws, interactive axis rescaling and buttons.

2. **Threaded data pipelines** *(new `measpy/_data_tools.py`)* : queue-based plumbing to process acquired data while it streams: `Pipeline_manager` and `Process_manager` threads, queue dispatch to several consumers, the `use_queues` decorator turning a list-to-list function into a queue-to-queue one, and `Queue2array` / `Queue2prealocated_array`.

3. **NI input terminal configuration** : new `in_sig_config` measurement parameter, accepting `RSE`, `NRSE`, `DIFF`, `PSEUDO_DIFF` or `DEFAULT`.

4. **Pause and stop control of a running NI measurement** : `ni_callback_measurement.run()` now takes `pause` and `stop` `threading.Event` triggers, and the acquisition thread is terminated cleanly.

5. **True moving-average spectrum smoothing** : `nth_oct_smooth` and `nth_oct_smooth_complex` now compute, at **every** frequency of the spectrum, the mean of the values inside the 1/nth octave band centred on it, using cumulative sums. The frequency resolution is preserved. `nth_oct_smooth_complex` gains a `mode` argument: `'amplitude_phase'` (default), `'complex'` and `'power'`.

6. **The csaps dependency is gone** : weightings are now interpolated on a log-frequency scale with the new public `Weighting.values_at_freqs`, which interpolates amplitude in dB, does not extrapolate beyond the tabulated range, and handles negative frequencies as conjugates. `scipy`, used all along but never declared, is now in `install_requires`.

7. **One frequency-range convention across the whole API** : every function taking a frequency range accepts `freqs_range` (tuple/list/array), `freq_min` and `freq_max`. The old names (`fmin`, `fmax`, `freqsrange`, `freqs`) still work and emit a visible deprecation message. `Signal.iir` keeps its standard scipy parameter `Wn` as a valid, non-deprecated alternative; `freqs_range`/`freq_min`/`freq_max` simply take precedence over it when given.

8. **Three new tutorials** : Measurement, spectrum smoothing and weighting, and synchronized measurements, as notebooks and in the online documentation.

9. **A test suite** : 57 new tests (`tests/test_regressions.py`, `tests/test_smoothing.py`), one per fixed bug plus coverage of the smoothing core and the frequency-range parsing.

10. **Automatic file naming and better HDF5 chunking** : `ensure_new_filename` avoids overwriting, and chunk sizes are derived from the HDF5 recommendations.

### Bug fixes

11. **31 bugs fixed**, the ones most likely to have affected results being:

    - `Signal.imag()` and `Spectral.imag()` returned the *real* part
    - `Signal.dB()` modified its reference argument in place, corrupting the global `PREF`/`VREF` constants for the rest of the session
    - `timelag` was biased by half a sample, and `sync_render` cut one sample too late — synchronized measurements were left with a 0 or 1 sample error depending on the parity of the delay
    - `Signal.cut()` with no argument returned a 1-sample signal
    - adding a numpy array to a `Signal` silently did nothing (`value`/`values` typo)
    - `from_csvwav` read integer WAV data with a wrong scale **and** a flipped sign
    - `Spectral.dBV()`/`dBu()` were unusable (converted to pascals), and the dBV/dBu reference values were swapped and wrong
    - `dB_SPL`/`dB_SVL` discarded their unit conversion, giving results off by 60 dB on a mPa or kPa spectrum
    - `Signal.to_csv()` raised on every call
    - `Measurement(in_sig=None)` / `(out_sig=None)` raised, and several constructor checks compared a list with itself or read `out_map` before checking it existed
    - multichannel acquisition with `audio_run_measurement` wrote to a discarded copy

### Behaviour changes to be aware of

12. Some fixes change numerical output:

    - smoothed spectra now contain a value at every frequency bin instead of a spline through ~120 band centres (expect more detail than v0.3 produced)
    - `DBVREF` is now 1 V and `DBUREF` 0.7746 V (they were 1.414 V and 1 V), and `dB_SPL`/`dB_SVL`/`dBV`/`dBu` return a spectrum of unit `dB`
    - synchronized measurements shift by up to one sample compared with v0.3
    - `examples/5_multichannel_signals.ipynb` is now `6_multichannel_signals.ipynb`, tutorial 5 being the synchronized measurement one

Contributors since v0.3: Clément Savaro, Olivier Doaré.
