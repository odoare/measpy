
# Before v0.1

- [x] Signal.classmethods : change freqs list to freq_min & freq_max

- [x] Convert cal, dbfs to _cal, _dbfs optional properties

- [x] Signal.to_csv

- [x] Measurement -> new oldbranch

- [x] Daqtask -> Measurement

- [x] to_dir when dir exists

- [x] neperseg defaults ?

# v0.2

- [x] HDF5

- [x] Multichannel signals

- [x] Continuous integration sphinx doc compilation

- [x] Continuous integration : pip publication

# v0.3

.....

# v0.4

- [x] Remove the csaps dependency

- [x] Spectrum smoothing based on a true moving average over 1/nth octave
      bands (measpy._tools.band_moving_average), computed at every frequency
      of the spectrum

- [x] Weighting interpolation on a log frequency scale
      (Weighting.values_at_freqs), replaces the smoothing spline

- [x] Same frequency range arguments everywhere : freqs_range, freq_min and
      freq_max. The old arguments (fmin, fmax, freqsrange, freqs, Wn) still
      work and print a deprecation message

- [x] Bug fixing session (see tests/test_regressions.py)

- [x] Tutorials 3 (Measurement), 4 (smoothing and weighting) and
      5 (synchronized measurements)

- [x] Add scipy to install_requires of setup.py (it was used but not declared)

# Long term todo

- [ ] Pandas
- [ ] Units everywhere ?
- [ ] Continuous integration : tests

# Roadmap : numpy/scipy functionalities to bring into measpy

A function deserves a measpy wrapper when measpy adds something that scipy
cannot : unit propagation, calibration, fs/t0 bookkeeping, multichannel
handling, labelled plots. When it adds nothing, the user is better served
calling scipy directly on the `values` property.

## Priority 1 : gaps that block common workflows

- [ ] **Time integration** `Signal.integrate(method='trapezoid', detrend=True)`
      with `scipy.integrate.cumulative_trapezoid`. `Signal.diff` exists but has
      no counterpart : going from acceleration to velocity and displacement is
      not possible today. measpy handles the unit (`unit*s`), the integration
      constant and the detrending that makes double integration usable.

- [ ] **Zero phase filtering** `Signal.iir(..., zero_phase=False)` with
      `scipy.signal.sosfiltfilt`. Only the causal `sosfilt` is available, which
      adds phase to a signal in a package that otherwise cares a lot about it.

- [ ] **Detrending** `Signal.detrend(type='linear')` with
      `scipy.signal.detrend`. Currently only passed through to welch/csd.

- [ ] **Resampling** `Signal.resample(fs, method='poly')` with
      `scipy.signal.resample_poly`, and expose `Signal.decimate(q)`
      (`scipy.signal.decimate` is already imported for the queue code).
      The current FFT based `scipy.signal.resample` assumes periodicity and
      rings at the edges of transients.

- [ ] **Octave band filtering (time domain)**
      `Signal.octave_bands(n=3, freqs_range=...)` returning a multichannel
      signal, one channel per band. Built from `scipy.signal.butter` +
      `sosfiltfilt`, reusing `_tools.nth_octave_bands`. Complements the
      spectrum smoothing, and is what the standards require for band levels
      and decay measurements.

- [ ] **Reverberation time** `Signal.schroeder()` and
      `Signal.reverberation_time(n=3, method='T20')`. Backward integration
      (`numpy.cumsum` on the reversed squared signal) and a linear fit of the
      dB decay (`numpy.polyfit`). Needs the octave band filtering above.

- [ ] **Peak analysis of spectra** `Spectral.find_peaks(...)` with
      `scipy.signal.find_peaks`, `peak_widths` and `peak_prominences`, to get
      resonance frequencies, -3dB bandwidths and Q factors. measpy converts
      the bin indices into frequencies.

- [ ] **Missing excitation signals** :
    - [ ] MLS, `scipy.signal.max_len_seq`
    - [ ] Linear sweep, `scipy.signal.chirp`
    - [ ] Pink and brown noise : one `color=` argument in
          `_tools.noise`, which already shapes the spectrum amplitude

- [ ] **Sound level meter quantities** `Signal.leq(integration_time, weighting)`
      and `Signal.percentile_levels([10,50,90])` (`numpy.percentile` on
      `rms_smooth`).

## Priority 2 : useful wrappers

- [ ] `Signal.envelope()` : `abs(hilbert)`, `hilbert_ana` is already there
- [ ] `Signal.savgol(window, order)` : `scipy.signal.savgol_filter`,
      peak preserving alternative to the rectangular `smooth`
- [ ] `Signal.medfilt(k)` : `scipy.signal.medfilt`, despiking
- [ ] `Signal.stft()` returning data : `spectrogram()` only plots today
- [ ] `Signal.crest_factor` and `Signal.kurtosis` properties, next to `rms`
- [ ] `Signal.periodogram()` : `scipy.signal.periodogram`, for deterministic
      signals where Welch averaging is not appropriate
- [ ] Filter frequency response as a `Spectral` object : `scipy.signal.sosfreqz`
- [ ] `Spectral.tfe(..., estimator='H1'|'H2'|'Hv')` : `tfe_welch` is H1 only,
      H2 and Hv are a few lines from the same csd/welch calls
- [ ] Use `scipy.fft.next_fast_len` in `tfe_farina`, which pads to
      `2**ceil(log2(n))` and can nearly double the amount of computation

## Not worth wrapping

Filter design (`butter`, `cheby`, `zpk`...), `scipy.optimize`, `scipy.stats`
distributions, interpolation, windows beyond `get_window`. They are used
directly on the `values` property, wrapping them would grow the API without
adding units or bookkeeping.

# Known issues and structural work

- [ ] **Multichannel support is mostly free**. Ten methods raise
      `NotImplementedError` for multichannel signals (`fft`, `rfft`, `psd`,
      `tfe_welch`, `coh`, `window`, `fade`, `unit_to`, `spectrogram`,
      `timelag`), but `welch`, `csd`, `coherence` and `numpy.fft` all accept
      an `axis` argument, and the channels are already stored as columns.
      Passing `axis=0` and letting `Spectral` hold a 2D array would remove
      most of these restrictions. Highest value per line of code.

- [ ] **`io_sync` does nothing**. The parameter is accepted by the
      `Measurement` constructor, stored, saved and restored, but no
      acquisition module reads it (only a comment in ni.py mentions it).
      Either implement it or remove it.

- [ ] **scipy legacy functions**. `scipy.signal.spectrogram` and
      `scipy.signal.stft` are legacy as of scipy 1.14, the documentation
      points to `ShortTimeFFT`. `Signal.spectrogram` uses the legacy one.

- [ ] `Spectral.__mul__` and `Spectral.__truediv__` reject numpy arrays,
      whereas `Signal` accepts them.

- [ ] `Signal.__getitem__` only takes an integer channel index : slicing a
      signal in time needs `cut`.

- [ ] `Spectral.plot` prints a `RuntimeWarning: divide by zero` when the
      spectrum contains exact zeros (after `filterout` for instance).
      The non positive values should be masked before `log10`.

- [ ] The phase branch of `nth_oct_smooth_complex(mode='amplitude_phase')`
      unwraps the phase of the raw spectrum before averaging it. On very
      noisy data the unwrap itself can drift. Unwrapping after the smoothing
      of the complex value could be more robust.

- [ ] `Spectral.dB_SPL` and friends require exactly the reference dimension
      (Pa, V...), so they cannot be used on an `rfft` output whose unit is
      Pa*s or V*s. Decide what the unit of a spectrum should be.
