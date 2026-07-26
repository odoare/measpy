""" Tests of the frequency range arguments and of the spectrum smoothing """

import unittest
import sys
import warnings

sys.path.insert(0, "./measpy")

import numpy as np
import matplotlib
matplotlib.use('Agg')

import measpy as mp
from measpy._tools import (parse_freq_range,
                           band_moving_average,
                           MeaspyDeprecationWarning)

FS = 48000


class TestFreqRangeParsing(unittest.TestCase):
    """ measpy._tools.parse_freq_range """

    def test_defaults(self):
        self.assertEqual(parse_freq_range(default=(5.0, 20000.0)), (5.0, 20000.0))

    def test_freqs_range_forms(self):
        for freqs in ((20, 20000), [20, 20000], np.array([20, 500, 20000])):
            self.assertEqual(parse_freq_range(freqs), (20.0, 20000.0))
        # The lowest and highest values are taken, whatever the order
        self.assertEqual(parse_freq_range((20000, 20)), (20.0, 20000.0))

    def test_scalars_take_precedence(self):
        self.assertEqual(parse_freq_range((20, 20000), freq_max=8000),
                         (20.0, 8000.0))
        self.assertEqual(parse_freq_range((20, 20000), freq_min=100, freq_max=8000),
                         (100.0, 8000.0))
        self.assertEqual(parse_freq_range(freq_min=100, default=(5.0, 20000.0)),
                         (100.0, 20000.0))

    def test_invalid(self):
        with self.assertRaises(TypeError):
            parse_freq_range(1000)
        with self.assertRaises(ValueError):
            parse_freq_range([1000])
        with self.assertRaises(ValueError):
            parse_freq_range(freq_min=20000, freq_max=20)

    def test_deprecated_arguments(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(
                parse_freq_range(deprecated={'fmin': (100, 'freq_min'),
                                             'fmax': (8000, 'freq_max')}),
                (100.0, 8000.0))
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(issubclass(c.category, MeaspyDeprecationWarning)
                            for c in caught))
        # The new arguments win over the deprecated ones
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(
                parse_freq_range(freq_min=50,
                                 deprecated={'fmin': (100, 'freq_min')}),
                (50.0, None))


class TestFreqRangeArguments(unittest.TestCase):
    """ The freqs_range/freq_min/freq_max arguments of the measpy functions """

    def test_signal_creation(self):
        for method in (mp.Signal.noise, mp.Signal.log_sweep):
            sig = method(fs=FS, dur=0.5, freqs_range=(50, 5000))
            self.assertEqual((sig.freq_min, sig.freq_max), (50.0, 5000.0))
            sig = method(fs=FS, dur=0.5, freq_min=50)
            self.assertEqual((sig.freq_min, sig.freq_max), (50.0, 20000.0))

    def test_iir_equivalent_calls(self):
        sig = mp.Signal.noise(fs=FS, dur=0.5)
        ref = sig.iir(N=4, freqs_range=(100, 2000))
        self.assertTrue(np.allclose(
            ref.values, sig.iir(N=4, freq_min=100, freq_max=2000).values))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            deprecated = sig.iir(N=4, Wn=(100, 2000))
        self.assertTrue(np.allclose(ref.values, deprecated.values))
        self.assertEqual(len(caught), 1)
        # Lowpass and highpass only use one end of the range
        self.assertTrue(np.allclose(
            sig.iir(N=4, btype='lowpass', freq_max=1000).values,
            sig.iir(N=4, btype='lowpass', freqs_range=(20, 1000)).values))

    def test_deprecated_calls_still_work(self):
        sig = mp.Signal.noise(fs=FS, dur=0.5)
        spectrum = sig.rfft()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            old = spectrum.nth_oct_smooth_complex(12, 50, 5000)
            self.assertTrue(np.allclose(
                old.values,
                spectrum.nth_oct_smooth_complex(12, freqs_range=(50, 5000)).values))
            old = sig.tfe_farina((20, 20000))
            self.assertTrue(np.allclose(
                old.values, sig.tfe_farina(freqs_range=(20, 20000)).values))
            old = spectrum.filterout((50, 5000))
            self.assertTrue(np.allclose(
                old.values, spectrum.filterout(freqs_range=(50, 5000)).values))
        self.assertEqual(len(caught), 4)

    def test_filterout_without_argument(self):
        spectrum = mp.Signal.noise(fs=FS, dur=0.5).rfft()
        self.assertTrue(np.allclose(spectrum.filterout().values[1:-1],
                                    spectrum.values[1:-1]))


class TestBandMovingAverage(unittest.TestCase):
    """ measpy._tools.band_moving_average """

    def setUp(self):
        rng = np.random.default_rng(42)
        self.freqs = np.linspace(0, 24000, 4001)
        self.values = rng.normal(size=self.freqs.size) \
            + 1j*rng.normal(size=self.freqs.size)

    def _brute_force(self, freqs_out, n):
        fac = 2**(1/(2*n))
        out = []
        for freq in freqs_out:
            low, high = min(freq/fac, freq*fac), max(freq/fac, freq*fac)
            inband = self.values[(self.freqs >= low) & (self.freqs <= high)]
            out.append(inband.mean() if inband.size else np.nan)
        return np.array(out)

    def test_against_brute_force(self):
        for n in (3, 12, 48):
            self.assertTrue(np.allclose(
                band_moving_average(self.freqs, self.values, n=n),
                self._brute_force(self.freqs, n)))

    def test_output_frequencies(self):
        fout = np.array([1.0, 100.0, 1000.0, 10000.0])
        computed = band_moving_average(self.freqs, self.values, n=6,
                                       freqs_out=fout, empty='nan')
        expected = self._brute_force(fout, 6)
        # 1Hz is below the frequency resolution: the band is empty
        self.assertTrue(np.isnan(computed[0]))
        self.assertTrue(np.allclose(computed[1:], expected[1:]))
        # With empty='interp' no NaN is returned
        interpolated = band_moving_average(self.freqs, self.values, n=6,
                                           freqs_out=fout)
        self.assertFalse(np.any(np.isnan(interpolated)))

    def test_unsorted_frequencies(self):
        """ Full spectra have their frequencies in the numpy fft order """
        freqs = np.fft.fftfreq(2000, 1/FS)
        rng = np.random.default_rng(7)
        values = rng.normal(size=2000)
        computed = band_moving_average(freqs, values, n=3)
        fac = 2**(1/6)
        for i in (1, 500, 1500):
            low = min(freqs[i]/fac, freqs[i]*fac)
            high = max(freqs[i]/fac, freqs[i]*fac)
            self.assertAlmostEqual(
                computed[i], values[(freqs >= low) & (freqs <= high)].mean())

    def test_constant_is_preserved(self):
        ones = np.ones_like(self.freqs)
        self.assertTrue(np.allclose(
            band_moving_average(self.freqs, ones, n=12), 1.0))


class TestSpectrumSmoothing(unittest.TestCase):
    """ Spectral.nth_oct_smooth* """

    def setUp(self):
        np.random.seed(1)
        self.sig = mp.Signal.noise(fs=FS, dur=2.0, freqs_range=(20, 20000))
        spectrum = self.sig.rfft()
        freqs = spectrum.freqs
        # A single resonance at 1kHz
        self.resonance = spectrum.similar(
            values=1/(1-(freqs/1000)**2-0.02j*(freqs/1000)))

    def test_grid_is_preserved(self):
        """ The smoothed spectrum keeps the frequency resolution """
        smoothed = self.resonance.nth_oct_smooth_complex(12, freqs_range=(20, 20000))
        self.assertEqual(smoothed.length, self.resonance.length)
        self.assertEqual(smoothed.fs, self.resonance.fs)
        self.assertTrue(np.allclose(smoothed.freqs, self.resonance.freqs))

    def test_mean_level_is_preserved(self):
        """ Smoothing a noisy spectrum must not change its mean level """
        psd = self.sig.psd()
        smoothed = psd.nth_oct_smooth(3, freqs_range=(20, 20000))
        inband = (psd.freqs > 100) & (psd.freqs < 10000)
        self.assertAlmostEqual(
            float(smoothed.values[inband].mean()/psd.values[inband].mean()),
            1.0, places=2)

    def test_resonance_is_not_shifted(self):
        for n in (48, 24, 12):
            smoothed = self.resonance.nth_oct_smooth_complex(
                n, freqs_range=(20, 20000))
            peak = smoothed.freqs[np.argmax(np.abs(smoothed.values))]
            self.assertAlmostEqual(peak, 1000.0, delta=10.0)

    def test_wider_bands_smooth_more(self):
        peaks = [np.abs(self.resonance.nth_oct_smooth_complex(
            n, freqs_range=(20, 20000)).values).max() for n in (48, 12, 3)]
        self.assertTrue(peaks[0] > peaks[1] > peaks[2])

    def test_modes(self):
        for mode in ('amplitude_phase', 'complex', 'power'):
            smoothed = self.resonance.nth_oct_smooth_complex(
                12, freqs_range=(20, 20000), mode=mode)
            self.assertEqual(smoothed.length, self.resonance.length)
            self.assertTrue(np.all(np.isfinite(smoothed.values)))
        with self.assertRaises(ValueError):
            self.resonance.nth_oct_smooth_complex(12, mode='wrong')

    def test_to_weight(self):
        weighting = self.resonance.nth_oct_smooth_to_weight_complex(
            12, freqs_range=(20, 20000))
        self.assertGreater(len(weighting.freqs), 10)
        self.assertEqual(len(weighting.freqs), len(weighting.amp))
        self.assertEqual(len(weighting.freqs), len(weighting.phase))
        self.assertTrue(np.all(np.isfinite(weighting.amp)))
        # The Weighting can be interpolated back on a spectrum
        back = self.resonance.similar(w=weighting)
        self.assertEqual(back.length, self.resonance.length)
        self.assertTrue(np.all(np.isfinite(back.values)))


class TestWeightingInterpolation(unittest.TestCase):
    """ measpy.signal.Weighting.values_at_freqs (replaces csaps) """

    def test_tabulated_values(self):
        """ The interpolation must go through the tabulated points """
        for freq, expected in ((100.0, -19.1), (1000.0, 0.0), (10000.0, -2.5)):
            self.assertAlmostEqual(
                20*np.log10(abs(mp.WDBA.values_at_freqs(freq))),
                expected, places=6)

    def test_no_extrapolation(self):
        """ Outside of the table, the value of the closest end is used """
        outside = mp.WDBA.values_at_freqs(np.array([0.1, 1e6]))
        # First and last values of the dBA table
        self.assertAlmostEqual(20*np.log10(abs(outside[0])), -85.4, places=6)
        self.assertAlmostEqual(20*np.log10(abs(outside[1])), -9.3, places=6)

    def test_zero_and_negative_frequencies(self):
        """ Full spectra contain f=0 and negative frequencies """
        values = mp.WDBA.values_at_freqs(np.array([0.0, -440.0, 440.0]))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values[1], np.conj(values[2]))

    def test_applied_to_spectrum(self):
        spectrum = mp.Signal.noise(fs=FS, dur=1.0, unit='Pa').rfft()
        weighted = spectrum.apply_dBA()
        self.assertTrue(np.all(np.isfinite(weighted.values)))
        index = np.argmin(np.abs(spectrum.freqs-1000))
        self.assertAlmostEqual(
            float(np.abs(weighted.values[index]/spectrum.values[index])),
            1.0, places=6)


if __name__ == '__main__':
    unittest.main()
