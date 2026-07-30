""" Regression tests

Each test of this file corresponds to a bug that has been fixed.
They are kept here so that these bugs cannot come back unnoticed.
"""

import unittest
import sys
from os import remove

sys.path.insert(0, "./measpy")

import numpy as np
import matplotlib
matplotlib.use('Agg')

import measpy as mp
from measpy.signal import PREF, VREF, DBVREF, DBUREF

FS = 1000
DUR = 1.0


class TestSignalRegressions(unittest.TestCase):
    """ measpy.signal.Signal """

    def setUp(self):
        self.s = mp.Signal.noise(fs=FS, dur=DUR, unit='Pa')

    def test_imag_is_not_real(self):
        """ Signal.imag returned the real part """
        c = self.s.similar(values=self.s.values+2j*np.ones(self.s.length))
        self.assertTrue(np.allclose(c.imag().values, 2.0))
        self.assertTrue(np.allclose(c.real().values, self.s.values))

    def test_dB_does_not_modify_reference(self):
        """ Signal.dB converted the reference quantity in place, which
            corrupted the module constants PREF and VREF """
        pref_before = PREF.copy()
        vref_before = VREF.copy()
        mp.Signal(fs=FS, unit='kPa', values=np.ones(10)).dB_SPL()
        mp.Signal(fs=FS, unit='km/s', values=np.ones(10)).dB_SVL()
        self.assertEqual(PREF, pref_before)
        self.assertEqual(VREF, vref_before)

    def test_dB_SPL_value(self):
        """ 1 Pa is 94dB SPL, whatever the unit used """
        for unit, value in (('Pa', 1.0), ('mPa', 1000.0), ('kPa', 1e-3)):
            sig = mp.Signal(fs=FS, unit=unit, values=value*np.ones(10))
            self.assertAlmostEqual(float(sig.dB_SPL().values[0]), 93.9794, places=3)

    def test_cut_defaults_to_whole_signal(self):
        """ Signal.cut() returned a one sample signal """
        self.assertEqual(self.s.cut().length, self.s.length)
        self.assertTrue(np.allclose(self.s.cut().raw, self.s.raw))

    def test_cut_empty_and_reversed(self):
        """ Signal.cut with equal positions raised a ValueError """
        self.assertEqual(self.s.cut(pos=(10, 10)).length, 0)
        self.assertTrue(np.allclose(self.s.cut(pos=(100, 0)).raw,
                                    self.s.raw[100:0:-1]))

    def test_add_ndarray(self):
        """ 'values' was misspelled 'value': the array was not added """
        out = self.s + np.ones(self.s.length)
        self.assertTrue(np.allclose(out.values, self.s.values+1))

    def test_add_silence_no_argument(self):
        """ Signal.add_silence() raised an UnboundLocalError """
        self.assertEqual(self.s.add_silence().length, self.s.length)
        self.assertEqual(self.s.add_silence(extras=(10, 20)).length,
                         self.s.length+30)

    def test_values_volts_raw_precedence(self):
        """ The winner between values, volts and raw depended on the
            order of the keyword arguments """
        sig = mp.Signal(fs=FS, cal=2.0, raw=np.zeros(3), values=np.ones(3))
        self.assertTrue(np.allclose(sig.values, 1.0))
        sig = mp.Signal(fs=FS, dbfs=2.0, raw=np.zeros(3), volts=np.ones(3))
        self.assertTrue(np.allclose(sig.volts, 1.0))

    def test_string_calibration_needs_invcal(self):
        """ The invcal argument never reached its property setter, and
            the guard tested the always existing 'invcal' property """
        sig = mp.Signal(fs=FS, cal='2*x', invcal='y/2', values=np.ones(3))
        self.assertTrue(np.allclose(sig.raw, 0.5))
        self.assertTrue(np.allclose(sig.values, 1.0))
        with self.assertRaises(ValueError):
            mp.Signal(fs=FS, cal='2*x', values=np.ones(3))

    def test_multichannel_messages(self):
        """ All the multichannel exceptions mentioned transfer functions """
        multi = mp.Signal.pack((self.s, self.s))
        for method in ('fft', 'rfft', 'unit_to_std', 'spectrogram'):
            with self.assertRaises(NotImplementedError) as ctx:
                getattr(multi, method)()
            self.assertNotIn('Transfer function', str(ctx.exception))

    def test_to_csv(self):
        """ to_csv wrote a float instead of a row and raised an Error """
        name = '/tmp/measpy_test_to_csv'
        self.s.to_csv(name, includetime=True)
        with open(name+'.csv', encoding='utf-8') as file:
            lastline = file.read().strip().split('\n')[-1]
        remove(name+'.csv')
        time, value = (float(v) for v in lastline.split(','))
        self.assertAlmostEqual(time, self.s.time[-1])
        self.assertAlmostEqual(value, self.s.raw[-1])

    def test_timelag_is_exact(self):
        """ Signal.corr shifted the lag axis by half a sample, so that
            timelag returned the delay minus half a sample """
        for delay in (0, 1, 137):
            shifted = self.s.similar(values=np.roll(self.s.values, delay))
            self.assertAlmostEqual(shifted.timelag(self.s)*FS, delay, places=6)

    def test_from_csvwav_integer_data(self):
        """ min and max were swapped, integer wav data was read with a
            wrong scale and a wrong sign """
        import scipy.io.wavfile as wav
        name = '/tmp/measpy_test_int16'
        data = np.linspace(-1, 1, 100)
        self.s.similar(raw=data).to_csvwav(name)
        wav.write(name+'.wav', FS, (data*32767).astype(np.int16))
        out = mp.Signal.from_csvwav(name)
        remove(name+'.csv')
        remove(name+'.wav')
        self.assertAlmostEqual(float(out.raw.max()), 1.0, places=3)
        self.assertAlmostEqual(float(out.raw.min()), -1.0, places=3)


class TestSpectralRegressions(unittest.TestCase):
    """ measpy.signal.Spectral and measpy.signal.Weighting """

    def test_imag_is_not_real(self):
        sp = mp.Spectral(values=np.array([1+2j, 3+4j]), fs=4)
        self.assertTrue(np.allclose(sp.imag().values, [2, 4]))

    def test_unit_to_accepts_string(self):
        """ Spectral.unit_to failed with a string, unlike Signal.unit_to """
        sp = mp.Spectral(values=np.array([1+0j]), fs=4, unit='Pa*s')
        self.assertAlmostEqual(float(np.real(sp.unit_to('mPa*s').values[0])), 1000.0)

    def test_dbv_dbu_references(self):
        """ dBV and dBu converted to pascals, and the two reference
            values were swapped """
        self.assertAlmostEqual(float(DBVREF.v), 1.0)
        self.assertAlmostEqual(float(DBUREF.v), 0.774597, places=5)
        sp = mp.Spectral(values=np.array([1.0+0j, 0.5+0j]), fs=4, unit='V')
        self.assertTrue(np.allclose(sp.dBV().values, [0.0, -6.0206], atol=1e-3))
        self.assertTrue(np.allclose(sp.dBu().values, [2.2185, -3.8021], atol=1e-3))
        # Same physical value given in another unit gives the same dB
        spm = mp.Spectral(values=np.array([1000.0+0j]), fs=4, unit='mV')
        self.assertAlmostEqual(float(spm.dBV().values[0]), 0.0)
        with self.assertRaises(ValueError):
            mp.Spectral(values=np.array([1+0j]), fs=4, unit='Pa').dBV()

    def test_dur_constructor_length(self):
        """ The length was computed as fs*dur/2+1, which is wrong for
            an odd number of samples """
        for dur, nsamples in ((1.0, 1000), (0.999, 999)):
            sp = mp.Spectral(fs=FS, dur=dur)
            self.assertEqual(sp.sample_number, nsamples)
            self.assertEqual(sp.length,
                             mp.Signal(fs=FS, raw=np.zeros(nsamples)).rfft().length)

    def test_weighting_from_csv(self):
        """ 'Weighting' was passed as the phase argument """
        name = '/tmp/measpy_test_weighting.csv'
        mp.WDBA.to_csv(name)
        w = mp.signal.Weighting.from_csv(name)
        remove(name)
        self.assertTrue(np.allclose(w.freqs, mp.WDBA.freqs))
        self.assertTrue(np.allclose(w.amp, mp.WDBA.amp))
        self.assertTrue(np.allclose(w.phase, 0.0))


class TestMeasurementRegressions(unittest.TestCase):
    """ measpy.measurement.Measurement constructor """

    @staticmethod
    def _sig(fs=FS):
        return mp.Signal(fs=fs)

    def test_none_signals(self):
        """ out_sig=None raised a TypeError, in_sig=None left the
            attribute undefined """
        meas = mp.Measurement(out_sig=None, in_sig=[self._sig()], in_map=[1], dur=1)
        self.assertIsNone(meas.out_sig)
        meas = mp.Measurement(in_sig=None,
                              out_sig=[mp.Signal.noise(fs=FS)], out_map=[1])
        self.assertIsNone(meas.in_sig)

    def test_out_sig_without_out_map(self):
        """ out_map was read before being known to exist (KeyError) """
        meas = mp.Measurement(out_sig=[mp.Signal.noise(fs=FS)])
        self.assertEqual(meas.out_map, [1])

    def test_wrong_signal_lists(self):
        """ A list of non-Signals left out_sig undefined instead of
            raising, and a wrong type was not detected """
        with self.assertRaises(TypeError):
            mp.Measurement(out_sig=[1, 2], out_map=[1, 2])
        with self.assertRaises(TypeError):
            mp.Measurement(in_sig='not a signal')
        with self.assertRaises(ValueError):
            mp.Measurement(in_sig=[self._sig(44100), self._sig(48000)],
                           in_map=[1, 2], dur=1)

    def test_in_map_length_check(self):
        """ The check compared in_sig with itself """
        with self.assertRaises(Exception):
            mp.Measurement(in_sig=[self._sig(), self._sig(), self._sig()],
                           in_map=[1, 2], dur=1)

    def test_sync_render(self):
        """ sync_render cut the input signals one sample too late, which
            compensated the half sample bias of timelag only for some
            values of the delay """
        for delay in (0, 1, 137, 138):
            out = mp.Signal.noise(fs=FS, dur=1.0, freqs_range=(20, FS/2), unit='V')
            meas = mp.Measurement(device_type='ni', fs=FS,
                                  out_sig=[out], out_map=[1],
                                  in_sig=[mp.Signal(fs=FS, unit='V')], in_map=[1],
                                  dur=1.0, in_device='Dev1', out_device='Dev1')
            meas.sync_prepare(out_chan=0)
            # Simulate an acquisition delayed by 'delay' samples
            meas.in_sig[0].raw = np.roll(meas.out_sig[0].raw, delay)
            measured = meas.sync_render(out_chan=0, in_chan=0)
            self.assertAlmostEqual(measured*FS, delay, places=6)
            # After the synchronization the two signals are sample aligned
            self.assertLess(
                np.abs(meas.in_sig[0].raw-meas.out_sig[0].raw).max(), 1e-12)

    def test_pico_freq_stop(self):
        """ freq_stop was read from the freq_start parameter """
        meas = mp.Measurement(device_type='pico', in_sig=[self._sig()],
                              in_map=[1], dur=1, freq_start=100, freq_stop=5000)
        self.assertEqual(meas.freq_start, 100)
        self.assertEqual(meas.freq_stop, 5000)


class TestHarmonicDistoRegressions(unittest.TestCase):
    """ measpy.signal.Signal.harmonic_disto """

    def test_harmonic_phase_is_flat_for_memoryless_nonlinearity(self):
        """ decal used np.ceil where the window position (ns) uses
            np.round, leaving up to a full sample of uncorrected phase
            error on the extracted harmonics (amplified by the harmonic
            order once frequencies are realigned). For a memoryless,
            zero-delay nonlinearity the phase of each harmonic must be
            flat versus frequency: any residual slope reveals a leftover
            fractional-sample timing error in the extraction.

            fs/dur/freq_min/freq_max below are chosen so that
            L*log(order)*fs has a fractional part below 0.5 for orders
            2 and 3, which is exactly the regime where np.ceil and
            np.round disagreed.
        """
        fs = 48000
        x = mp.Signal.log_sweep(fs=fs, dur=3.0, freqs_range=(100.0, 10000.0), unit='V')
        y = x.similar(values=x.values + 0.15*x.values**2 + 0.08*x.values**3, unit='V')

        _, Hfr, _, delay = y.harmonic_disto(nh=3, freqs_range=(100.0, 10000.0),
                                            nsmooth=48, win_max_length=2**14)
        self.assertAlmostEqual(delay, 0.0, places=6)

        for order in (1, 2, 3):
            H = Hfr[order-1]
            # Only the part of the realigned band where the harmonic's
            # raw content stays well inside the original sweep band is
            # analysed, higher orders alias close to fmax otherwise.
            band = (H.freqs > 300) & (H.freqs < 9000.0/order)
            self.assertGreater(band.sum(), 20)
            phase = np.unwrap(np.angle(H.values[band]))
            freqs = H.freqs[band]
            slope = np.polyfit(freqs, phase, 1)[0]
            residual_delay_samples = -slope/(2*np.pi)*fs
            self.assertLess(abs(residual_delay_samples), 0.1,
                            f"harmonic {order}: residual delay "
                            f"{residual_delay_samples:.3f} samples")

    def test_non_integer_dl_does_not_raise(self):
        """ dl=prop_before*win_max_length was not guaranteed to be an
            integer, silently truncating (not rounding) the window
            position for odd win_max_length/prop_before combinations """
        x = mp.Signal.log_sweep(fs=48000, dur=2.0, freqs_range=(100.0, 10000.0), unit='V')
        y = x.similar(values=x.values + 0.1*x.values**2, unit='V')
        Hnl, Hfr, thd, delay = y.harmonic_disto(
            nh=2, freqs_range=(100.0, 10000.0), win_max_length=999, prop_before=0.33)
        self.assertTrue(np.all(np.isfinite(Hfr[0].values)))
        self.assertTrue(np.all(np.isfinite(Hfr[1].values)))


class TestUtilsRegressions(unittest.TestCase):
    """ measpy.utils """

    def test_mic_calibration_freq(self):
        """ nperseg was passed as a positional argument, and the
            Wref+nperseg branch called apply_weighting on a Signal """
        sig = mp.Signal.noise(fs=8000, dur=2.0, unit='Pa')
        ref = mp.Signal.noise(fs=8000, dur=2.0, unit='Pa')
        for kwargs in ({}, {'nperseg': 1024},
                       {'Wref': mp.WDBA}, {'Wref': mp.WDBA, 'nperseg': 1024}):
            weighting = mp.mic_calibration_freq(sig, ref, **kwargs)
            self.assertIsInstance(weighting, mp.signal.Weighting)
            self.assertGreater(len(weighting.freqs), 0)


if __name__ == '__main__':
    unittest.main()
