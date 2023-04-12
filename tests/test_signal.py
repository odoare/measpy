import unittest
import sys
sys.path.insert(0, "measpy")
import measpy as mp
import numpy as np
import unyt
import matplotlib.pyplot as plt
from os import remove

err_tol = 1e-4
class TestSignal(unittest.TestCase):

    global s1, sp1, s2, s3, s4, s5, dur, fs
    dur = 3.0
    fs = 44100

    # A 3s signal with a even number of samples
    s1 = mp.Signal.noise(fs=fs,dur=dur,amp=1.0,freqs=[20.0,20000.0],unit='1',cal=1.0,dbfs=1.0)
    
    # A 3s signal with an odd number of samples
    sp1 = mp.Signal.noise(fs=fs+1,dur=dur,amp=1.0,freqs=[20.0,20000.0],unit='1',cal=1.0,dbfs=1.0)
    s3 = mp.Signal.sine(fs=44100,freq=441,dur=1.0)
    s4 = mp.Signal.log_sweep(fs=44100,freqs=[20,20000],dur=1.0)
    s5 = mp.Signal.log_sweep(fs=48000,freqs=[20,20000],dur=1.0)

    def test_durations(self):
        for s in (s1,sp1):
            self.assertEqual(s.dur,dur,"Calculation in Signal.dur is wrong ?")
            self.assertEqual(1/(s.time[1]-s.time[0]),s.fs,"Calculation of Signal.time is wrong ?")
            self.assertEqual(s.length,s.fs*s.dur,"Calculation of Signal.length is wrong ?")
            
    def test_signalops(self):
        self.assertLess((s1-s1).max(),err_tol, "Difference of two equal signals should be zero" )
        self.assertLess(((s1/s1)-1).max(),err_tol, "Difference of two equal signals should be zero" )
        self.assertLess( abs((abs(-(s1/s1))-1).max()), err_tol, "Absolute value of a -1 constant signal should be one"  )
        self.assertLess( abs((abs(-(s1*(1/s1)))-1).max()), err_tol, "Absolute value of a -1 constant signal should be one"  )

    def test_corr(self):
        self.assertLess(abs(abs(s1.corr(s1.similar(values=np.roll(s1.values,s1.fs))).tmax())-1*unyt.s), err_tol, "Max of correlation between signals should be at 1 sec.")
        self.assertLess(abs(abs(s1.add_silence(extrat=(1,0)).corr(s1.add_silence(extrat=(0,1))).tmax())-1*unyt.s), err_tol, "Max of correlation between signals should be at 1 sec.")
        self.assertLess(abs(abs((s1.timelag(s1.similar(values=np.roll(s1.values,s1.fs)))))-1), err_tol, "Calculated timelag should be 1 sec.")

    def test_normalize(self):
        self.assertEqual(s1.normalize().max(), 1.0, "Should be one")

    def test_rms(self):
        self.assertLess(s1.rms()-np.sqrt(2), err_tol, "RMS of 1V sine should be sqrt(2)")

    def test_hilbert(self):
        self.assertLess((s3.timelag(s3.hilbert())-1/(4*s3.fs)),err_tol, "Hilbert transform should induce a pi/2 (1/4 period) phase shift")

    def test_fft(self):
        for s in (s1,sp1):
            print (s)
            self.assertEqual(s.fft().dur,dur,"Calculation of Spectral.dur is wrong ?")
            self.assertEqual(s.rfft().dur,dur,"Calculation of Spectral.dur is wrong ?")
            self.assertEqual(s.fft().freqs[1],1/s.dur,"Calculation of frequency vector is wrong ?")
            self.assertLess(min(abs(s.rfft(norm='ortho').values_at_freqs(range(30,19000,100))))-1,err_tol,"Noise spectrum has to be unitary in the (20,20000) frequency range")
            self.assertLess((s.rfft().irfft()-s).rms(),err_tol,"RFFT + IRFFT gets the same signal")
            self.assertLess((s.fft().ifft()-s).rms(),err_tol,"FFT + IFFT gets the same signal")
            self.assertEqual(s.fft().sample_number,s.length,"After FFT the sample number of the original signal is correct")
            self.assertEqual(s.rfft().sample_number,s.length,"After RFFT the sample number of the original signal is correct")
            
    def test_resample(self):
        a1=abs(s4.rfft(norm='forward').values_at_freqs(range(30,19000,100)))
        a2=abs(s5.rfft(norm='forward').values_at_freqs(range(30,19000,100)))
        self.assertLess(max(abs(a2-a1)),err_tol, "Difference between signals should be zero")

        a1=abs(s4.rfft(norm='forward').values_at_freqs(range(30,19000,100)))
        a2=abs(s5.resample(s4.fs).rfft(norm='forward').values_at_freqs(range(30,19000,100)))
        self.assertLess(max(abs(a2-a1)),err_tol, "Difference between signals should be zero")

    def test_importexport(self):
        filname = 'test'+str(int(np.random.random()*10000))
        s1.to_csvwav(filname)
        self.assertLess((s1-mp.Signal.from_csvwav(filname)).max(),err_tol, "Difference of two equal signals should be zero" )
        remove(filname+".csv")
        remove(filname+".wav")

if __name__ == '__main__':
    unittest.main()
