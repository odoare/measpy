import measpy as mp
import nidaqmx
from measpy.ni import ni_run_measurement, ni_get_true_fs

#list of Ni devices availables
listdev = nidaqmx.system.System.local().devices

# Choose input and output device
dev_in = listdev[0]
dev_out = listdev[1]

#Sampling frequency, 
fs = 60000
fs = ni_get_true_fs(fs, dev_out.name)

#2^nbits+2 samples to scann every adc code
nsamps = 2**18
dur = nsamps / fs
#voltage range for input and output
volt_range = 10

#Signal definition
#Amplitude = 95% range
A = .95*volt_range
#Frequency duch as frq/fs = n/samps with n and nsamps coprimes (so n odd since nsamps is a poqer od 2)
freq = (501 / nsamps) * fs
Sout = mp.Signal.sine(fs, dur, A, freq)

#Choose from wich device the clock is taken (in or out)
ref_device="out"
#Digital terminal to transmit clock signal [source, reciever]
clock_terminal = ["PFI0", "PFI3"]

if ref_device=="out":
    freq_device=dev_in.name
elif ref_device=="in":
    freq_device=dev_out.name
    
#Digital terminal that recieve clock signal to measure frequency
freq_map="PFI2"
#Number of samples used to measure frequency
freq_nsamp=15

M = mp.measurement.Measurement(
    fs=fs,
    dur=dur,
    device_type="ni",
    in_sig=[mp.Signal(fs=fs)],
    in_device=dev_in.name,
    in_map=[1],
    in_range=[volt_range],
    in_sig_config="DIFF",
    out_sig=Sout,
    out_device=dev_out.name,
    out_map=[1],
    out_range=[volt_range],
    ref_device=ref_device,
    clock_terminal=clock_terminal,
    freq_device=freq_device,
    freq_map=freq_map,
    freq_nsamp=freq_nsamp,
)
M.intask_delay = 10e-6 #Optionnal : adjust delay of input task to compensate hardware delay
ni_run_measurement(M)