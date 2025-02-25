# measpy/_tools.py
#
# ----------------------------
# Utilities for measpy package
# ----------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

import csv
import numpy as np
import h5py
import numbers
from unyt import Unit
from pathlib import Path
from abc import ABC, abstractmethod
from queue import Empty

import matplotlib.pyplot as plt


class plot_data_from_queue(ABC):
    """
    Abstract class used to analyse and plot data that are feed into a queue (by a measurment callback)
    """
    plot_attribute = [
        "plotbuffer",
        "x_data",
        "fig",
        "axes",
        "lines",
        "autoscale",
    ]
    def __init__(self, fs, updatetime=0.1, plotbuffersize=2000, nchannel = 1):
        """
        :param fs: Sampling frequency
        :type fs: float
        :param updatetime: Time beetwen 2 plot update, defaults to 0.1
        :type updatetime: flaot, optional
        :param plotbuffersize: Number of datapoint plotted, defaults to 2000
        :type plotbuffersize: int, optional

        """
        for x in self.plot_attribute:
            setattr(self, x, None)
        self.timesincelastupdate = 0
        self.plotbuffersize = plotbuffersize
        self.updatetime = updatetime
        self.timeout = 0.1 * updatetime
        self.fs = fs
        self.timeinterval = 1 / self.fs
        self.databuffersize = max(int(updatetime * self.fs),plotbuffersize)
        if nchannel>1:
            self.data_buffer = np.zeros((self.databuffersize, nchannel))
        else:
            self.data_buffer = np.zeros((self.databuffersize))
        self.plot_setup()
        for x in self.plot_attribute:
            if getattr(self, x) is None:
                raise TypeError(
                    f"Subclasses 'plot_setup' method must set {x} to a non-None value"
                )

    @abstractmethod
    def plot_setup(self):
        """
        Create a plot object
        This method should be overridden, should create attribute used by the class:
            plotbuffer: list of Numpy array that contain data to be plotted
            x_data : list of Numpy array that contain the x axis values of data
            fig : matplotlib figure
            axes : list of matplotlib axes
            lines : list of  matplotlib lines
            autoscale = list of boolean, if true the corresponding axis is autoscaled after each plot

        """
        pass

    @abstractmethod
    def data_process(self):
        """
        Process data to be plotted
        This method should be overridden, it modify plotbuffer and x_data using data_buffer

        """
        pass

    def _plotting_buffer(self):
        self.data_process()
        for ax, line, x, y, autoscale in zip(
            self.axes, self.lines, self.x_data, self.plotbuffer, self.autoscale
        ):
            line.set_xdata(x)
            line.set_ydata(y)
            if autoscale:
                ax.relim()
                ax.autoscale_view()
        self.rescaling()
        plt.pause(0.0001)
        self.timesincelastupdate = 0

    def rescaling(self):
        """
        This method is called automatically to rescale the data after each plot
        By default, it does nothing.

        """
        pass

    def _update_data_buffer(self, item):
        n_values = len(item)
        # item = np.asarray(item) * 0.001  #mv to V
        self.timesincelastupdate += n_values
        self.data_buffer[:-n_values] = self.data_buffer[n_values:]
        self.data_buffer[-n_values:] = item

    def update_plot(self, updatetime=None):
        updatetime = self.updatetime if updatetime is None else updatetime
        try:
            if (item := self.dataqueue.get(timeout=self.timeout)) is not None:
                self._update_data_buffer(item)
                if self.timesincelastupdate * self.timeinterval > updatetime:
                    self._plotting_buffer()
        except (Empty, AttributeError):
            pass

    def update_plot_until_empty(self):
        try:
            while (item := self.dataqueue.get(timeout=10)) is not None:
                self._update_data_buffer(item)
                if self.timesincelastupdate * self.timeinterval > self.updatetime:
                    self._plotting_buffer()
            if self.timesincelastupdate > 0:
                self._plotting_buffer()
        except (Empty, AttributeError):
            pass

    def close(self):
        plt.close(self.fig)

    @property
    def dataqueue(self):
        try:
            return self._dataqueue
        except AttributeError:
            print("No dataqueue defined")
            return None

    @dataqueue.setter
    def dataqueue(self, dataqueue):
        if (
            item := dataqueue.get(timeout=10 * self.timeout)
        ) is not None and item[0].size == self.data_buffer[0].size:
            self._update_data_buffer(item)
            if self.timesincelastupdate * self.timeinterval > self.updatetime:
                self._plotting_buffer()
            self._dataqueue = dataqueue
        else:
            raise ValueError("Invalid queue")

def csv_to_dict(filename):
    """ Conversion from a CSV (produced by the class Measurement) to a dict
          Default separator is (,)
          First row is the key string
          The value is a list
    """
    dd={}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dd[row[0]]=row[1:]
    return dd

def convl(fun,xx):
    if type(xx) in [np.ndarray ,list]:
        yy=list(map(fun,xx))
    else:
        yy=fun(xx)
    return yy

def convl1(fun,xx):
    if type(xx) in [np.ndarray ,list]:
        yy=None if xx[0]=='None' else fun(xx[0])
    else:
        yy=None if xx=='None' else fun(xx)
    return yy

def add_step(a,b):
    if isinstance(a,str):
        return a+'\n -->'+b
    if isinstance(a,list):
        return list(s+'\n -->'+b for s in a)
    else:
        raise TypeError('First argument has to be a string or list of strings')

def wrap(phase):
    """ Opposite of np.unwrap   
    """
    return np.mod((phase + np.pi), (2 * np.pi)) - np.pi

def unwrap_around_index(phase,n):
    """ Unwraps a phase array around a specified index  
    """
    return np.hstack((np.unwrap(phase[n-1::-1])[::-1],np.unwrap(phase[n:])))

def smooth(in_array,l=20):
    ker = np.ones(l)/l
    if len(in_array.shape) == 1:
        return np.convolve(in_array,ker,mode='same')
    elif len(in_array.shape) == 2:
        out = np.zeros_like(in_array)
        for i in range(in_array.shape[1]):
            out[:,i] = np.convolve(in_array[:,i],ker,mode='same')
        return out
    else:
        raise ValueError('This smooth function manages array of dimension <= 2')

def nth_octave_bands(n,fmin=5,fmax=20000):
    """ 1/nth octave band frequency range calculation """
    nmin = int(np.ceil(n*np.log2(fmin*10**-3)))
    nmax = int(np.ceil(n*np.log2(fmax*10**-3)))
    indices = range(nmin,nmax+1)
    f_centre = 1000 * (2**(np.array(indices)/n))
    f2 = 2**(1/n/2)
    f_upper = f_centre * f2
    f_lower = f_centre / f2
    return f_centre, f_lower, f_upper

def picv(long):
    """ Create a 1D-array of length long with a unitary peak in the middle """
    return np.hstack((np.zeros(long),1,np.zeros(long-1)))

def _create_time1(fs,dur):
    return np.linspace(0,dur-1/fs,int(round(dur*fs)))  # time axis

def _create_time2(fs,length):
    return np.linspace(0,(length-1)/fs,length)  # time axis

def create_time(fs,dur=None,length=None):
    if dur==None and length==None:
        raise Exception('dur=duration in s or length=number of samples must be specified.')
    if dur!=None and length!=None:
        raise Exception("dur and length can't be both specified.")
    if dur!=None:
        return _create_time1(fs,dur)
    else:
        return _create_time2(fs,length)

def apply_fades(s,fades):
    if fades[0]>0:
        s[0:fades[0]] = s[0:fades[0]] * ((-np.cos(np.arange(fades[0])/fades[0]*np.pi)+1) / 2)
    if fades[1]>0:
        s[-fades[1]:] = s[-fades[1]:] *  ((np.cos(np.arange(fades[1])/fades[1]*np.pi)+1) / 2)
    return s

def noise(fs, dur, out_amp, freq_min, freq_max):
    """ Create band-limited noise """
    leng = int(dur*fs)
    lengs2 = int(np.ceil(leng/2))
    f = fs*np.arange(lengs2+1, dtype=float)/leng
    amp = ((f > freq_min) & (f < freq_max))*np.sqrt(leng)
    phase = 2*np.pi*(np.random.rand(lengs2+1)-0.5)
    fftx = amp*np.exp(1j*phase)
    s = out_amp*np.fft.irfft(fftx, leng)
    return s

def log_sweep(fs, dur, out_amp, freq_min, freq_max):
    """ Create log sweep """
    L = (dur-1/fs)/np.log(freq_max/freq_min)
    t = create_time(fs, dur=dur)
    s = np.sin(2*np.pi*freq_min*L*np.exp(t/L))
    return out_amp*s

def sine(fs, dur, out_amp, freq):
    s = out_amp*np.sin(2*np.pi*create_time(fs=fs, dur=dur)*freq)
    return (s)

def saw(fs, dur, out_amp, freq):
    return out_amp*wrap(2*np.pi*freq*create_time(fs,dur=dur))/np.pi
    
def tri(fs, dur, out_amp, freq):
    return out_amp*2*np.abs(wrap(2*np.pi*freq*create_time(fs,dur=dur))/np.pi)-1

def t_min(siglist):
    return min(s.t0 for s in siglist)

def t_max(siglist):
    return max(s.t0+s.dur for s in siglist)

def calc_dur_siglist(siglist):
    return t_max(siglist)-t_min(siglist)

def siglist_to_array(siglist):
    durtot = calc_dur_siglist(siglist)
    out = np.zeros((round(durtot*siglist[0].fs),len(siglist)))
    t0s = t_min(siglist)
    for i,s in enumerate(siglist):
        out[round((s.t0-t0s)*s.fs):round(((s.t0-t0s)+s.dur)*s.fs),i] = s.raw
    return out

def get_index(array,value):
    """
    Get the index of the nearest value
    """
    return np.argmin((array-value)**2)

def decodeH5str(h5str):
    if h5str == "None":
        return
    else:
        try:
            return float(h5str)
        except:
            return h5str.strip("\'")

def h5file_write_from_queue(queue, filename, dataset_name, Nchannel):
    """
    Data writer in hdf5 file from a Queue
    :param queue: A Queue which contains data, the shape is [lenght,Nchannel].
    :type queue: queue.Queue
    :param filename: Path of the hdf5 file, it should already exist with an empty extensible dataset.
    :type filename: str,Pat
    :param dataset_name: Name of the hdf5 dataset where data will be written.
    :type dataset_name: str
    :param Nchannel: Number of expected channel,
    :type Nchannel: int
    :return: None
    :rtype: None

    """

    print(f"Starting saving data in {filename}/{dataset_name}")
    with h5py.File(filename, "r+") as H5file:
        item = np.array(queue.get()).transpose()
        #Get dimension of item for multichannel case
        dims = item.shape
        if Nchannel>1:
            assert dims[1] == Nchannel, f"Wrong format, queue item shape = {dims}, for a {Nchannel}-channel signal"
        Npoints = dims[0]
        dataset = H5file[dataset_name]
        #Get the chunksize and datatype of the dataset
        chunksize = dataset.chunks[0]
        datatype = dataset.dtype
        #Define a buffer with chuncksize and datatype
        writebuffer = np.empty((chunksize, Nchannel),dtype=datatype).squeeze()
        buffer_position = _add_item(writebuffer, 0, item, Npoints, dataset, chunksize)
        while (item := queue.get(timeout=5)) is not None:
            item = np.array(item).transpose()
            Npoints = item.shape[0]
            buffer_position = _add_item(
                writebuffer, buffer_position, item, Npoints, dataset, chunksize
            )
        if buffer_position > 0:
            _add_N_data(dataset, writebuffer, buffer_position)


def _add_item(writebuffer, buffer_position, item, Npoints, dataset, chunksize):
    """
    Add new item into buffer and into dataset if it fill up the buffer
    Parameters
    ----------
    writebuffer : np.array
        Data buffer.
    buffer_position : int
        Current position in the buffer (last written data+1).
    item : np.array
        New data.
    Npoints : int
        Number of new data point.
    dataset : HDF5 dataset
        Where to write data.
    chunksize : int
        Size of the dataset chunk.

    Returns
    -------
    Nrest : int
        New position in the buffer (last written data+1).

    """
    #Calcul the number of chunk to write and the new position in buffer
    Nchuncktowrite, Nrest = divmod((buffer_position + Npoints), chunksize)
    old_buffer_position = buffer_position
    #Loop over the number of new chunk in item
    for i in range(Nchuncktowrite):
        #Write item data into buffer until it is full
        writebuffer[buffer_position:] = item[
            i * chunksize
            - np.sign(i) * old_buffer_position : (i + 1) * chunksize
            - old_buffer_position,
        ]
        #Write the whole buffer into dataset
        _add_N_data(dataset, writebuffer, chunksize)
        buffer_position = 0
    if Nrest:
        #Write Nrest data into the buffer
        writebuffer[buffer_position:Nrest] = item[(buffer_position - Nrest):]
    return Nrest


def _add_N_data(dataset, data, N):
    #Write N data point into the dataset
    chunk_start = dataset.shape[0]
    dataset.resize(chunk_start + N, axis=0)
    dataset[chunk_start:] = data[:N]


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def to_list(elt,n):
    if isinstance(elt,(numbers.Number,str,Unit)):
        return [elt] * n
    if isinstance(elt,(list,np.ndarray)):
        return list(elt)
    return [None] * n

def array_mult_unitlist(values,unit):
    """ Multiplies an array with a unyt instance
    or a list of unyts with the same numer of elements
    """
    if isinstance(unit,list):
        return list(values[i]*u for i,u in enumerate(unit))
    return values*unit

def mix_dicts(a,b,na,nb):
    out = {}
    for k,v in a.items():
        vb = b.pop(k,None)
        out[k] = to_list(v,na)+to_list(vb,nb)
    for k,v in b.items():
        va = a.pop(k,None)
        out[k] = to_list(va,na)+to_list(v,nb)
    return out

def H5file_valid(filename):
    if filename:
        path = Path(filename)
        if path.suffix == ".h5":
            if path.exists():
                print(f"file {filename} already exist")
                return False
            if not path.parent.exists():
                print(f"Warning : creating directory for {filename}")
                path.parent.mkdir(parents=True)
        else:
            print(f"Invalid filename : {filename}")
            return False
        return True
    return False
# def _tfe_farina(y, fs, freqs):
#     """ Transfer function estimate
#         Farina's method """
#     leng = int(2**np.ceil(np.log2(len(y))))
#     Y = np.fft.rfft(y, leng)/fs
#     f = np.linspace(0, fs/2, num=round(leng/2)+1)  # frequency axis
#     L = len(y)/fs/np.log(freqs[1]/freqs[0])
#     S = 2*np.sqrt(f/L)*np.exp(-1j*2*np.pi*f*L *
#                               (1-np.log(f/freqs[0])) + 1j*np.pi/4)
#     S[0] = 0j
#     H = Y*S
#     return f, H

# def _tfe_welch(x, y, **kwargs):
#     """ Transfer function estimate (Welch's method)       
#         Arguments and defaults :
#         NFFT=None,
#         Fs=None,
#         detrend=None,
#         window=None,
#         noverlap=None,
#         pad_to=None,
#         sides=None,
#         scale_by_freq=None
#     """
#     if type(x) != type(y):
#         raise Exception(
#             'x and y must have the same type (numpy array or Signal object).')

#     # Set default values for welch's kwargs
#     if not "fs" in kwargs:
#         kwargs["fs"] = x.fs
#     if not "nperseg" in kwargs:
#         kwargs["nperseg"] = 2**(np.ceil(np.log2(x.fs)))

#     if type(x) == Signal:
#         f, p = welch(x.values_in_unit, **kwargs)
#         f, c = csd(y.values_in_unit, x.values_in_unit, **kwargs)
#         out = Spectral(desc='Transfer function between '+x.desc+' and '+y.desc,
#                        fs=x.fs,
#                        unit=y.unit+'/'+x.unit)
#         out.values = c/p
#         return out
#     else:
#         f, p = welch(x, **kwargs)
#         f, c = csd(y, x, **kwargs)
#     return f, c/p
