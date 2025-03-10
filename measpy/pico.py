# measpy/pico.py
#
# ---------------------------------------
# Data acquisition with Picoscope devices
# ---------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy


import numpy as np
from inspect import getfullargspec

from datetime import datetime
from scipy.signal import decimate

import ctypes
from picosdk.ps4000 import ps4000
from picosdk.functions import assert_pico_ok, adc2mV, mV2adc

from threading import Thread, Event
from queue import Queue
import time
from functools import partial
from unyt import Unit
from unyt.exceptions import UnitConversionError

from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.PicoDeviceEnums import picoEnum


from abc import ABC, abstractmethod

from measpy._tools import H5file_valid

# plt.style.use('seaborn-v0_8')

maxtimeout = 10

Channel_state = {key: None for key in ["enabled", "coupling", "range", "buffer"]}

PS2000_channel = {"A": 1, "B": 2, 1: "A", 2: "B"}
PS4000_channel = {"A": 1, "B": 2, "C": 3, "D": 4, 1: "A", 2: "B", 3: "C", 4: "D"}


def transposelist(L):
    return [list(i) for i in zip(*L)]


def dispatch(q_in, qs_out):
    while (item := q_in.get(timeout=maxtimeout)) is not None:
        for q_out in qs_out:
            q_out.put(item)
    for q_out in qs_out:
        q_out.put(None)


def convert_to_mutichannel(listSig):
    ret = listSig[0]
    for i in range(len(listSig) - 1):
        ret = ret.pack_with(listSig[i + 1])
    return ret


def Fill_Signal_Queue(Sig, q_in, unit_in="mV", Ndata=None):
    for s in Sig:
        if s.unit == Unit("1"):
            s.unit = "mV"
            conversion = 1.0
        else:
            try:
                conversion = (1.0 * Unit("mV")).to_value(s.unit)
            except UnitConversionError:
                print(
                    f"Signal unit ({s.unit}) incompatible with mV, unit set to mV"
                )
                s.unit = "mV"
                conversion = 1.0
    if Ndata:
        array = np.zeros((Ndata, Sig.nchannels))
        Sig.raw = conversion * Queue2prealocated_array(q_in, array)
    else:
        Sig.raw = conversion * Queue2array(q_in)


def Queue2prealocated_array(q_in, array):
    datasize = array.shape[0]
    nextSample = 0
    while (item := q_in.get(timeout=maxtimeout)) is not None:
        noOfSamples = len(item)
        lastsample = nextSample + noOfSamples
        try:
            array[nextSample:lastsample] = item
            nextSample = lastsample
        except ValueError:
            N = datasize - nextSample
            if N > 0:
                array[nextSample:] = item[:N]
            break
    return array


def Queue2array(q_in):
    data = []
    while (item := q_in.get(timeout=maxtimeout)) is not None:
        data.extend(item)
    return np.fromiter(
        data, np.dtype((float, (len(data[0]),))), count=len(data)
    ).squeeze()


def findindex(l, e):
    try:
        a = l.index(e)
    except:
        a = None
    return a


def detect_rising_pulses_threshold_ind(
    values,
    threshold,
    ind0,
    previous_data_point,
):
    """
    Detect rising pulse using a threshold

    :param values: Data.
    :type values: List
    :param ind0: indice of the first data point.
    :type ind0: int
    :param previous_data_point: Value of the last data point (indice = ind0-1).
        useful to not lose the peak at ind0
    :type previous_data_point: number
    :param threshold: threshold (in adc values).
    :type threshold: int
    :return: List of indices where a rising pulse is detected.
    :rtype: List
    """
    V = np.asarray(previous_data_point + values)
    rising = np.flatnonzero((V[:-1] <= threshold) & (V[1:] > threshold))
    previous_data_point = [values[-1]]
    id0 = ind0[0]
    ind0[0] += len(values)
    return rising + id0


def ps2000_run_measurement(M, serial=None, filename=None, **kwargs):
    with _ps2000_run_measurement_threaded(
        M, filename=filename, serial=serial, **kwargs
    ) as device:
        device.start()
        device.join()


def ps2000_plot(
    M, plotting_class, plotbuffersize=20000, updatetime=0.1, serial=None, **kwargs
):
    queueplot = Queue()
    nchannel = len(M.in_sig)
    plotting_instance = plotting_class(
        M.fs,
        updatetime=updatetime,
        plotbuffersize=plotbuffersize,
        nchannel=nchannel,
    )
    with _ps2000_run_measurement_threaded(
        M,
        output_queue=queueplot,
        serial=serial,
        stop_event=plotting_instance.stop_event,
        **kwargs,
    ) as device:
        device.start()
        plotting_instance.dataqueue = queueplot
        plotting_instance.update_plot_until_empty()
        device.join()


def ps2000_pulse_detection(
    M, serial=None, min_chunksize_processed=10000, **kwargs
):
    """
    This function needs M to contain the property
     in_threshold : a list of threshold for pulse height (in Volt) for each channel
    """
    if hasattr(M, "in_threshold") and M.in_threshold is not None:
        Q = Queue()
        ps2000_run_measurement(
            M,
            pre_process=detect_rising_pulses_threshold_ind,
            save_into_Signal=False,
            output_queue=Q,
            min_chunksize_processed=min_chunksize_processed,
            serial=serial,
            **kwargs,
        )
        return np.double(Queue2array(Q)) / M.fs
    else:
        raise ValueError("There is no threshold defined")


def ps4000_run_measurement(M, serial=None, filename=None, **kwargs):
    with _ps4000_run_measurement_threaded(
        M, filename=filename, serial=serial, **kwargs
    ) as device:
        device.start()
        device.join()


def ps4000_plot(
    M, plotting_class, plotbuffersize=20000, updatetime=0.1, serial=None, **kwargs
):
    queueplot = Queue()
    nchannel = len(M.in_sig)
    plotting_instance = plotting_class(
        M.fs,
        updatetime=updatetime,
        plotbuffersize=plotbuffersize,
        nchannel=nchannel,
    )
    with _ps4000_run_measurement_threaded(
        M,
        output_queue=queueplot,
        serial=serial,
        stop_event=plotting_instance.stop_event,
        **kwargs,
    ) as device:
        device.start()
        plotting_instance.dataqueue = queueplot
        plotting_instance.update_plot_until_empty()
        device.join()


def ps4000_pulse_detection(
    M, serial=None, min_chunksize_processed=10000, **kwargs
):
    """
    This function needs M to contain the property
     in_threshold : a list of threshold for pulse height (in Volt) for each channel
    """
    if hasattr(M, "in_threshold") and M.in_threshold is not None:
        Q = Queue()
        ps4000_run_measurement(
            M,
            pre_process=detect_rising_pulses_threshold_ind,
            save_into_Signal=False,
            output_queue=Q,
            min_chunksize_processed=min_chunksize_processed,
            serial=serial,
            **kwargs,
        )
        return np.double(Queue2array(Q)) / M.fs
    else:
        raise ValueError("There is no threshold defined")


class Pico_thread(ABC, Thread):
    """
    This needs M to contain the following properties:
        - in_range : a list of strings specifying the voltage range.
            Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
            "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
        - upsampling_factor : upsampling factor
        - in_coupling : Coupling configuration of the channels.
            Can be "ac" or "dc"
    """

    dataqueue = Queue()
    Channels = {}
    channel_name = {}
    Data_type = {}

    def __init__(
        self,
        M,
        output_queue=None,
        pre_process=adc2mV,
        save_into_Signal=True,
        min_chunksize_processed=0,
        filename=None,
        buffersize=20_000,
        max_samples=100_000,
        serial=None,
        stop_event=Event(),
    ):
        """
        :param M: Parameter container
        :type M: measpy.measurement.
        :param output_queue: Send data into this queue, defaults to None
        :type output_queue: queue.Queue
        :param pre_process: Method to process raw data, defaults to adc2mV
        :type pre_process: callable
        :param save_into_Signal: Saving the data into the signal, defaults to True
        :type save_into_Signal: bool, optional
        :param min_chunksize_processed: Minimum size of data to process at once, defaults to 0
        :type min_chunksize_processed: int, optional
        :param filename: Hdf5 file path to direct save into disk, defaults to None
        :type filename: str or pathlib.Path, optional
        :param buffersize: Size of picoscope buffer, defaults to 20_000
        :type buffersize: int, optional
        :param max_samples: Maximum number of sample that stay in memory, defaults to 100_000
        :type max_samples: int, optional
        :param serial: Serial number of the scope, defaults to None
        :type serial: str, optional
        :param stop_event: Event that stop measurment when set, can be defined or used
        :type stop_event: threading.Event()
        """
        # init threading, deamon = True to keyboard interupt when threading
        super().__init__(daemon=True)
        # easier to use only multichannel signal
        if isinstance(M.in_sig, list):
            M.in_sig = convert_to_mutichannel(M.in_sig)
        self.M = M
        if M.device_type != "pico":
            raise ValueError("Error: device_type must be 'pico'.")

        if type(M.out_sig) != type(None):
            print("Warning: out_sig property ignored with picoscopes")

        ## parameters
        self.stop_measurement = stop_event
        self.min_chunksize_processed = min_chunksize_processed
        self.pre_process = pre_process

        ## Buffer are always read in the same order, remapping to match in_map
        self.buffers_map = np.argsort(np.argsort(self.M.in_map))

        # serial number of the scope
        if serial is None:
            print("No picoscope given, searching for connected picoscope")
        self.serial = serial

        # Number of sample stored by picoscope driver that can be retrieved after
        # streaming stopped (by ps2000_get_streaming_values or ps4000GetValuesAsyn)
        self.max_samples = max_samples
        # Size of the temporary buffers used for storing the data
        self.overview_buffer_size = buffersize
        self.save_into_Signal = save_into_Signal
        self.output_queue = output_queue

        # If filename given, save data in hdf5 file
        self.filename = filename
        self.savehdf5 = False
        if H5file_valid(self.filename):
            print(f"Raw measurment will be save in {filename}")
            self.savehdf5 = True

        # Effective sampling frequency
        # If upsampling_factor is > 1, the actual data acquisition is
        # performed at fs*upsampling_factor, and then decimated to
        # the desired frequency

        self.effective_fs = M.fs * M.upsampling_factor
        self.duree_ns = M.dur * 1e9

        # Sample interval is the duration between two consecutive samples
        # As it is the sampling frequency that is specified,
        # and the sampling interval can only take integer increments
        # of 1 nanoseconds, the actual sampling frequency might
        # necessitate adjustments

        if (si := (1e9 / self.effective_fs)).is_integer():
            self.sampleInterval = ctypes.c_int32(int(si))
        else:
            si = int(round(si))
            self.sampleInterval = ctypes.c_int32(si)
            self.effective_fs = 1e9 / si
            print(
                "Warning : Effective sampling frequency ",
                f"changed to nearest possible value of {self.effective_fs} Hz",
            )
            M.fs = self.effective_fs / M.upsampling_factor
            for s in M.in_sig:
                s.fs = M.fs
            print(
                "Warning : Signal sampling frequency fs ",
                f"changed to nearest possible value of {M.fs} Hz",
            )

        numdesiredsamples = int(round(self.effective_fs * self.M.dur))
        numBuffersToCapture = int(
            np.ceil(numdesiredsamples / self.overview_buffer_size)
        )
        self.totalSamples = self.overview_buffer_size * numBuffersToCapture
        self.time_to_fill_half_buffer = (
            0.5 * self.overview_buffer_size / self.M.fs
        )

    def __enter__(self):
        # open comunication with picoscope and set up channel and preprocesing tasks
        self.open_unit()
        try:
            self.channels_setup()
            self.process = self.setup_threads()
        except Exception as e:
            self.__exit__(type(e), e.args, e)
            raise e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close cominucation
        self.close_unit()

    def channels_setup(self):
        # Use Measurment data to set up the channels
        for chan in self.Channels.values():
            i = chan["number"]
            self.setup_channel(i)
            if (
                (Vthreshold := getattr(self.M, "in_threshold", None)) is not None
            ) and (ind := findindex(self.M.in_map, i)) != None:
                chan["adc_threshold"] = mV2adc(
                    [Vthreshold[ind] * 0.001],
                    chan["range"],
                    self.Data_type["max"],
                )[0]
            else:
                chan["adc_threshold"] = None
        self.nchannels = sum([chan["enabled"] for chan in self.Channels.values()])

    def run(self):
        """
        self.start() run this into a thread

        """
        ##start preprocess threads
        for P in self.process:
            P.start()
        # run measurment
        self.start_measurment()

        # Wait for all threads to finish
        print("Waiting for data processing...")
        for P in self.process:
            P.join()

        # Upsampling
        if self.save_into_Signal and self.M.upsampling_factor > 1:
            values = self.M.in_sig.raw
            self.M.in_sig.raw = decimate(
                np.double(values), self.M.upsampling_factor, ftype="fir"
            )
        print("Preprocess data done")

    def dbfs_save(self):
        """
        save h5file with dbfs according to channel parameter
        :return: None
        :rtype: None

        """
        for sig, prange in zip(self.M.in_sig, self.M.in_range):
            sig.unit = "mV"
            sig.dbfs = adc2mV([1], self.pico_range(prange), self.Data_type)[0]
        self.M.datatype = np.dtype(self.Data_type).name
        print("Creating the H5file with measurment parameters")
        self.M.to_hdf5(self.filename)

    def setup_threads(self):
        # threads and outputs set up (preprocessing, put into queue, put into Signal)
        ret = []
        if self.savehdf5:
            # In hdf5 ADC data (integers) are directly saved to file
            # dbfs convert ADC to mVolt and is saved in the parameters
            self.dbfs_save()
            queuesave = Queue()
            queueprocess = Queue()
            ret.append(
                Thread(
                    target=dispatch,
                    args=(self.dataqueue, [queuesave, queueprocess]),
                )
            )
            ret.append(Thread(target=self.M.h5save_data, args=(queuesave,)))
        else:
            queueprocess = self.dataqueue
        process_data = self.setup_preprocess()
        if self.save_into_Signal:
            processed_queue = Queue()
            if self.output_queue:
                queueSignal = Queue()
                ret.append(
                    Thread(
                        target=process_data, args=(queueprocess, processed_queue)
                    )
                )
                ret.append(
                    Thread(
                        target=dispatch,
                        args=(processed_queue, [self.output_queue, queueSignal]),
                    )
                )
                ret.append(
                    Thread(
                        target=Fill_Signal_Queue,
                        args=(self.M.in_sig, queueSignal),
                        kwargs={
                            "Ndata": int(
                                self.M.dur * self.M.fs * self.M.upsampling_factor
                            )
                        },
                    )
                )
            else:
                ret.append(
                    Thread(
                        target=process_data, args=(queueprocess, processed_queue)
                    )
                )
                ret.append(
                    Thread(
                        target=Fill_Signal_Queue,
                        args=(self.M.in_sig, processed_queue),
                        kwargs={
                            "Ndata": int(
                                self.M.dur * self.M.fs * self.M.upsampling_factor
                            )
                        },
                    )
                )
        elif self.output_queue:
            ret.append(
                Thread(
                    target=process_data, args=(queueprocess, self.output_queue)
                )
            )
        elif not self.savehdf5:
            raise ValueError("There are not output for data")
        return ret

    def methods_setup(self):
        # use data in measurment and from channel setup to "partial" the preprocessing method (convert to mV or other)
        if callable(self.pre_process):
            self.pre_process = [self.pre_process] * self.nchannels
        methods = []
        for method, ind in zip(self.pre_process, self.M.in_map):
            chan = self.Channels[self.channel_name[ind]]
            method_args = getfullargspec(method).args
            # range, threshold and maxADC: optional argument not used by the func method
            kwargs = {
                "range": chan["range"],
                "threshold": chan["adc_threshold"],
                "maxADC": self.Data_type,
                "ind0": [0],
                "previous_data_point": [],
            }
            partial_kwargs = {a: b for a, b in kwargs.items() if a in method_args}
            if partial_kwargs:
                methods.append(partial(method, **partial_kwargs))
            else:
                methods.append(method)
        return methods

    def setup_preprocess(self):
        # define func that preprocess raw data
        methods = self.methods_setup()

        if self.min_chunksize_processed > 0:

            def func(queue, queueout):
                chunk_data = [[] for _ in range(self.nchannels)]
                while (item := queue.get(timeout=maxtimeout)) is not None:
                    _ = [c.extend(it) for (c, it) in zip(chunk_data, item)]
                    if len(chunk_data[0]) > self.min_chunksize_processed:
                        values = transposelist(
                            [
                                method(chunk_data[ind])
                                for (ind, method) in zip(
                                    self.buffers_map, methods
                                )
                            ]
                        )
                        queueout.put(values)
                        chunk_data = [[] for _ in range(self.nchannels)]
                if len(chunk_data[0]) > 0:
                    values = transposelist(
                        [
                            method(chunk_data[ind])
                            for (ind, method) in zip(self.buffers_map, methods)
                        ]
                    )
                    queueout.put(values)
                queueout.put(None)

        else:

            def func(queue, queueout):
                while (item := queue.get(timeout=maxtimeout)) is not None:
                    values = transposelist(
                        [
                            method(item[ind])
                            for (ind, method) in zip(self.buffers_map, methods)
                        ]
                    )
                    queueout.put(values)
                queueout.put(None)

        return func

    @abstractmethod
    def open_unit(self):
        """
        Open picoscope

        """
        pass

    @abstractmethod
    def close_unit(self):
        """
        Close picoscope

        """
        pass

    @abstractmethod
    def setup_channel(self, chan_index):
        """setup a channel.

        should define channel status (used or not), channel coupling(ac or dc),
        channel range and channel data buffer.
        """

    @abstractmethod
    def setup_callback(self):
        """
        Setup the callback method to read data.
        should return the callback metod used to read and put data into dataqueue

        """
        pass

    @abstractmethod
    def start_measurment(self):
        """
        Run the scope.
        """

    @staticmethod
    @abstractmethod
    def pico_range(prange):
        """
        Picoscope voltage range (mapped from picoscope lib)
        This property should be overridden, used by dbfs_save
        """
        pass


class _ps2000_run_measurement_threaded(Pico_thread):
    Channels = {
        "A": {"number": 1} | Channel_state.copy(),
        "B": {"number": 2} | Channel_state.copy(),
    }
    channel_name = {1: "A", 2: "B"}
    Data_type = ctypes.c_int16(32767)

    def open_unit(self):
        self.device = ps2000.open_unit(serial=self.serial)
        print("Device info: {}".format(self.device.info))
        self.closed = False

    def close_unit(self):
        if not self.closed:
            # stop
            ps2000.ps2000_stop(self.device.handle)
            # disconect
            self.device.close()
            self.closed = True

    def setup_channel(self, chan_index):
        if (ind := findindex(self.M.in_map, chan_index)) != None:
            enabled = True
            pico_range = self.pico_range(self.M.in_range[ind])
            if self.M.in_coupling[ind].capitalize() == "Dc":
                coupling = "PICO_DC"
            elif self.M.in_coupling[ind].capitalize() == "Ac":
                coupling = "PICO_AC"
            else:
                print(
                    f"Input {self.channel_name[chan_index]} coupling not recognized, set to 'dc'"
                )
                self.M.in_coupling[ind] = "dc"
                coupling = "PICO_DC"
            res = ps2000.ps2000_set_channel(
                self.device.handle,
                picoEnum.PICO_CHANNEL[
                    f"PICO_CHANNEL_{self.channel_name[chan_index]}"
                ],
                enabled,
                picoEnum.PICO_COUPLING[coupling],
                pico_range,
            )
            assert_pico2000_ok(res)
            print(
                f"Channel {self.channel_name[chan_index]}: enabled with range "
                + "PS2000_"
                + self.M.in_range[ind]
                + " ("
                + str(pico_range)
                + ")"
            )
        else:
            enabled = False
            pico_range = self.pico_range("10V")
            coupling = "PICO_DC"
            print(f"Channel {self.channel_name[chan_index]}: disabled")

        self.Channels[self.channel_name[chan_index]]["enabled"] = enabled
        self.Channels[self.channel_name[chan_index]]["range"] = pico_range
        self.Channels[self.channel_name[chan_index]]["coupling"] = coupling

    def setup_callback(self):
        CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
            None,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_int16,
            ctypes.c_uint32,
            ctypes.c_int16,
            ctypes.c_int16,
            ctypes.c_uint32,
        )
        ##Define get_overview_buffers for channel configuration
        if self.nchannels == 2:

            def get_overview_buffers(
                buffers,
                _overflow,
                _triggered_at,
                _triggered,
                _auto_stop,
                n_values,
            ):
                self.dataqueue.put(
                    [buffers[0][0:n_values].copy(), buffers[2][0:n_values].copy()]
                )

        elif self.Channels["A"]["enabled"]:

            def get_overview_buffers(
                buffers,
                _overflow,
                _triggered_at,
                _triggered,
                _auto_stop,
                n_values,
            ):
                self.dataqueue.put([buffers[0][0:n_values].copy()])

        elif self.Channels["B"]["enabled"]:

            def get_overview_buffers(
                buffers,
                _overflow,
                _triggered_at,
                _triggered,
                _auto_stop,
                n_values,
            ):
                self.dataqueue.put([buffers[2][0:n_values].copy()])

        return CALLBACK(get_overview_buffers)

    def start_measurment(self):
        callback = self.setup_callback()
        ########## Signal generator ##########
        if self.M.sig_gen:
            res = ps2000.ps2000_set_sig_gen_built_in(
                self.device.handle,
                int(self.M.offset * 1e6),  # offset voltage in uV
                int(self.M.amp / 2 * 1e6),  # peak-to-peak votage in uV
                self.M.wave,  # type of waveform (0 = sine wave)
                self.M.freq_start,  # start frequency in Hz
                self.M.freq_stop,  # stop frequency in Hz
                self.M.freq_change,  # frequency change per interval in Hz
                self.M.freq_int,  # interval of frequency change in seconds
                self.M.sweep_dir,  # sweep direction (0 = up)
                self.M.sweep_number,  # number of times to sweep
            )
            assert_pico2000_ok(res)
        #######################################
        # Time unit = nanosecond
        sampleUnits = 2
        # no autostop
        autoStopOn = False
        # No downsampling:
        downsampleRatio = 1

        res = ps2000.ps2000_run_streaming_ns(
            self.device.handle,
            self.sampleInterval.value,
            sampleUnits,
            self.max_samples,
            autoStopOn,
            downsampleRatio,
            self.overview_buffer_size,
        )
        assert_pico2000_ok(res)

        now = datetime.now()
        self.M.date = now.strftime("%Y-%m-%d")
        self.M.time = now.strftime("%H:%M:%S")

        print("Start measurement")
        start_time = time.time_ns()

        # loop until wanted duration + maximum time of one loop without buffer overrun or until stop_measurement is set
        margin = min(int(self.max_loop_time * 1e9), self.duree_ns * 0.1)
        try:
            while (
                time.time_ns() - start_time < self.duree_ns + margin
                and not self.stop_measurement.is_set()
            ):
                ps2000.ps2000_get_streaming_last_values(
                    self.device.handle, callback
                )
        except KeyboardInterrupt:
            pass
        self.dataqueue.put(None)
        print("Measurment done")
        overrun = False
        ps2000.ps2000_overview_buffer_status(self.device.handle, overrun)
        if overrun:
            print("Buffer have overrun")
        # Stop the scope
        self.close_unit()

    @staticmethod
    def pico_range(prange):
        return ps2000.PS2000_VOLTAGE_RANGE["PS2000_" + prange]


class _ps4000_run_measurement_threaded(Pico_thread):
    Channels = {
        "A": {"number": 1} | Channel_state.copy(),
        "B": {"number": 2} | Channel_state.copy(),
    }
    channel_name = {1: "A", 2: "B"}
    Data_type = ctypes.c_int16(32767)

    def open_unit(self):
        # picoscope 4000 handle
        self.chandle = ctypes.c_int16()
        # picoscope 4000 status
        self.status = {}
        # Open PicoScope 4000 Series device
        # Returns handle to chandle for use in future API functions
        if self.serial is not None:
            self.status["openunit"] = ps4000.ps4000OpenUnitEx(
                ctypes.byref(self.chandle),
                ctypes.c_char_p(bytes(self.serial, "ascii")),
            )
        else:
            self.status["openunit"] = ps4000.ps4000OpenUnit(
                ctypes.byref(self.chandle),
            )
        assert_pico_ok(self.status["openunit"])
        self.closed = False

    def close_unit(self):
        if not self.closed:
            # Stop the scope
            self.status["stop"] = ps4000.ps4000Stop(self.chandle)
            assert_pico_ok(self.status["stop"])

            # Disconnect the scope
            # handle = chandle
            self.status["close"] = ps4000.ps4000CloseUnit(self.chandle)
            assert_pico_ok(self.status["close"])
            self.closed = True

    def setup_channel(self, chan_index):
        if (ind := findindex(self.M.in_map, chan_index)) != None:
            enabled = True
            pico_range = self.pico_range(self.M.in_range[ind])
            if self.M.in_coupling[ind].capitalize() == "Dc":
                coupling = 1
            elif self.M.in_coupling[ind].capitalize() == "Ac":
                coupling = 0
            else:
                print(
                    f"Input {self.channel_name[chan_index]} coupling not recognized, set to 'dc'"
                )
                self.M.in_coupling[ind] = "dc"
                coupling = 1
            # Configure picoscope channel
            self.status[
                f"setCh{self.channel_name[chan_index]}"
            ] = ps4000.ps4000SetChannel(
                self.chandle,
                ps4000.PS4000_CHANNEL[
                    f"PS4000_CHANNEL_{self.channel_name[chan_index]}"
                ],
                int(enabled),
                coupling,
                pico_range,
            )
            assert_pico_ok(self.status[f"setCh{self.channel_name[chan_index]}"])
            print(
                f"Channel {self.channel_name[chan_index]}: enabled with range "
                + "PS4000_"
                + self.M.in_range[ind]
                + " ("
                + str(pico_range)
                + ")"
            )
            # Create buffer for data collection
            buffer = np.zeros(shape=self.overview_buffer_size, dtype=np.int16)
            # Assigning pointer's buffer
            self.status[
                f"setDataBuffers{self.channel_name[chan_index]}"
            ] = ps4000.ps4000SetDataBuffers(
                self.chandle,
                ps4000.PS4000_CHANNEL[
                    f"PS4000_CHANNEL_{self.channel_name[chan_index]}"
                ],
                buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                self.overview_buffer_size,
            )
            assert_pico_ok(
                self.status[f"setDataBuffers{self.channel_name[chan_index]}"]
            )
        else:
            enabled = False
            pico_range = self.pico_range("10V")
            coupling = 1
            print(f"Channel {self.channel_name[chan_index]}: disabled")
            buffer = None
        self.Channels[self.channel_name[chan_index]]["enabled"] = enabled
        self.Channels[self.channel_name[chan_index]]["range"] = pico_range
        self.Channels[self.channel_name[chan_index]]["coupling"] = coupling
        self.Channels[self.channel_name[chan_index]]["buffer"] = buffer

    def setup_callback(self):
        """
        Create threads queues and callback metod

        :param chan_to_plot: Channel to plot
        :type chan_to_plot: str
        :raises NotImplementedError: save in hdf5 should be multichannel
        :return: calback function used by picoscope
        :rtype: method

        """
        ##For each case setup process and define get_overview_buffers
        self.nextSample = 0
        self.autoStopOuter = False
        self.wasCalledBack = False
        # listbuff = [chan["buffer"] for chan in self.Channels.values() if chan["enabled"]]
        # def streaming_callback(
        #     handle,
        #     noOfSamples,
        #     startIndex,
        #     overflow,
        #     triggerAt,
        #     triggered,
        #     autoStop,
        #     param,
        # ):
        #     self.wasCalledBack = True
        #     sourceEnd = startIndex + noOfSamples
        #     self.dataqueue.put([buff[startIndex:sourceEnd].copy() for buff in listbuff])
        #     if autoStop:
        #         self.autoStopOuter = True

        if self.nchannels == 2:

            def streaming_callback(
                handle,
                noOfSamples,
                startIndex,
                overflow,
                triggerAt,
                triggered,
                autoStop,
                param,
            ):
                self.wasCalledBack = True
                sourceEnd = startIndex + noOfSamples
                self.dataqueue.put(
                    [
                        self.Channels["A"]["buffer"][startIndex:sourceEnd].copy(),
                        self.Channels["B"]["buffer"][startIndex:sourceEnd].copy(),
                    ]
                )
                if autoStop:
                    self.autoStopOuter = True

        elif self.Channels["A"]["enabled"]:

            def streaming_callback(
                handle,
                noOfSamples,
                startIndex,
                overflow,
                triggerAt,
                triggered,
                autoStop,
                param,
            ):
                self.wasCalledBack = True
                sourceEnd = startIndex + noOfSamples
                self.dataqueue.put(
                    [self.Channels["A"]["buffer"][startIndex:sourceEnd].copy()]
                )
                self.nextSample += noOfSamples
                if autoStop:
                    self.autoStopOuter = True

        elif self.Channels["B"]["enabled"]:

            def streaming_callback(
                handle,
                noOfSamples,
                startIndex,
                overflow,
                triggerAt,
                triggered,
                autoStop,
                param,
            ):
                self.wasCalledBack = True
                sourceEnd = startIndex + noOfSamples
                self.dataqueue.put(
                    self.Channels["B"]["buffer"][startIndex:sourceEnd].copy()
                )
                self.nextSample += noOfSamples
                if autoStop:
                    self.autoStopOuter = True

        return ps4000.StreamingReadyType(streaming_callback)

    def start_measurment(self):
        callback = self.setup_callback()
        now = datetime.now()
        self.M.date = now.strftime("%Y-%m-%d")
        self.M.time = now.strftime("%H:%M:%S")

        # Begin streaming mode:
        sampleUnits = ps4000.PS4000_TIME_UNITS["PS4000_NS"]
        # We are not triggering:
        maxPreTriggerSamples = 0
        autoStopOn = 1
        # No downsampling:
        downsampleRatio = 1
        self.status["runStreaming"] = ps4000.ps4000RunStreaming(
            self.chandle,
            ctypes.byref(self.sampleInterval),
            sampleUnits,
            maxPreTriggerSamples,
            self.totalSamples,
            autoStopOn,
            downsampleRatio,
            self.overview_buffer_size,
        )
        assert_pico_ok(self.status["runStreaming"])

        print(f"Capturing at sample interval {self.sampleInterval.value} ns")
        try:
            while (
                self.nextSample < self.totalSamples
                and not self.autoStopOuter
                and not self.stop_measurement.is_set()
            ):
                self.wasCalledBack = False
                self.status[
                    "getStreamingLastestValues"
                ] = ps4000.ps4000GetStreamingLatestValues(
                    self.chandle, callback, None
                )
                if not self.wasCalledBack:
                    # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
                    # again.
                    time.sleep(self.time_to_fill_half_buffer)
        except KeyboardInterrupt:
            pass
        self.dataqueue.put(None)
        print("Measurment done")
        # Stop the scope
        self.close_unit()

    @staticmethod
    def pico_range(prange):
        return ps4000.PS4000_RANGE["PS4000_" + prange]
