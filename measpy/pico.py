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
from picosdk.functions import adc2mV, assert_pico_ok

from threading import Thread
from queue import Queue, Empty
import time
from scipy.signal import find_peaks
from functools import partial

from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.PicoDeviceEnums import picoEnum

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# plt.style.use('seaborn-v0_8')

maxtimeout = 10

PS2000_channel = {"A": 1, "B": 2, 1: "A", 2: "B"}

class inline_plotting:
    # """
    # Plot data from a Queue.queue
    # init create the plot
    # update_plot update the plot buffer from new data in dataqueues and update the plot if enough data has been received
    # end_plot call update plot until dataqueues contain None
    # ploting_duration estimate time of update_plot execution to check for buffer overflow risk
    # """

    def __init__(self, fs, timeout, updatetime=0.1, plotbuffersize=2000):
        self.timesincelastupdate = 0.0
        self.plotbuffersize = plotbuffersize
        self.updatetime = updatetime
        self.fs = fs
        self.timeinterval = 1/fs
        self.timeout = min(timeout,0.1 * updatetime)
        self.stop = False

        def fstop(event):
            global stop
            self.stop = True

        def tamp_plus(event):
            self.ax[0].set_ylim(np.array(self.ax[0].get_ylim()) / 2)
            plt.pause(0.0001)

        def tamp_moins(event):
            self.ax[0].set_ylim(np.array(self.ax[0].get_ylim()) * 2)
            plt.pause(0.0001)

        def famp_plus(event):
            self.ax[1].set_ylim(np.array(self.ax[1].get_ylim()) / 2)
            plt.pause(0.0001)

        def famp_moins(event):
            self.ax[1].set_ylim(np.array(self.ax[1].get_ylim()) * 2)
            plt.pause(0.0001)

        def on_close(event):
            global stop
            stop = True

        self.x = 1.0 * np.arange(-self.plotbuffersize, 0) * self.timeinterval
        self.plotbuffer = np.zeros_like(self.x)
        self.fig, self.ax = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.subplots_adjust(bottom=0.2)
        self.ax[0].set_xlabel("Temps [s]", fontsize=15)
        self.ax[0].set_ylabel("Tension [V]", fontsize=15)
        self.ax[1].set_xlabel("Fréquence [Hz]", fontsize=15)
        self.ax[1].set_ylabel("Tension [V]", fontsize=15)
        self.ax[1].set_xlim([0.01, self.fs / 2])
        self.ax[1].set_ylim([0, 10])
        (self.linet,) = self.ax[0].plot(self.x, self.plotbuffer)
        (self.linef,) = self.ax[1].plot(
            np.fft.fftfreq(n=self.plotbuffersize, d= self.timeinterval),
            np.fft.fft(self.plotbuffer, norm="ortho"),
        )
        axs = self.fig.add_axes([0.4, 0.01, 0.2, 0.075])
        self.bstop = Button(axs, "Stop")
        self.bstop.on_clicked(fstop)
        atxp = self.fig.add_axes([0.02, 0.6, 0.04, 0.05])
        self.btplus = Button(atxp, "+")
        self.btplus.on_clicked(tamp_plus)
        atxm = self.fig.add_axes([0.02, 0.4, 0.04, 0.05])
        self.btmoins = Button(atxm, "-")
        self.btmoins.on_clicked(tamp_moins)
        afxp = self.fig.add_axes([0.92, 0.6, 0.04, 0.05])
        self.bfplus = Button(afxp, "+")
        self.bfplus.on_clicked(famp_plus)
        afxm = self.fig.add_axes([0.92, 0.4, 0.04, 0.05])
        self.bfmoins = Button(afxm, "-")
        self.bfmoins.on_clicked(famp_moins)
        self.fig.canvas.mpl_connect("close_event", on_close)
        plt.pause(0.0001)

    def _plotting_buffer(self):
        self.x += self.timesincelastupdate
        self.linet.set_ydata(self.plotbuffer)
        self.linet.set_xdata(self.x)
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        self.linef.set_ydata(
            np.abs(
                np.fft.fft(
                    self.plotbuffer * np.hanning(self.plotbuffersize),
                    norm="ortho",
                )
            )
        )
        plt.pause(0.01)
        self.timesincelastupdate = 0.0

    def _update_buffer(self, item):
        n_values = len(item)
        item = np.asarray(item)*0.001
        self.timesincelastupdate += n_values * self.timeinterval
        self.plotbuffer = np.roll(self.plotbuffer, int(-n_values), axis=0)
        self.plotbuffer[-n_values:] = item[-self.plotbuffersize :]

    def update_plot(self, dataqueue, updatetime=None):
        updatetime = self.updatetime if updatetime is None else updatetime
        try:
            if (item := dataqueue.get(timeout=self.timeout)) is not None:
                self._update_buffer(item)
                if self.timesincelastupdate > updatetime:
                    self._plotting_buffer()
        except Empty:
            pass

    def end_plot(self, dataqueue):
        try:
            while (item := dataqueue.get(timeout=maxtimeout)) is not None:
                self._update_buffer(item)
                if self.timesincelastupdate > self.updatetime:
                    self._plotting_buffer()
            self._plotting_buffer()
        except Empty:
            pass
        print("End of datastream")

    @property
    def ploting_duration(self):
        try:
            return self._ploting_duration
        except AttributeError:
            queuetest = Queue()
            queuetest.put(np.random.normal(0, 3, self.plotbuffersize))
            queuetest.put(None)
            start = time.time()
            self.update_plot(queuetest, updatetime=0)
            self._ploting_duration = time.time() - start
            print(f"Plotting duration = {self._ploting_duration}")
            self.x = 1.0 * np.arange(-self.plotbuffersize, 0) * self.timeinterval
            self.timesincelastupdate = 0
            self._plotting_buffer()
            return self._ploting_duration


def findindex(l, e):
    try:
        a = l.index(e)
    except:
        a = None
    return a


def detect_rising_pulses_grad_ind(values, ind0, previous_data_points, **kwargs):
    # """
    # Detect rising pulse with signal.find_peaks on gradient of the signal

    # Parameters
    # ----------
    # values : List
    #     Data.
    # ind0 : int
    #     indice of the first data point.
    # previous_data_points : number
    #     Value of the last data point (indice = ind0-1).
    #     useful to not lose peak if it is at ind0 (work not very well)
    # range_ : int
    #     picoscope range.

    # Returns
    # -------
    # List
    #     List of indices where a rising pulse is detected.

    # """
    try:
        N = len(previous_data_points)
        previous_data_points.extend(values)
        grad = np.diff(previous_data_points)
        lefts, _ = find_peaks(grad, height=np.mean(grad) + np.std(grad))
        # rights, _ = find_peaks(-grad, height=2000)
        return lefts + ind0 - N + 1
    except TypeError:
        grad = np.diff(values)
        lefts, _ = find_peaks(grad, height=np.mean(grad) + np.std(grad))
        # rights, _ = find_peaks(-grad, height=2000)
        return lefts + ind0 + 1
    except AttributeError:
        grad = np.diff([previous_data_points] + values)
        lefts, _ = find_peaks(grad, height=np.mean(grad) + np.std(grad))
        # rights, _ = find_peaks(-grad, height=2000)
        return lefts + ind0


def detect_rising_pulses_threshold_ind(
    values, ind0, previous_data_point, threshold, **kwargs
):
    # """
    # Detect rising pulse using a threshold
    # Parameters
    # ----------
    # values : List
    #     Data.
    # ind0 : int
    #     indice of the first data point.
    # previous_data_points : number
    #     Value of the last data point (indice = ind0-1).
    #     useful to not lose peak if it is at ind0
    # range_ : int
    #     picoscope range.
    # threshold : int
    #     threshold (in adc values).

    # Returns
    # -------
    # List
    #     List of indices where a rising pulse is detected.

    # """
    try:
        V = np.asarray([previous_data_point] + values)
        rising = np.flatnonzero((V[:-1] <= threshold) & (V[1:] > threshold))
    except TypeError:
        V = np.asarray(values)
        rising = np.flatnonzero((V[:-1] <= threshold) & (V[1:] > threshold)) + 1
    return rising + ind0


def rising_pulse_to_raw(M, channel, values):
    for i in range(len(M.in_map)):
        if M.in_map[i] == channel:
            M.in_sig[i].fs = None
            M.in_sig[i].raw = np.double([v / M.fs for v in values])
            M.in_sig[i].desc = "Timming of rising edge pulse"
            M.in_sig[i].unit = "s"


def adc_to_mv(values, range_, bitness=16, **kwargs):
    v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    return [(x * v_ranges[range_]) / (2 ** (bitness - 1) - 1) for x in values]


def mv_to_adc(values, range_, bitness=16):
    v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    return [
        int((x * (2 ** (bitness - 1) - 1)) / v_ranges[range_]) for x in values
    ]


def mv_to_raw(M, channel, values):
    for i in range(len(M.in_map)):
        if M.in_map[i] == channel:
            M.in_sig[i].fs = M.fs
            if M.upsampling_factor > 1:
                M.in_sig[i].raw = decimate(
                    np.double(values) * 0.001, M.upsampling_factor, ftype="fir"
                )[0 : int(round(M.dur * M.fs))]
            else:
                M.in_sig[i].raw = (np.double(values) * 0.001)[
                    0 : int(round(M.dur * M.fs))
                ]


def ps2000_run_measurement(M):
    return _ps2000_run_measurement_threaded(M, adc_to_mv, mv_to_raw, None)


def ps2000_plot(M, plotbuffersize=2000, updatetime=0.1, chan_to_plot="A"):
    plotting = partial(inline_plotting, plotbuffersize=plotbuffersize, updatetime=updatetime)
    return _ps2000_run_measurement_threaded(
        M,
        adc_to_mv,
        mv_to_raw,
        plotting,
        chan_to_plot=chan_to_plot,
    )


def ps2000_pulse_detection(M):
    # """
    # To use threshold detection (more efficient), this function needs M to contain the property
    #  in_threshold : a list of threshold for pulse height (in Volt) for each channel
    # """
    if hasattr(M, "in_threshold") and M.in_threshold is not None:
        detect_rising_pulses = detect_rising_pulses_threshold_ind
        info = ", ".join([f"Threshold = {th} V for chan {PS2000_channel[chan]}" for chan,th in zip(M.in_map,M.in_threshold)])
        print(
            f"Detect pulse with threshold method ({info})"
        )
    else:
        detect_rising_pulses = detect_rising_pulses_grad_ind
        print("Detect pulse with gradient method")
    return _ps2000_run_measurement_threaded(
        M,
        detect_rising_pulses,
        rising_pulse_to_raw,
        None,
        min_chunksize_processed=10000,
    )

def ps2000_hdf5(M,filename):
    return _ps2000_run_measurement_threaded(M,None,None,None,filename=filename)

def _ps2000_run_measurement_threaded(
    M,
    pre_process,
    save,
    plotting,
    chan_to_plot=None,
    min_chunksize_processed=0,
    filename = None
):
    # """
    # This function needs M to contain the following properties:
    #     - in_range : a list of strings specifying the voltage range.
    #         Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
    #         "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
    #     - upsampling_factor : upsampling factor
    #     - in_coupling : Coupling configuration of the channels.
    #         Can be "ac" or "dc"
    # """

    savehdf5 = False
    if filename is not None:
        print(f"Measurment will be save in {filename}, plotting disabled, preprocess and save method ignored")
        savehdf5 = True

    multichannel = False
    max_samples = 100_000
    overview_buffer_size = 20_000

    CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
        None,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_int16,
        ctypes.c_uint32,
        ctypes.c_int16,
        ctypes.c_int16,
        ctypes.c_uint32,
    )

    if M.device_type != "pico":
        raise Exception("Error: device_type must be 'pico'.")

    if type(M.out_sig) != type(None):
        print("Warning: out_sig property ignored with picoscopes")

    # Effective sampling frequency
    # If upsampling_factor is > 1, the actual data acquisition is
    # performed at fs*upsampling_factor, and then decimated to
    # the desired frequency

    effective_fs = M.fs * M.upsampling_factor
    duree_ns = M.dur * 1e9

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments

    si = round(1e9 / effective_fs)
    if effective_fs != (1e9 / si):
        effective_fs = 1e9 / si
        print(
            "Warning : Effective sampling frequency fs changed to nearest possible value of "
            + str(effective_fs)
            + " Hz"
        )

    # print("Effective sampling frequency: " + str(effective_fs))
    sampleInterval = int(si)

    if M.fs != effective_fs / M.upsampling_factor:
        M.fs = effective_fs / M.upsampling_factor
        for s in M.in_sig:
            s.fs = M.fs
        print(
            "Warning : Signal sampling frequency fs changed to nearest possible value of "
            + str(M.fs)
            + " Hz"
        )

    max_loop_time = overview_buffer_size / M.fs
    if plotting is not None:
        if (
            chan_to_plot is not None
            and findindex(M.in_map, PS2000_channel.get(chan_to_plot)) is None
        ):
            print(
                f"The channel to plot {chan_to_plot} is not enabled in measurement setup, plotting canceled"
            )
            plotting = None
        else:
            if callable(plotting):
                plot = plotting(
                    M.fs,
                    timeout=0.1 * max_loop_time,
                )
            else:
                plot = inline_plotting(
                    M.fs,
                    timeout=0.1 * max_loop_time,
                )
            if plot.ploting_duration > 0.85 * max_loop_time:
                print(
                    "Update plot function is too long to execute : "
                    f"{plot.ploting_duration} s, there is a high risk to"
                    f" overflow the buffer (the buffer is full in {max_loop_time} s) "
                )
                plt.close(plot.fig)
                plotting = None
                if input("Cancel measurment? y/n : ") == "y":
                    return

    ## Setup channels
    def setup_channel(chan_index):
        if (ind := findindex(M.in_map, chan_index)) != None:
            enabled = True
            pico_range = ps2000.PS2000_VOLTAGE_RANGE["PS2000_" + M.in_range[ind]]
            if M.in_coupling[ind].capitalize() == "Dc":
                coupling = "PICO_DC"
            elif M.in_coupling[ind].capitalize() == "Ac":
                coupling = "PICO_AC"
            else:
                print(f"Input {PS2000_channel[chan_index]} coupling not recognized, set to 'dc'")
                M.in_coupling[ind] = "dc"
                coupling = "PICO_DC"
            print(
                f"Channel {PS2000_channel[chan_index]}: enabled with range "
                + "PS2000_"
                + M.in_range[ind]
                + " ("
                + str(pico_range)
                + ")"
            )
        else:
            enabled = False
            pico_range = ps2000.PS2000_VOLTAGE_RANGE["PS2000_10V"]
            coupling = "PICO_DC"
            print(f"Channel {PS2000_channel[chan_index]}: disabled")
        return enabled, coupling, pico_range

    enabledA, couplingA, rangeA = setup_channel(1)
    enabledB, couplingB, rangeB = setup_channel(2)

    def setup_save_hdf5(chan_index):
        # """
        # Setup to save directly on hdf5 file

        # Parameters
        # ----------
        # chan_index : int
        #     Index of he channel (1or 2).

        # Returns
        # -------
        # queue : queue.Queue
        #     Queue where data are sent by get_overview_buffers.
        # Process : method
        #     Preprocessing method to be sent in thread.

        # """
        queue = Queue()
        if isinstance(chan_index,list):
            method = M.h5save_data
            Process = Thread(target=method,args=(queue,))
        elif (ind := findindex(M.in_map, chan_index)) != None:
            Sig = M.in_sig[ind]
            method = Sig.h5save_data
            Process = Thread(target=method,args=(queue,))
        return queue, Process

    def setup_preprocess(chan_index, pico_range, chan_plot=False):
        # """
        # Define the method for preprocessing data in thread based on the method in arguments

        # Parameters
        # ----------
        # chan_index : int
        #     Index of the channel (1 or 2).
        # pico_range : int
        #     Picoscope voltage range (mapped from PS2000_VOLTAGE_RANGE).
        # chan_plot : bool, optional
        #     If True the data are send to another queue for plotting. The default is False.

        # Returns
        # -------
        # queue : queue.Queue
        #     Queue where data are sent by get_overview_buffers.
        # queue_plot : queue.Queue
        #     Queue for plotting.
        # Process : method
        #     Preprocessing method to be sent in thread.
        # result : list
        #     Data after preprocessing.

        # """
        if (ind := findindex(M.in_map, chan_index)) != None:
            result = []
            queue = Queue()

            if chan_plot:
                queue_plot = Queue()
            else:
                queue_plot = None

            if (Vthreshold := getattr(M, "in_threshold", None)) is not None:
                adc_threshold = mv_to_adc([Vthreshold[ind] * 0.001], pico_range)[0]
            else:
                adc_threshold = None

            # list pre_process argument, to choose wich method to use
            method_args = getfullargspec(pre_process).args

            #range_ and threshold, optianla argument that do not influence the func method
            kwargs = {"range_": pico_range, "threshold": adc_threshold}
            partial_kwargs = {
                a: b for a, b in kwargs.items() if a in method_args
            }
            if partial_kwargs:
                method = partial(pre_process, **partial_kwargs)
            else:
                method = pre_process

            if "ind0" and "previous_data_point" in method_args:
            #pre_process need the index of the start of each chunk and 
            #the value of the last datapoint (detect_rising_pulses_threshold_ind)
                def func(queue, result):
                    ind0 = 0
                    chunk_data = []
                    previous_data_point = None
                    while (item := queue.get(timeout=maxtimeout)) is not None:
                        chunk_data.extend(item)
                        if (N := len(chunk_data)) > min_chunksize_processed:
                            values = method(chunk_data, ind0, previous_data_point)
                            previous_data_point = chunk_data[-1]
                            result.extend(values)
                            ind0 += N
                            chunk_data = []
                    if len(chunk_data) > 0:
                        values = method(chunk_data, ind0, previous_data_point)
                        previous_data_point = chunk_data[-1]
                        result.extend(values)

            elif not ("ind0" in method_args or "previous_data_point" in method_args):
                if min_chunksize_processed > 0:
                # if preprocessing has to be done in large chuncks
                    if queue_plot is not None:

                        def func(queue, result):
                            chunk_data = []
                            while (item := queue.get(timeout=maxtimeout)) is not None:
                                chunk_data.extend(item)
                                if len(chunk_data) > min_chunksize_processed:
                                    values = method(chunk_data)
                                    result.extend(values)
                                    queue_plot.put(values)
                                    chunk_data = []
                            if len(chunk_data) > 0:
                                values = method(chunk_data)
                                result.extend(values)
                                queue_plot.put(values)
                            queue_plot.put(None)

                    else:
                        def func(queue, result):
                            chunk_data = []
                            while (item := queue.get(timeout=maxtimeout)) is not None:
                                chunk_data.extend(item)
                                if len(chunk_data) > min_chunksize_processed:
                                    values = method(chunk_data)
                                    result.extend(values)
                                    chunk_data = []
                            if len(chunk_data) > 0:
                                values = method(chunk_data)
                                result.extend(values)
                else:
                    if queue_plot is not None:
                        def func(queue, result):
                            while (item := queue.get(timeout=maxtimeout)) is not None:
                                values = method(item)
                                result.extend(values)
                                queue_plot.put(values)
                            queue_plot.put(None)

                    else:
                        def func(queue, result):
                            while (item := queue.get(timeout=maxtimeout)) is not None:
                                values = method(item)
                                result.extend(values)

            else:
                raise NotImplementedError(f"Pre_process arguments ({method_args}) is not supported")

            Process = Thread(target=func,args=(queue,result,))
            return queue, queue_plot, Process, result


    if savehdf5:
        #update dbfs before saving parameters
        for sig, prange in zip(M.in_sig,M.in_range):
            pico_range = ps2000.PS2000_VOLTAGE_RANGE["PS2000_" + prange]
            sig.dbfs = adc_to_mv([1],pico_range)[0] * 0.001

        #Picoscope2000 return 16bits signed integer
        M.datatype = "i2"
        M.to_hdf5(filename)
        if not multichannel:
            if enabledA:
                if enabledB:
                    queueAB, ProcessAB = setup_save_hdf5([1,2])
                    multichannel=True
                else:
                    queueA, ProcessA = setup_save_hdf5(1)
            elif enabledB:
                queueB, ProcessB = setup_save_hdf5(2)
    elif not multichannel:
        if enabledA:
            if chan_to_plot == "A":
                queueA, queue_plot, ProcessA, retA = setup_preprocess(
                    1, rangeA, True
                )
            else:
                queueA, _, ProcessA, retA = setup_preprocess(1, rangeA, False)

        if enabledB:
            if chan_to_plot == "B":
                queueB, queue_plot, ProcessB, retB = setup_preprocess(
                    2, rangeB, True
                )
            else:
                queueB, _, ProcessB, retB = setup_preprocess(2, rangeB, False)
    else:
        raise NotImplementedError #Need change to allow preprocess with multichannel
        # queueAB, queue_plot, ProcessAB, retAB = setup_preprocessAB(
        #     [1,2], [rangeA,rangeB], chan
        # )

    ##get_overview_buffers_factory
    if enabledA:
        if enabledB:
            if multichannel:
                def get_overview_buffers(
                    buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values
                ):
                    queueAB.put([buffers[0][0:n_values],buffers[2][0:n_values]])
            else:

                def get_overview_buffers(
                    buffers,
                    _overflow,
                    _triggered_at,
                    _triggered,
                    _auto_stop,
                    n_values,
                ):
                    queueA.put(buffers[0][0:n_values])
                    queueB.put(buffers[2][0:n_values])

        else:

            def get_overview_buffers(
                buffers,
                _overflow,
                _triggered_at,
                _triggered,
                _auto_stop,
                n_values,
            ):
                queueA.put(buffers[0][0:n_values])

    elif enabledB:

        def get_overview_buffers(
            buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values
        ):
            queueB.put(buffers[2][0:n_values])

    callback = CALLBACK(get_overview_buffers)

    with ps2000.open_unit() as device:
        print("Device info: {}".format(device.info))

        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
            enabledA,
            picoEnum.PICO_COUPLING[couplingA],
            rangeA,
        )
        assert_pico2000_ok(res)
        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL["PICO_CHANNEL_B"],
            enabledB,
            picoEnum.PICO_COUPLING[couplingB],
            rangeB,
        )
        assert_pico2000_ok(res)

        ########## Signal generator ##########
        if M.sig_gen:
            res = ps2000.ps2000_set_sig_gen_built_in(
                device.handle,
                int(M.offset * 1e6),  # offset voltage in uV
                int(M.amp / 2 * 1e6),  # peak-to-peak votage in uV
                M.wave,  # type of waveform (0 = sine wave)
                M.freq_start,  # start frequency in Hz
                M.freq_stop,  # stop frequency in Hz
                M.freq_change,  # frequency change per interval in Hz
                M.freq_int,  # interval of frequency change in seconds
                M.sweep_dir,  # sweep direction (0 = up)
                M.sweep_number,  # number of times to sweep
            )
            assert_pico2000_ok(res)
        #######################################

        res = ps2000.ps2000_run_streaming_ns(
            device.handle,
            sampleInterval,
            2,
            max_samples,
            False,
            1,
            overview_buffer_size,
        )
        assert_pico2000_ok(res)

        now = datetime.now()
        M.date = now.strftime("%Y-%m-%d")
        M.time = now.strftime("%H:%M:%S")

        ##start preprocess threads
        if multichannel:
            ProcessAB.start()
            if not savehdf5:
                ProcessA.start()
                ProcessB.start()
        else:
            if enabledA:
                ProcessA.start()

            if enabledB:
                ProcessB.start()

        print("Start")
        start_time = time.time_ns()

        # loop until wanted duration + maximum time of one loop without buffer overrun
        if plotting is not None:
            margin = max(min(int(max_loop_time*1e9),duree_ns * 0.1), plot.ploting_duration*(1.1))
            while time.time_ns() - start_time < duree_ns + margin and not plot.stop:
                ps2000.ps2000_get_streaming_last_values(device.handle, callback)
                plot.update_plot(queue_plot)

        else:
            margin = min(int(max_loop_time*1e9), duree_ns * 0.1)
            while time.time_ns() - start_time < duree_ns  + margin:
                ps2000.ps2000_get_streaming_last_values(device.handle, callback)

        print("Measurment done")
        print("Waiting for data processing...")
        overrun = False
        ps2000.ps2000_overview_buffer_status(device.handle, overrun)
        if overrun:
            print("Buffer have overrun")

        ##Put end flag in queues
        if multichannel:
            queueAB.put(None)
        else:
            if enabledA:
                queueA.put(None)
            if enabledB:
                queueB.put(None)

        ##Wait for all thread to finish and save data
        if multichannel:
            ProcessAB.join()
            if not savehdf5:
                ProcessA.join()
                save(M, 1, retA)
                ProcessB.join()
                save(M, 2, retB)
        else:
            if enabledA:
                ProcessA.join()
                if not savehdf5:
                    save(M, 1, retA)

            if enabledB:
                ProcessB.join()
                if not savehdf5:
                    save(M, 2, retB)

        print("Preprocess data done")

        ## Continue plotting until last data
        if plotting is not None:
            plot.end_plot(queue_plot)


def ps4000_plot(M,plotbuffersize=2000,updatetime=0.1):
    # """
    # This function needs M to contain the following properties:
    #     - in_range : a list of strings specifying the voltage range.
    #         Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
    #         "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
    #     - upsampling_factor : upsampling factor
    #     - in_coupling : Coupling configuration of the channels.
    #         Can be "ac" or "dc"
    # """

    import time


    # Plotting part

    global plotbuffer, ax, x, line, stop, timesincelastupdate, updtime

    updtime = updatetime

    timesincelastupdate = 0.0

    stop = False

    def fstop(event):
        global stop
        stop = True

    def tamp_plus(event):
        ax[0].set_ylim(np.array(ax[0].get_ylim())/2)
    def tamp_moins(event):
        ax[0].set_ylim(np.array(ax[0].get_ylim())*2)
    def famp_plus(event):
        ax[1].set_ylim(np.array(ax[1].get_ylim())/2)
    def famp_moins(event):
        ax[1].set_ylim(np.array(ax[1].get_ylim())*2)
    def on_close(event):
        global stop
        stop = True

    x = np.arange(plotbuffersize)
    plotbuffer = np.sin(x)
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    fig.subplots_adjust(bottom=0.2)
    ax[0].set_xlabel('Temps [s]',fontsize=15)
    ax[0].set_ylabel('Tension [V]',fontsize=15)
    ax[1].set_xlabel('Fréquence [Hz]',fontsize=15)
    ax[1].set_ylabel('Tension [V]',fontsize=15)
    ax[1].set_xlim([0.01,M.fs/2])
    ax[1].set_ylim([0,10])
    linet, = ax[0].plot(x/M.fs,plotbuffer)
    linef, = ax[1].plot(np.fft.fftfreq(n=plotbuffersize,d=1/M.fs),np.fft.fft(plotbuffer,norm='ortho'))
    axs = fig.add_axes([0.4, 0.01, 0.2, 0.075])
    bstop = Button(axs, 'Stop')
    bstop.on_clicked(fstop)
    atxp = fig.add_axes([0.02, 0.6, 0.04, 0.05])
    btplus = Button(atxp, '+')
    btplus.on_clicked(tamp_plus)
    atxm = fig.add_axes([0.02, 0.4, 0.04, 0.05])
    btmoins = Button(atxm, '-')
    btmoins.on_clicked(tamp_moins)
    afxp = fig.add_axes([0.92, 0.6, 0.04, 0.05])
    bfplus = Button(afxp, '+')
    bfplus.on_clicked(famp_plus)
    afxm = fig.add_axes([0.92, 0.4, 0.04, 0.05])
    bfmoins = Button(afxm, '-')
    bfmoins.on_clicked(famp_moins)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.pause(0.0001)

    # End of plotting part

    # Find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16(32767)

    global nextSample, autoStopOuter, wasCalledBack

    # Buffer size fixed to 20k samples
    sizeOfOneBuffer = 20_000

    # Effective sampling frequency
    # If upsampling is > 1, the actual data acquisition is
    # performed at fs*upsampling, and then decimated to
    # the desired frequency
    effective_fs = M.fs * M.upsampling_factor

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments
    si = round(1e9/effective_fs)
    if effective_fs != (1e9/si):
        effective_fs = (1e9/si)
        print("Warning : Sampling frequency fs changed to nearest possible value of "+str(effective_fs)+" Hz")
    
    print("Effective sampling frequency: "+str(effective_fs))

    numdesiredsamples = int(round(effective_fs*M.dur))
    numBuffersToCapture = int(np.ceil(numdesiredsamples/sizeOfOneBuffer))
    sampleInterval = ctypes.c_int32(si)
    totalSamples = sizeOfOneBuffer * numBuffersToCapture
    chandle = ctypes.c_int16()
    status = {}

    # Open PicoScope 4000 Series device
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps4000.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indA]]
        if M.in_coupling[indA].capitalize()=='Dc':
           couplingA = 1
        elif M.in_coupling[indA].capitalize()=='Ac':
            couplingA = 0
        else:
            print("Input A coupling not recognized, set to 'dc'")
            couplingA = 1
            M.in_coupling[indA]='dc'
        # Create buffers ready for assigning pointers for data collection
        bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel A: enabled with range '+'PS4000_'+M.in_range[indA]+' ('+str(rangeA)+')')
    else:
        enabledA = False
        rangeA = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel A: disabled')

    # Set up channel A
    channel_range = ps4000.PS4000_RANGE['PS4000_50MV']
    status["setChA"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                            int(enabledA),
                                            couplingA,
                                            rangeA)
    assert_pico_ok(status["setChA"])

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indB]]
        # Create buffers ready for assigning pointers for data collection
        bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel B: enabled with range '+'PS4000_'+M.in_range[indB]+' ('+str(rangeB)+')')
        if M.in_coupling[indB].capitalize()=='Dc':
           couplingB = 1
        elif M.in_coupling[indB].capitalize()=='Ac':
            couplingB = 0
        else:
            print("Input B coupling not recognized, set to 'dc'")
            couplingB = 1
            M.in_coupling[indB]='dc'
    else:
        enabledB = False
        couplingB = 1
        rangeB = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel B: disabled')

    # Set up channel B
    status["setChB"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                            int(enabledB),
                                            couplingB,
                                            rangeB)
    assert_pico_ok(status["setChB"])

    # Set data buffer location for data collection from channel A
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_A = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledA:
        status["setDataBuffersA"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                                        bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersA"])

    # Set data buffer location for data collection from channel B
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_B = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledB:
        status["setDataBuffersB"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                                        bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersB"])

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Begin streaming mode:
    sampleUnits = ps4000.PS4000_TIME_UNITS['PS4000_NS']
    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1
    # No downsampling:
    downsampleRatio = 1
    status["runStreaming"] = ps4000.ps4000RunStreaming(chandle,
                                                    ctypes.byref(sampleInterval),
                                                    sampleUnits,
                                                    maxPreTriggerSamples,
                                                    totalSamples,
                                                    autoStopOn,
                                                    downsampleRatio,
                                                    sizeOfOneBuffer)
    assert_pico_ok(status["runStreaming"])

    actualSampleInterval = sampleInterval.value
    actualSampleIntervalNs = actualSampleInterval * 1000

    print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False

    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):

        global plotbuffer, timesincelastupdate, updtime
        global nextSample, autoStopOuter, wasCalledBack
        wasCalledBack = True
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        timesincelastupdate += noOfSamples/M.fs
        plotbuffer=np.roll(plotbuffer,int(-noOfSamples),axis=0)
        plotbuffer[-noOfSamples:] = (np.array(adc2mV(bufferAMax[startIndex:sourceEnd],rangeA,maxADC))/1000)
        if timesincelastupdate > updtime:
            linet.set_ydata(plotbuffer)
            linef.set_ydata(np.abs(np.fft.fft(plotbuffer*np.hanning(plotbuffersize),norm='ortho')))
            plt.pause(0.0001)
            timesincelastupdate = 0.0
        # if enabledA:
        #     bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
        # if enabledB:
        #     bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
        nextSample += noOfSamples
        if autoStop:
            autoStopOuter = True

    # Convert the python function into a C function pointer.
    cFuncPtr = ps4000.StreamingReadyType(streaming_callback)

    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples and not autoStopOuter and not stop:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps4000.ps4000GetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    print("Done.")

    # # Convert ADC counts data to mV
    # if enabledA:
    #     adc2mVChAMax = adc2mV(bufferCompleteA, rangeA, maxADC)
    # if enabledB:
    #     adc2mVChBMax = adc2mV(bufferCompleteB, rangeB, maxADC)

    # # Create time data
    # time = np.linspace(0, (totalSamples - 1) * actualSampleIntervalNs, totalSamples)

    # Stop the scope
    # handle = chandle
    status["stop"] = ps4000.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    # handle = chandle
    status["close"] = ps4000.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    # for i in range(len(M.in_map)):
    #     if M.in_map[i] == 1:
    #         M.data[M.in_name[i]].raw = decimate(np.double(adc2mVChAMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)
    #     elif M.in_map[i] == 2:
    #         M.data[M.in_name[i]].raw = decimate(np.double(adc2mVChBMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)

    # if M.fs!=effective_fs/M.upsampling_factor:
    #     M.fs = effective_fs/M.upsampling_factor
    #     print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
    #     for i in range(len(M.in_map)):
    #         M.data[M.in_name[i]].fs = M.fs
    

def ps4000_run_measurement(M):
    # """
    # This function needs M to contain the following properties:
    #     - in_range : a list of strings specifying the voltage range.
    #         Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
    #         "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
    #     - upsampling_factor : upsampling factor
    #     - in_coupling : Coupling configuration of the channels.
    #         Can be "ac" or "dc"
    # """

    import time
    global nextSample, autoStopOuter, wasCalledBack

    # Buffer size fixed to 20k samples
    sizeOfOneBuffer = 100_000

    # Effective sampling frequency
    # If upsampling is > 1, the actual data acquisition is
    # performed at fs*upsampling, and then decimated to
    # the desired frequency
    effective_fs = M.fs * M.upsampling_factor

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments
    si = round(1e9/effective_fs)
    if effective_fs != (1e9/si):
        effective_fs = (1e9/si)
        print("Warning : Sampling frequency fs changed to nearest possible value of "+str(effective_fs)+" Hz")
    
    print("Effective sampling frequency: "+str(effective_fs))

    numdesiredsamples = int(round(effective_fs*M.dur))
    numBuffersToCapture = int(np.ceil(numdesiredsamples/sizeOfOneBuffer))
    sampleInterval = ctypes.c_int32(si)
    totalSamples = sizeOfOneBuffer * numBuffersToCapture
    chandle = ctypes.c_int16()
    status = {}

    # Open PicoScope 4000 Series device
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps4000.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indA]]
        if M.in_coupling[indA].capitalize()=='Dc':
           couplingA = 1
        elif M.in_coupling[indA].capitalize()=='Ac':
            couplingA = 0
        else:
            print("Input A coupling not recognized, set to 'dc'")
            couplingA = 1
            M.in_coupling[indA]='dc'
        # Create buffers ready for assigning pointers for data collection
        bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel A: enabled with range '+'PS4000_'+M.in_range[indA]+' ('+str(rangeA)+')')
    else:
        enabledA = False
        rangeA = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel A: disabled')

    # Set up channel A
    channel_range = ps4000.PS4000_RANGE['PS4000_50MV']
    status["setChA"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                            int(enabledA),
                                            couplingA,
                                            rangeA)
    assert_pico_ok(status["setChA"])

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indB]]
        # Create buffers ready for assigning pointers for data collection
        bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel B: enabled with range '+'PS4000_'+M.in_range[indB]+' ('+str(rangeB)+')')
        if M.in_coupling[indB].capitalize()=='Dc':
           couplingB = 1
        elif M.in_coupling[indB].capitalize()=='Ac':
            couplingB = 0
        else:
            print("Input B coupling not recognized, set to 'dc'")
            couplingB = 1
            M.in_coupling[indB]='dc'
    else:
        enabledB = False
        rangeB = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel B: disabled')

    # Set up channel B
    status["setChB"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                            int(enabledB),
                                            couplingB,
                                            rangeB)
    assert_pico_ok(status["setChB"])

    # Set data buffer location for data collection from channel A
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_A = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledA:
        status["setDataBuffersA"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                                        bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersA"])

    # Set data buffer location for data collection from channel B
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_B = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledB:
        status["setDataBuffersB"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                                        bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersB"])

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Begin streaming mode:
    sampleUnits = ps4000.PS4000_TIME_UNITS['PS4000_NS']
    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1
    # No downsampling:
    downsampleRatio = 1
    status["runStreaming"] = ps4000.ps4000RunStreaming(chandle,
                                                    ctypes.byref(sampleInterval),
                                                    sampleUnits,
                                                    maxPreTriggerSamples,
                                                    totalSamples,
                                                    autoStopOn,
                                                    downsampleRatio,
                                                    sizeOfOneBuffer)
    assert_pico_ok(status["runStreaming"])

    actualSampleInterval = sampleInterval.value
    actualSampleIntervalNs = actualSampleInterval * 1000

    print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False

    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        global nextSample, autoStopOuter, wasCalledBack
        wasCalledBack = True
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        if enabledA:
            bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
        if enabledB:
            bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
        nextSample += noOfSamples
        if autoStop:
            autoStopOuter = True
            print("auto")

    # if M.sig_gen:
    #     res=ps4000.ps4000SetSigGenBuiltIn(chandle,
    #             int(M.offset*1e6),      # offset voltage in uV
    #             int(M.amp/2*1e6),       # peak-to-peak votage in uV
    #             M.wave,                 # type of waveform (0 = sine wave)
    #             M.freq_start,           # start frequency in Hz
    #             M.freq_stop,            # stop frequency in Hz
    #             M.freq_change,          # frequency change per interval in Hz
    #             M.freq_int,             # interval of frequency change in seconds
    #             M.sweep_dir,            # sweep direction (0 = up)
    #             0,
    #             1,
    #             0,
    #             0,
    #             0,
    #             0
    #         )
        
    #     print("config")
    #     assert_pico_ok(res)
    #     print("fin assert")
    # print("suite")

    # Convert the python function into a C function pointer.
    cFuncPtr = ps4000.StreamingReadyType(streaming_callback)

    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples:  #and not autoStopOuter:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps4000.ps4000GetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    print("Done.")

    # Find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16(32767)

    # Convert ADC counts data to mV
    if enabledA:
        adc2mVChAMax = adc2mV(bufferCompleteA, rangeA, maxADC)
    if enabledB:
        adc2mVChBMax = adc2mV(bufferCompleteB, rangeB, maxADC)

    # Create time data
    time = np.linspace(0, (totalSamples - 1) * actualSampleIntervalNs, totalSamples)

    # Stop the scope
    # handle = chandle
    status["stop"] = ps4000.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    # handle = chandle
    status["close"] = ps4000.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    for i in range(len(M.in_map)):
        if M.in_map[i] == 1:
            M.in_sig[i].raw = decimate(np.double(adc2mVChAMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)
        elif M.in_map[i] == 2:
            M.in_sig[i].raw = decimate(np.double(adc2mVChBMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)

    if M.fs!=effective_fs/M.upsampling_factor:
        M.fs = effective_fs/M.upsampling_factor
        print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
        for i,s in enumerate(M.in_sig):
            s.fs = M.fs
    