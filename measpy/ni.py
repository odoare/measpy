# measpy/ni.py
#
# -------------------------------------------------
# Data acquisition with National Instrument devices
# -------------------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy

from datetime import datetime
from time import sleep

import nidaqmx
from nidaqmx import stream_readers
import nidaqmx.constants as niconst
from threading import Event
import numpy as np
import h5py

from ._tools import siglist_to_array, t_min, _add_N_data, H5file_valid
from .signal import Signal

def _n_to_ain(n):
    return 'ai'+str(n-1)
def _n_to_aon(n):
    return 'ao'+str(n-1)

def ni_run_measurement(M, filename=None, duration="default"):
    """
    Runs a measurement defined in the object of
    the class measpy.measurement.Measurement given
    as argument.

    Run callback at refresh_rate with data acquired

    Once the data acquisition process is terminated,
    the measurement object given in argument contains
    a property in_sig consisting of a list of signals.

    :param M: The measurement object that defines the measurement properties
    :type M: measpy.measurement.Measurement
    :param filename: .h5 filename to direct write on disk, defaults to None
    :type filename: str or Path, optional
    :param duration: optional duration in second, take value in M is default, defaults to "default"
    :type duration: float, optional
    :return: Nothing, the measurement passed as argument is modified in place.

    """
    if filename is not None and H5file_valid(filename):
        M.to_hdf5(filename)

        def callback(buffer_in, n_values):
            _add_N_data(H5file["in_sig"], buffer_in, n_values)

        with h5py.File(filename, "r+") as H5file:
            try:
                n_values, Nchannel = H5file["in_sig"].chunks
            except ValueError:
                n_values = H5file["in_sig"].chunks[0]
            with ni_callback_measurement(M) as NI:
                NI.set_callback(callback, n_values)
                NI.run(duration=duration)
        M.load_h5data()
    else:
        samples = []

        def callback(buffer_in, n_values):
            nonlocal samples
            samples.extend(buffer_in.copy())

        with ni_callback_measurement(M) as NI:
            NI.set_callback(callback, 2**14)
            NI.run(duration=duration)
        if isinstance(M.in_sig, Signal):
            M.in_sig.raw = np.fromiter(
                samples, np.dtype((float, (len(M.in_map),))), count=len(samples)
            ).squeeze()
            M.in_sig.t0 = NI.tmin
        else:
            if len(M.in_map) == 1:
                M.in_sig[0].raw = np.fromiter(
                    samples,
                    np.dtype(float),
                    count=len(samples),
                )
                M.in_sig[0].t0 = NI.tmin
            else:
                data = np.fromiter(
                    samples,
                    np.dtype((float, (len(M.in_map),))),
                    count=len(samples),
                )
                for i, s in enumerate(M.in_sig):
                    s.raw = data[:, i]
                    s.t0 = NI.tmin
    print("done")


class ni_callback_measurement:
    """
    Measurment using a callback function called when specified number of sample is written from the device to the buffer.
    Using nidaqmx.Task.register_every_n_samples_acquired_into_buffer_event
    """

    def __init__(self, M):
        #Parameters setup
        system = nidaqmx.system.System.local()
        self.callback_set = False
        self.Nchannel = len(M.in_map)

        if isinstance(M.in_sig, list) and len(M.in_sig) != self.Nchannel:
            raise ValueError(
                f"in_sig property of measurement must be a multichannel signal or a list of {len(M.in_map)} single channel signals"
            )

        self.in_multichannel = isinstance(M.in_sig, Signal)

        if isinstance(M.out_sig, list) and len(M.out_sig) != len(M.out_map):
            raise ValueError(
                f"out_sig property of measurement must be a multichannel signal or a list of {len(M.out_map)} single channel signals"
            )

        out_multichannel = isinstance(M.out_sig, Signal)

        if M.device_type != "ni":
            print("Warning: deviceType != 'ni'. Changing to 'ni'.")
            M.device_type = "ni"
        if M.in_device == "":
            print(
                "Warning: no output device specified, changing to "
                + system.devices[0].name
            )
            M.in_device = system.devices[0].name

        if hasattr(M, "in_range"):
            inr = M.in_range is not None
        else:
            inr = False
        if not (inr):
            val = nidaqmx.system.device.Device(M.in_device).ai_voltage_rngs[-1]
            print(
                "Warning: no input range specified, changing to the max value of "
                + M.in_device
                + " -> "
                + str(val)
            )
            M.in_range = list(val for b in M.in_map)

        if hasattr(M, "out_sig") and M.out_sig is not None:
            if out_multichannel:
                self.outx = siglist_to_array(M.out_sig.unpack())
            else:
                self.outx = siglist_to_array(M.out_sig)
            self.tmin = t_min(M.out_sig)
            if M.out_device == "":
                print(
                    "Warning: no output device specified, changing to "
                    + system.devices[0].name
                )
                M.out_device = system.devices[0].name
            if hasattr(M, "out_range"):
                if M.out_range is None:
                    outr = False
                else:
                    outr = True
            else:
                outr = False
            if not (outr):
                val = nidaqmx.system.device.Device(M.out_device).ao_voltage_rngs[
                    -1
                ]
                print(
                    "Warning: no output range specified, changing to the max value of "
                    + M.out_device
                    + " -> "
                    + str(val)
                )
                M.out_range = list(val for b in M.out_map)
        else:
            self.tmin = 0
        self.M = M

    def __enter__(self):
        #Open and set up NI Tasks
        try:
            # Set up the read tasks
            if self.M.in_sig is not None:
                self.intask = nidaqmx.Task(new_task_name="in")  # read task
                for i, n in enumerate(self.M.in_map):
                    print(_n_to_ain(n))
                    self.intask.ai_channels.add_ai_voltage_chan(
                        physical_channel=self.M.in_device + "/" + _n_to_ain(n),
                        terminal_config=niconst.TerminalConfiguration.DEFAULT,
                        min_val=-self.M.in_range[i],
                        max_val=self.M.in_range[i],
                        units=niconst.VoltageUnits.VOLTS,
                    )

                self.intask.timing.cfg_samp_clk_timing(
                    rate=self.M.fs,
                    sample_mode=niconst.AcquisitionType.CONTINUOUS,
                )

                for i, iepeval in enumerate(self.M.in_iepe):
                    if iepeval:
                        self.intask.ai_channels[i].ai_excit_val = 0.002
                        self.intask.ai_channels[
                            i
                        ].ai_coupling = niconst.Coupling.AC

            self.reader = stream_readers.AnalogMultiChannelReader(
                self.intask.in_stream
            )

            # Set up the write tasks
            if self.M.out_sig is not None:
                self.nsamps = int(round(self.M.dur * self.M.fs))
                self.outtask = nidaqmx.Task(new_task_name="out")  # write task

                # Set up the write tasks, use the sample clock of the Analog input if possible
                for i, n in enumerate(self.M.out_map):
                    self.outtask.ao_channels.add_ao_voltage_chan(
                        physical_channel=self.M.out_device + "/" + _n_to_aon(n),
                        min_val=-self.M.out_range[i],
                        max_val=self.M.out_range[i],
                        units=niconst.VoltageUnits.VOLTS,
                    )

                if self.M.in_device.startswith("myDAQ"):
                    # If the device is a myDAQ card, we keep most default values
                    # The myDAQ devices are set up separately because
                    # there is no error messages when setting up properties
                    # that are not supported, and the acquisition then fails
                    print("This is a NI my DAQ device")
                    self.outtask.timing.cfg_samp_clk_timing(
                        rate=self.M.fs,
                        sample_mode=niconst.AcquisitionType.CONTINUOUS,
                        samps_per_chan=self.nsamps,
                    )
                else:
                    try:
                        # We first try to use analog input sample clock as output clock
                        self.outtask.timing.cfg_samp_clk_timing(
                            rate=self.M.fs,
                            source="/"
                            + self.M.in_device
                            + "/ai/SampleClock",  # "OnboardClock",
                            sample_mode=niconst.AcquisitionType.CONTINUOUS,
                            samps_per_chan=self.nsamps,
                        )
                        print(
                            "Use of /"
                            + self.M.in_device
                            + "/ai/SampleClock as output clock : success !"
                        )
                    except:
                        # If it fails, use defaults
                        # Then the in/out are not synchronized
                        # There is hence the possibility to use one analog input
                        # to do the in/out sync (io_sync=input channel number)
                        print(
                            'Choosing "'
                            + "/"
                            + self.M.in_device
                            + '/ai/SampleClock" as clock source causes trouble, let\'s try "OnboardClock" '
                        )
                        self.outtask.timing.cfg_samp_clk_timing(
                            rate=self.M.fs,
                            sample_mode=niconst.AcquisitionType.CONTINUOUS,
                            samps_per_chan=self.nsamps,
                        )

                if len(self.M.out_map) == 1:
                    self.outtask.write(self.outx[:, 0], auto_start=False)
                else:
                    # If there are more than one output channel,
                    # the outx.T array argument produces an error.
                    # Temporary dirty fix consists of converting
                    # the array to a list.
                    # TODO: Find better solution
                    self.outtask.write((self.outx.T).tolist(), auto_start=False)
        except Exception as e:
            self.__exit__(type(e), e.args, e)
            raise e
        return self

    def __exit__(self, exc_type, exc_value, tb):
        #Close all tasks
        if self.M.in_sig != None:
            self.intask.close()
        if self.M.out_sig != None:
            self.outtask.close()
        print("ni tasks closed")

    def run(self, stop=Event(), duration="default"):
        """
        Run the measurment
        :param stop: Trigger to stop the measurment, defaults to Event()
        :type stop: threading.Event, optional
        :param duration: Duration of measurment in seconds if default it use the duration in M, defaults to "default"
        :type duration: float, optional
        :return: Nothing

        """
        if not self.callback_set:
            print("Cancelled: there is no callback set")
            return
        self.now = datetime.now()
        self.M.date = self.now.strftime("%Y-%m-%d")
        self.M.time = self.now.strftime("%H:%M:%S")
        if not self.in_multichannel:
            for s in self.M.in_sig:
                s.t0 = self.tmin
        else:
            self.M.in_sig.t0 = self.tmin
        if duration:
            if duration == "default":
                numdesiredsamples = int(round(self.M.fs * self.M.dur))
            else:
                numdesiredsamples = int(round(self.M.fs * duration))
            numBuffersToCapture = int(np.ceil(numdesiredsamples / self.n_values))
        else:
            numBuffersToCapture = float("inf")
        self.buffer_captured = 0
        time_to_fill_buffer = self.n_values / self.M.fs
        if self.M.out_sig != None:
            self.outtask.start()  # Start the write task first, waiting for the analog input sample clock
        if self.M.in_sig != None:
            self.intask.start()
        try:
            print("Start measurement")
            while (
                not stop.is_set() and self.buffer_captured < numBuffersToCapture
            ):
                sleep(time_to_fill_buffer)
        except KeyboardInterrupt:
            pass
        self.stop()

    def stop(self):
        #stop all task
        if self.M.in_sig != None:
            self.intask.stop()
        if self.M.out_sig != None:
            self.outtask.stop()
        print("ni tasks stopped")

    def set_callback(self, callback_method, n_values):
        """
        Create the buffer containing n_values and set the callback that read and use the data using
        the custom 'callback_method', it is called every 'n_values' written into buffer.
        :param callback_method: Method with 2 arguments, the buffer and buffer lenght
        :type callback_method: callable
        :param n_values: Number of datapoint read each call
        :type n_values: int
        :return: Nothing

        """
        self.n_values = n_values
        self.buffer_captured = 0
        if self.Nchannel > 1:
            buffer_in = np.zeros((self.Nchannel, self.n_values), dtype="f8")

            def callback(task_handle, event_type, n_values, callback_data):
                self.reader.read_many_sample(
                    buffer_in, n_values, timeout=niconst.WAIT_INFINITELY
                )
                callback_method(buffer_in.T, n_values)
                self.buffer_captured += 1
                return 0

        else:
            buffer_in = np.zeros((1, self.n_values), dtype="f8")

            def callback(task_handle, event_type, n_values, callback_data):
                self.reader.read_many_sample(
                    buffer_in, n_values, timeout=niconst.WAIT_INFINITELY
                )
                callback_method(buffer_in[0, :], n_values)
                self.buffer_captured += 1
                return 0

        self.intask.register_every_n_samples_acquired_into_buffer_event(
            self.n_values, callback
        )
        self.intask.in_stream.input_buf_size = int(10 * self.n_values)
        self.callback_set = True
        print("callback set")

    def reset_callback(self, callback_method, n_values):
        self.intask.register_every_n_samples_acquired_into_buffer_event(
            self.n_values, None
        )
        self.set_callback(callback_method, n_values)

def ni_run_synced_measurement(M,in_chan=0,out_chan=0):
    """
    Before running a measurement, added_time second of silence
    is added at the begining and end of the selected output channel.
    The measurement is then run, and the time lag between
    a selected acquired signal and the output signal is computed
    from cross-correlation calculation.
    All the acquired signals are then re-synced from the time lag value.

    :param M: The measurement object
    :type M: measpy.measurement.Measurement
    :param out_chan: The selected output channel for synchronization. It is the index of the selected output signal in the list ``M.out_sig``
    :type out_chan: int
    :param in_chan: The selected input channel for synchronization. It is the index of the selected input signal in the list ``M.in_sig``
    :type in_chan: int
    :param added_time: Duration of silence added before and after the selected output signal
    :type added_time: float
    :return: Measured delay between i/o sync channels
    :rtype: float
    """
    M.sync_prepare(out_chan=out_chan)
    ni_run_measurement(M)
    d = M.sync_render(in_chan=in_chan,out_chan=out_chan)
    return d

def ni_get_devices():
    """
    Get the list of NI devices present in the system

    :returns: A list of devices object
    """
    system = nidaqmx.system.System.local()
    print(system.devices)
    return system.devices
