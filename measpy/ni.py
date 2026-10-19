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

from ._tools import siglist_to_array, t_min, _add_N_data, H5file_valid, to_list
from .signal import Signal, SignalType

def _n_to_ain(n):
    return 'ai'+str(n-1)
def _n_to_aon(n):
    return 'ao'+str(n-1)
def _n_to_don(n):
    return 'port'+str(n-1)

def ni_run_measurement(M, filename=None, duration="default", chunck_size = 0):
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
    if filename is not None:
        if H5file_valid(filename):
            # M.to_hdf5(filename)
            M.create_hdf5(filename,chunck_size=chunck_size)

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
                mesu = M._to_dict()
                _ = mesu.pop("in_sig",None)
                _ = mesu.pop("out_sig",None)
                _ = mesu.pop("data_type",None)
                _ = mesu.pop("filename",None)
                for name, value in mesu.items():
                    if value is not None:
                        H5file.attrs[name] = value
            M.load_h5data()
        else:
            if input("Write in file canceled, cancel measurement ? y/n\n") == "y":
                return
    else:
        samples = []

        def callback(buffer_in, n_values):
            nonlocal samples
            samples.extend(buffer_in.copy())

        n_values = min(2**14,int(0.1*M._nsamp))
        with ni_callback_measurement(M) as NI:
            NI.set_callback(callback, n_values)
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

        # The ni specific properties are only set by the Measurement
        # constructor when device_type='ni'. They are given their
        # default value here if the measurement was created otherwise
        # (a wrong device_type is corrected a few lines below)
        M.in_sig_config = getattr(M, 'in_sig_config', 'DEFAULT')
        M.in_iepe = getattr(M, 'in_iepe', None)
        if M.in_iepe is None:
            M.in_iepe = list(False for b in M.in_map)

        if M.in_sig_config not in ['RSE', 'NRSE', 'DIFF', 'PSEUDO_DIFF', 'DEFAULT']:
            print(f"Warning invalid input config name {M.in_sig_config}, input config set to DEFAULT")
            M.in_sig_config = "DEFAULT"

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
                "Warning: no input device specified, changing to "
                + system.devices[0].name
            )
            M.in_device = system.devices[0].name

        self.in_Device = nidaqmx.system.device.Device(M.in_device)
        self_calib(self.in_Device) #check if self_calibration needed
        if hasattr(M, "in_range"):
            inr = M.in_range is not None
        else:
            inr = False
        if not (inr):
            val = self.in_Device.ai_voltage_rngs[-1]
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
            self.out_Device = nidaqmx.system.device.Device(M.out_device)
            self_calib(self.out_Device)
            if hasattr(M, "out_range"):
                if M.out_range is None:
                    outr = False
                else:
                    outr = True
            else:
                outr = False
            if not (outr):
                val = self.out_Device.ao_voltage_rngs[
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
                        terminal_config=niconst.TerminalConfiguration[self.M.in_sig_config],
                        min_val=-self.M.in_range[i],
                        max_val=self.M.in_range[i],
                        units=niconst.VoltageUnits.VOLTS,
                    )


                for i, iepeval in enumerate(self.M.in_iepe):
                    if iepeval:
                        self.intask.ai_channels[i].ai_excit_val = 0.002
                        self.intask.ai_channels[
                            i
                        ].ai_coupling = niconst.Coupling.AC

                #update measurment with actual values
                self.M.in_range = self.ai_channels_range
                self.M.ni_in_sig_bit_resolution = self.ai_channels_bits_resolution
                self.M.in_adc_temp = self.in_Device.cal_dev_temp #current temperature of the input card


            self.reader = stream_readers.AnalogMultiChannelReader(
                self.intask.in_stream
            )

            # Set up the write tasks
            if self.M.out_sig is not None:
                self.nsamps = self.M._nsamp
                self.outtask = nidaqmx.Task(new_task_name="out")  # write task

                if self.M.out_sig[0].type == SignalType.ANALOG:
                    # Set up the write tasks, use the sample clock of the Analog input if possible
                    for i, n in enumerate(self.M.out_map):
                        self.outtask.ao_channels.add_ao_voltage_chan(
                            physical_channel=self.M.out_device + "/" + _n_to_aon(n),
                            min_val=-self.M.out_range[i],
                            max_val=self.M.out_range[i],
                            units=niconst.VoltageUnits.VOLTS,
                        )
                elif self.M.out_sig[0].type == SignalType.DIGITAL:
                    for i, n in enumerate(self.M.out_map):
                        self.outtask.do_channels.add_do_chan(
                            lines=self.M.out_device + "/" + _n_to_don(n),
                            line_grouping=niconst.LineGrouping.CHAN_FOR_ALL_LINES
                        )
                #update measurment with actual values
                self.M.out_range = self.ao_channel_range
                self.M.ni_out_sig_bit_resolution = self.ao_channels_bits_resolution
                self.M.out_adc_temp = self.out_Device.cal_dev_temp #current temperature of the output card
            #set_up timing
            self.timing()
            #Set up frequency measurement task
            if hasattr(self.M, "freq_device"):
                self.f_task = nidaqmx.Task(new_task_name="fs measurement")
                device = nidaqmx.system.device.Device(self.M.freq_device)
                self.freq_divisor = max(4,int((self.M.fs/device.ci_max_timebase)*2e6)) #resolution ~.5ppm
                fs_meas = self.f_task.ci_channels.add_ci_freq_chan(
                    self.M.freq_device + "/ctr1",
                    min_val=self.M.fs*(1-.1),
                    max_val=self.M.fs*(1+.1),
                    meas_method=niconst.CounterFrequencyMethod.LARGE_RANGE_2_COUNTERS,
                    divisor=self.freq_divisor,
                    edge=niconst.Edge.RISING,
                )
                self.timebase = fs_meas.ci_ctr_timebase_rate
                if self.timebase!=device.ci_max_timebase:
                    raise ValueError("Fastest timebase not available to measure frequency")
                fs_meas.ci_freq_term = "/" + self.M.freq_device +"/"+ self.M.freq_map

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
        if hasattr(self.M, "freq_device"):
            self.f_task.close()
        print("ni tasks closed")

    def timing(self):
        if self.M.in_sig is not None and self.M.out_sig is not None and self.M.in_device !=self.M.out_device:
            if hasattr(self.M, "ref_device") and hasattr(self.M, "clock_terminal"):
                if self.M.ref_device == "in":
                    self.intask.timing.cfg_samp_clk_timing(
                        rate=self.M.fs,
                        sample_mode=niconst.AcquisitionType.CONTINUOUS,
                    )
                    self.check_fs(self.intask.timing)
                    self.source_terminal = "/" + self.M.in_device+ "/ai/SampleClock"
                    self.destination_terminal = "/" + self.M.in_device+ "/"+self.M.clock_terminal[0]
                    nidaqmx.system.system.System().connect_terms(
                        source_terminal=self.source_terminal,
                        destination_terminal=self.destination_terminal,
                    )
                    self.outtask.timing.cfg_samp_clk_timing(
                            rate = self.M.fs,
                            source = "/" + self.M.out_device+ "/"+self.M.clock_terminal[1],
                            sample_mode=niconst.AcquisitionType.CONTINUOUS,
                            samps_per_chan=self.nsamps,
                        )
                elif self.M.ref_device == "out":
                    self.outtask.timing.cfg_samp_clk_timing(
                        rate=self.M.fs,
                        sample_mode=niconst.AcquisitionType.CONTINUOUS,
                    )
                    self.check_fs(self.outtask.timing)
                    self.source_terminal = "/" + self.M.out_device+ "/ao/SampleClock"
                    self.destination_terminal = "/" + self.M.out_device+ "/"+self.M.clock_terminal[0]
                    nidaqmx.system.system.System().connect_terms(
                        source_terminal=self.source_terminal,
                        destination_terminal=self.destination_terminal,
                    )
                    self.intask.timing.cfg_samp_clk_timing(
                            rate = self.M.fs,
                            source = "/" + self.M.in_device+ "/"+self.M.clock_terminal[1],
                            sample_mode=niconst.AcquisitionType.CONTINUOUS,
                            samps_per_chan=self.nsamps,
                        )
                #Delay needed for synchronisation (hardware delay)
                self.intask.timing.delay_from_samp_clk_delay_units = (
                       niconst.DigitalWidthUnits.SECONDS
                )
                if not hasattr(self.M, "intask_delay"):
                    self.M.intask_delay=10e-6
                self.intask.timing.delay_from_samp_clk_delay =  self.M.intask_delay
                return
            else:
                print("Warning : Input and output will be based on separate clock sources")
        if self.M.in_sig is not None:
            self.intask.timing.cfg_samp_clk_timing(
                rate=self.M.fs,
                sample_mode=niconst.AcquisitionType.CONTINUOUS,
            )

            self.check_fs(self.intask.timing)
        if self.M.out_sig is not None:
            if self.M.in_device.startswith("myDAQ") and self.M.out_sig.type == SignalType.ANALOG:
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
                    #Success even if 2 different device without conexion
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


    def dicsconnect_clock(self):
        try:
            nidaqmx.system.system.System().disconnect_terms(
                source_terminal=self.source_terminal,
                destination_terminal=self.destination_terminal,
            )
        except AttributeError:
            pass

    def check_fs(self, timing):
        if self.M.fs!= timing.samp_clk_rate:
            print(f"Warning : sampling frequency changed from {self.M.fs} "
                  f"to {timing.samp_clk_rate} Hz")
            self.M.fs = timing.samp_clk_rate
        self.M.fc = timing.samp_clk_timebase_rate # internal clock rate
        self.M.n_pulse = timing.samp_clk_timebase_div  #fc/fs is integer


    def run(self, stop=Event(), pause=Event(), duration="default"):
        """
        Run the measurment
        :param stop: Trigger to stop the measurment, defaults to Event()
        :type stop: threading.Event, optional
        :param pause: Trigger to pause the measurment, and restart is when this trigger is cleared, defaults to Event()
        :type pause: threading.Event, optional
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
        if self.M.out_sig != None:
            print("values: ",self.outx[:, 0])
            if len(self.M.out_map) == 1:
                if self.M.out_sig[0].type == SignalType.DIGITAL:
                    self.outtask.write(list(map(int,map(round,self.outx[:, 0]))), auto_start=False)
                else:
                    self.outtask.write(self.outx[:, 0], auto_start=False)
            else:
                # If there are more than one output channel,
                # the outx.T array argument produces an error.
                # Temporary dirty fix consists of converting
                # the array to a list.
                # TODO: Find better solution
                self.outtask.write((self.outx.T).tolist(), auto_start=False)

        if not self.in_multichannel:
            for s in self.M.in_sig:
                s.t0 = self.tmin
        else:
            self.M.in_sig.t0 = self.tmin
        if duration:
            if duration == "default":
                numdesiredsamples = self.M._nsamp
            else:
                numdesiredsamples = int(round(self.M.fs * duration))
            numBuffersToCapture = int(np.ceil(numdesiredsamples / self.n_values))
        else:
            numBuffersToCapture = float("inf")
        self.buffer_captured = 0
        time_to_fill_buffer = self.n_values / self.M.fs
        if hasattr(self.M, "ref_device"):
            if self.M.ref_device=="out":
                self.intask.start()
                self.outtask.start()
            else:
                self.outtask.start()
                self.intask.start()
        else:
            if self.M.out_sig != None:
                self.outtask.start()  # Start the write task first, waiting for the analog input sample clock
            if self.M.in_sig != None:
                self.intask.start()

        #frequency measurement task
        if hasattr(self.M, "freq_device"):
            self.f_task.start()
            data =self.f_task.read(number_of_samples_per_channel=self.M.freq_nsamp)
            self.f_task.stop()
            result = np.mean(data)
            if self.M.freq_nsamp<10:
                t = [1.84,1.32,1.20,1.14,1.11,1.09,1.08,1.07,1.06][self.M.freq_nsamp-2]
            else:
                t=1
            err_measure = np.std(data)*t/np.sqrt(self.M.freq_nsamp)
            err_resolution = (result*result/(self.freq_divisor*self.timebase-result))/(2*np.sqrt(3))
            if err_measure>0:
                err_resolution/=2    #we know realfreq is in the interval data1,data2
            incertitude = 2*np.sqrt(err_measure**2+err_resolution**2)
            self.M.freq_result = [result, incertitude]
        try:
            print("Start measurement")
            while (
                not (stop.is_set() or pause.is_set()) and self.buffer_captured < numBuffersToCapture
            ):
                sleep(time_to_fill_buffer)
        except KeyboardInterrupt:
            print("Measurement interupted by ctrl-c")
            pass
        if pause.is_set():
            self.pause(pause,stop,duration)
        else:
            self.stop()
            print("measurment done")
            self.dicsconnect_clock()

    def stop(self):
        #stop all task
        if self.M.in_sig != None:
            self.intask.stop()
        if self.M.out_sig != None:
            self.outtask.stop()

    def pause(self, pause_event,stop_event,duration):
        print("Pause measurement...")
        self.stop()
        try:
            while pause_event.is_set():
                sleep(0.1)
            self.run(stop_event,pause_event,duration)
        except KeyboardInterrupt:
            pass

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


    @property
    def ai_channels_range(self):
        """
        Get range of input channels in volt
        """
        return [(A.ai_rng_low,A.ai_rng_high) for A in self.intask.ai_channels]


    @property
    def ao_channel_range(self):
        """
        Get range of output channels in volt
        """
        return [(A.ao_dac_rng_low,A.ao_dac_rng_high) for A in self.outtask.ao_channels]


    @property
    def ai_channels_bits_resolution(self):
        """
        Get resolution of input channels in bits

        """
        return [A.ai_resolution for A in self.intask.ai_channels]

    @property
    def ao_channels_bits_resolution(self):
        """
        Get resolution of output channels in bits

        """
        return [A.ao_resolution for A in self.outtask.ao_channels]


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

def ni_get_true_fs(fs,device_name=""):
    system = nidaqmx.system.System.local()
    if device_name == "":
        print(
            "Warning: no device specified, changing to "
            + system.devices[0].name
        )
        device_name = system.devices[0].name
    intask = nidaqmx.Task()
    intask.ai_channels.add_ai_voltage_chan(device_name + "/" + _n_to_ain(1))
    intask.timing.cfg_samp_clk_timing(
        rate=fs,
        sample_mode=niconst.AcquisitionType.CONTINUOUS,
    )
    if fs!= intask.timing.samp_clk_rate:
        print(f"Warning : sampling frequency changed from {fs} "
              f"to {intask.timing.samp_clk_rate} Hz")
        fs = intask.timing.samp_clk_rate
    else:
        print("No change to fs needed")
    intask.close()
    return fs

def self_calib(device):
    if device.self_cal_supported:
        if np.abs(device.self_cal_last_temp-device.cal_dev_temp)>1:
            print(f"Warning : Current temperature of {device.name} = {round(device.cal_dev_temp,1)}°C\n",
                  f"Last self calibration temperature = {round(device.self_cal_last_temp,1)}°C\n",
                  "a self calibration may be needed")
            #       "launching self calibration...")
            # self.in_Device.self_cal()
            # print("done")

def ni_get_devices():
    """
    Get the list of NI devices present in the system

    :returns: A list of devices object
    """
    system = nidaqmx.system.System.local()
    print(system.devices)
    return system.devices
