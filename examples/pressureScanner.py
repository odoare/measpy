import measpy as mp
from measpy._tools import _add_N_data
from measpy.signal import Signal, SignalType
from measpy.measurement import Measurement
from measpy.ni import ni_run_measurement, ni_callback_measurement
from measpy._plot_tools import plot_data_from_queue
import numpy as np
from queue import Queue
from threading import Thread, Event
import h5py
from copy import deepcopy
from time import sleep,time
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.transforms import Bbox
import math

logger = logging.getLogger(__name__)

def dispatch(q_in : Queue, qs_out : list[Queue], stop_event : Event, timeout : float|int):
    print("start dispatch")
    while not stop_event.is_set():
        if (item := q_in.get(timeout=timeout)) is not None:
            for idx,q_out in enumerate(qs_out):
                q_out.put(item)
        else:
            break
        sleep(0.1)
    for q_out in qs_out:
        q_out.put(None)

def scaleAddresses(addr: list, fb: int, fs: int)-> tuple[list,int]:
    """
    Add/duplicate addresses from addr list to match to difference frequency of fs and fb.

    :param addr: list of addresses, digital values
    :type addr: list
    :param fb: frequency of expected return measurement
    :type fb: int
    :param fs: frequency of the data acquisition on the card
    :type fs: int
    :return: scaled address list and new frequency fs to send to the card
    :rtype: tuple (addr,fs)
    """
    addr = deepcopy(addr)
    nbAdr = len(addr)
    nbIt = int(round(fs/(nbAdr*fb)))
    for it in range(nbAdr):
        addr[it*nbIt:it*nbIt+1] *= nbIt
    # return addr,fs/(nbAdr*nbIt)
    nfs = (nbAdr*nbIt*fb)
    if nfs != fs:
        logger.warning(f"fs has been adjusted from {fs} Hz to {nfs} Hz")
    return addr,nfs

def mapValuesByAdr(arrValues: np.ndarray, arrAdr: np.ndarray, rate: float, calc: str = "mean")->dict[str:np.ndarray]:
    """
    :param arrValues: values acquired
    :type arrValues: numpy.ndarray
    :param arrAdr: digital values of addresses sent
    :type arrAdr: numpy.ndarray
    :param rate: rate of values to keep per address, must be between 0 and 1
    :type rate: float
    :param calc: calculation type to get a single value from multiple per address. Can be "mean" or "median". "mean" by default.
    :type calc: str
    :return: mapped values with their address as key, one value per time.
    :rtype: dict[str:numpy.ndarray]
    """
    if rate<0 or rate>1:
        raise ValueError(f"rate should be between 0 and 1. currently : {rate}")

    mappedValues = {}
    nb = arrAdr.size/np.unique(arrAdr).size
    diff = int(round((1-rate)*nb/2))

    cleanValues = np.array(arrValues)
    if cleanValues.size%arrAdr.size != 0:
        logger.warning(f"Size of values is not perfect for their addresses. Last {-(len(cleanValues)%len(arrAdr))} values will not be processed. If used for continous measure, it might become out of sync.")
        cleanValues = cleanValues[:-(len(cleanValues)%len(arrAdr))]

    # split first by time / scan and then by channel
    scan_splitted = np.stack(np.split(cleanValues, cleanValues.size / arrAdr.size))
    cleanValues = np.stack(np.hsplit(scan_splitted, scan_splitted.shape[1]/nb))

    if diff != 0:
        cleanValues = cleanValues[:,:,diff:-diff]

    if      calc == "mean"   :
        cleanValues = np.mean(cleanValues,axis=2)
    elif    calc == "median"      :
        cleanValues = np.median(cleanValues,axis=2)
    
    for i,adr in enumerate(np.unique(arrAdr)):
        mappedValues[adr] = cleanValues[i]

    return cleanValues

# Define class with custom plot configuration.
class direct_plotting(plot_data_from_queue):
    def plot_setup(self):
        # define x_data : list of numpy array : x axis of the plot
        self.x_data = [ np.arange(0, self.plotbuffersize) * 1.0 ] * self.nchannel
        # define plotbuffer, list of numpy array : y axis of the plot
        self.plotbuffer = [ np.zeros_like(x) for x in self.x_data ]
        self.fig, axe = plt.subplots(1,1,figsize=(10,5))
        self.axes = [ axe ] * self.nchannel
        self.fig.subplots_adjust(bottom=0.2)
        # axes label
        self.axes[0].set_xlabel("Temps [s]", fontsize=15)
        self.axes[0].set_ylabel("Pressure [V]", fontsize=15)
        # axes limits
        self.axes[0].set_xlim([0, self.plotbuffersize])
        self.axes[0].set_ylim([-1,3])
        # plot the buffer
        linet = [ self.axes[chan].plot(self.x_data[chan], self.plotbuffer[chan], animated=True, label=chan)[0] for chan in range(self.nchannel) ]
        # print(f"BIGPACK :\n{linet}")
        plt.show(block=False)

        # Scrollable legends
        legend = plt.legend(bbox_to_anchor=(1.1,1))
        # pixels to scroll per mousewheel event
        d = {"down" : 30, "up" : -30}
        def func(evt):
            if legend.contains(evt):
                bbox = legend.get_bbox_to_anchor()
                bbox = Bbox.from_bounds(bbox.x0, bbox.y0+d[evt.button], bbox.width, bbox.height)
                tr = legend.axes.transAxes.inverted()
                legend.set_bbox_to_anchor(bbox.transformed(tr))
                self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect("scroll_event", func)

        # define lines : list of plt line updated with the buffer
        self.lines = linet
        # displace nans to the right for first plot
        self.istimedata = [ False ] * self.nchannel
        # stop button
        self.stop_event = Event()
        def fstop(event):
            print("button stop pressed")
            self.stop_event.set()
        axs = self.fig.add_axes([0.03,0.01,0.2,0.075])
        self.bstop = Button(axs, "Stop")
        self.bstop.on_clicked(fstop)

        # Define buttons to update a flag used to launch a method updating axis
        self.tamp_plus = False
        def tamp_plus(event):
            self.tamp_plus = True
        atxp = self.fig.add_axes([0.75, 0.08, 0.04, 0.05])
        self.btnPlus = Button(atxp, "+")
        self.btnPlus.on_clicked(tamp_plus)

        self.tamp_moins = False
        def tamp_moins(event):
            self.tamp_moins = True
        atxm = self.fig.add_axes([0.75, 0.01, 0.04, 0.05])
        self.btnMoins = Button(atxm, "-")
        self.btnMoins.on_clicked(tamp_moins)

        self.tamp_up = False
        def tamp_up(event):
            self.tamp_up = True
        atxu = self.fig.add_axes([0.8, 0.08, 0.1, 0.05])
        self.btnUp = Button(atxu, "UP")
        self.btnUp.on_clicked(tamp_up)

        self.tamp_down = False
        def tamp_down(event):
            self.tamp_down = True
        atxd = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])
        self.btnDown = Button(atxd, "DOWN")
        self.btnDown.on_clicked(tamp_down)

        def hide_channel(expr):
            try:
                if int(expr) >= 0 and int(expr) < len(self.lines):
                    logger.info(f"plot change visibility of line {expr} from {self.lines[int(expr)].get_visible()} to {not self.lines[int(expr)].get_visible()}")
                    self.lines[int(expr)].set_visible(not self.lines[int(expr)].get_visible())
                else:
                    logger.warning(f"plot cannot change visibility of line {expr}, it is out of range (0:{len(self.lines)-1})")
            except ValueError as e:
                logger.warning(f"plot cannot change visibility of specified line {expr}, should be an integer")
        atxht = self.fig.add_axes([0.5, 0.01, 0.1, 0.05])
        self.txbHide = TextBox(atxht, "Hide channel : ")
        self.txbHide.on_submit(hide_channel)

        self.fig.canvas.mpl_connect("close_event", fstop)

    def rescaling(self):
        # defines method that rescale axis when a flag is set to True
        # the other flag : 'self.bm.change_axe = True' is needed because changing axis
        # is impossible with fast plot method, the axis are changed using slower plot method

        if self.tamp_plus:
            logger.info("plot zoom")
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 0.5)
            self.tamp_plus = False
            self.bm.changed_axe = True

        if self.tamp_moins:
            logger.info("plot unzoom")
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 2)
            self.tamp_moins = False
            self.bm.changed_axe = True

        if self.tamp_up:
            logger.info("plot looking up")
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) + (np.diff(np.array(self.axes[0].get_ylim())) * 0.25))
            self.tamp_up = False
            self.bm.changed_axe = True

        if self.tamp_down:
            logger.info("plot looking down")
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) - (np.diff(np.array(self.axes[0].get_ylim())) * 0.25))
            self.tamp_down = False
            self.bm.changed_axe = True


    def data_process(self):
        # Update x_data and plotbuffer, using data_buffer that is updated by the queue
        # print(f"CURRENT X_DATA :\n{self.x_data}")
        # print(f"X_DATA TO COME :\n{round(self.timesincelastupdate * self.timeinterval)}")
        # self.x_data[0] += self.timesincelastupdate * self.timeinterval
        for x_data in self.x_data:
            x_data += self.timesincelastupdate * self.timeinterval

        plotbufferToCome = []

        data_buffer = self.data_buffer.transpose()
        for chan in range(self.nchannel):
            plotbufferToCome.append(
                np.array(np.concatenate([
                    self.plotbuffer[chan][self.timesincelastupdate :],
                    data_buffer[chan][-self.timesincelastupdate :].copy()
                ]))
            )

        # plotbufferToCome = np.array([plotbufferToCome])
        # print(f"Plot Buffer to come ({len(plotbufferToCome)}) :\n{plotbufferToCome}")
        # print(type(plotbufferToCome))
        self.plotbuffer = plotbufferToCome


class PressureScanner:
    """
    Include different methods to measure while sending a digital output signal. 
    Possibility to map the measured signal with the sent addresses in a multichannel signal.
    """
    def __init__(self, addresses: list, fs: int, fb: int, rate: float, **kwargs):
        """
        Include different methods to measure while sending a digital output signal. 
        Possibility to map the measured signal with the sent addresses in a multichannel signal.

        :param addresses: List of addresses as digital values meant to be sent to the card.
        :type addresses: list
        :param fs: Frequency to set on the card to acquire data by a measure and iterate addresses. from 10Hz to 1MHz.
        :type fs: int
        :param fb: Frequency of the measured signal expected after transformations and mapping. Should be below fs, the further the better.
        :type fb: int
        :param rate: rate of values to keep while doing the mapping with the addresses. Should be between 0 and 1.
        :type rate: float
        """
        self.stop_event = Event()
        self.M = None
        self.rate = rate

        self.raw_adrs = addresses
        addresses,self.fs = scaleAddresses(addr=addresses,fb=fb,fs=fs)
        if self.fs != fs:
            logger.warning(f"fs (frequency of the ni card) has changed from {fs} Hz to {self.fs} Hz to match the specified addresses and fb (frequency of the result) values")
        self.fb = fb

        self.rawQ = Queue()
        self.mappedQ = Queue()
        self.resultQs = []

        self.n_values = None
        self.n_raw_value = None
        self.n_clean_value = None

        self.adr_sig = Signal(fs=self.fs, raw=np.array(addresses), type=SignalType.DIGITAL)

    def createMeasurement(self, adr_port: int = None, out_device: str = None, **kwargs):
        """
        Create the measurement object. Device type is set as ni, adding an out_sig with the addresses given when create object.
        Adding adr_port to the out_map. All other measurement parameters can be given in the **kwargs.
        
        :param adr_port: port of the address signal to be defined in the measurement out_map
        :type adr_port: int
        :param out_device: measurement out_device
        :type out_device: str
        :param **kwargs: all other arguements of the measurement
        """
        # Use self.adr_sig instead of the method parameter
        _out_sig = [self.adr_sig]
        _out_map = [adr_port]
        if "out_sig" in kwargs.keys():
            _out_sig.extend(kwargs["out_sig"])
        if "out_map" in kwargs.keys():
            _out_map.extend(kwargs["out_map"])

        self.M = mp.Measurement(
            device_type = "ni",  
            out_sig     = _out_sig,
            out_map     = _out_map,
            out_device  = out_device,
            **kwargs
        )

    def runMeasurement(self, **kwargs)->Signal:
        """
        Launch a simple measurement and return the mapped signal with the addresses.
        Meant to be run in a thread.
        """
        times = []
        times.append(time())
        
        print("start measurement")
        ni_run_measurement(self.M, **kwargs)
        print("measurement finished")

        times.append(time())

        print("start matching values with their addresses")
        self.result = mapValuesByAdr(self.M.in_sig[0], self.adr_sig, self.rate)
        print("finisehd matching values with their addresses")

        times.append(time())

        logger.debug(f"Total time of the measurement : {times[len(times)-1]-times[0]} s")
        for i,t in enumerate(times):
            logger.debug(f"step {i+1} took {times[i+1]-t} s")

        return self.result

    def runCbMeasurement(self, n_values=None, **kwargs):
        """
        Meant to run in a separated thread.

        Run a measurement and put the values in rawQ every n_raw_value.

        :param n_value: number of values to put in rawQ each time.
        :type n_value: int
        :param **kwargs: parameters to put in ni_callback_measurement function.
        """
        if self.n_raw_value is None:
            if n_values is not None:
                self.n_raw_value = n_values
        elif n_values is not None:
            logger.warning(f"n_raw_values is already set as {self.n_raw_value}. Assignation as {n_values} has been aborted")

        def callback(buffer_in : np.ndarray, n_values : int):
            self.rawQ.put(buffer_in.copy())

        with ni_callback_measurement(self.M,**kwargs) as NI:
            NI.set_callback(callback,self.n_raw_value)

            def work(*args):
                NI.run(*args)
                logger.info("measurement done")
                self.rawQ.put(None)
            Tmeas = Thread(target=work, args=(self.stop_event, None))
            Tmeas.start()

            try:
                Tmeas.join()
            except Exception as e:
                self.stop_event.set()
                raise e

    # Create method that will read rawQ and write in mappedQ
    def runMapData(self, timeout=10):
        """
        Meant to run in a separated thread.

        Map raw data (in rawQ) measured with their addresses and put them in mappedQ.

        :param timeout: timeout to read measured data in rawQ.
        :type timeout: float | int
        """
        try:
            sleep(2)
            while (not self.stop_event.is_set() and (raw_data := self.rawQ.get(timeout=timeout)) is not None):
                # raw_data = self.rawQ.get(timeout=timeout)
                logger.debug(f"reading : {raw_data}")
                # clean_data = np.array(list(mapValuesByAdr(raw_data,self.adr_sig.values,self.rate).values()))
                clean_data = mapValuesByAdr(raw_data, self.adr_sig.values, self.rate)

                logger.debug(f"putting : {clean_data}")
                logger.debug(f"putting type : {type(clean_data)} [{type(clean_data[0])}]")

                self.mappedQ.put(clean_data.transpose())
            self.mappedQ.put(None)
        except Exception as e:
            self.stop()
            raise e

    def runAutoSaveH5(self, filepath: str):
        """
        Meant to run in a separated.

        :param filepath: absolute path of the hdf5 file you want to create to store the measured mapped data.
        :type filepath: str
        """
        try:
            rQ = Queue()
            self.resultQs.append(rQ)

            sleep(2)
            self.n_clean_value = int((self.n_raw_value * self.rate * self.fb) / self.fs)

            sigList = []
            for adr in np.unique(self.adr_sig.values):
                sigList.append(
                    Signal(
                        fs=self.fb,
                        desc=adr,
                        # unit= "V"
                        # cal=sigValues.cal
                    )
                )
            resultSignal : Signal = Signal.pack(sigList)

            resultSignal.create_hdf5dataset(filepath, chunck_size=self.n_clean_value)

            print(f"Starting saving data in {filepath}/in_sigs")
            with h5py.File(filepath, "r+") as H5file:
                item = np.array(rQ.get())
                #Get dimension of item for multichannel case
                dims = item.shape
                Npoints = dims[0]
                dataset = H5file["in_sigs"]
                #Get the chunksize and datatype of the dataset
                chunksize = dataset.chunks[0]
                datatype = dataset.dtype
                #Define a buffer with chuncksize and datatype
                while (item := rQ.get(timeout=5)) is not None:
                    item = np.array(item)
                    print("ITEM SIZD: ",item.shape, " - DATASET SIZD: ",dataset.shape, " - N_CLEAN: ",self.n_clean_value)
                    _add_N_data(dataset, item, self.n_clean_value)
        except Exception as e:
            self.stop()
            raise e

    def runDirectPlot(self, refresh_delay: float = None, plotbuffersize: int = None):
        """
        Meant to run in main thread !

        Display a plot of the current measurement with a line for each channel.  
        The stop button stop the measurement acquisition. You need to close the app after hitting the stop button.  
        The textbox allow you to enter the line number to change it's visibility.  
        The + and - buttons zoom and unzoom on the center of the plot.  
        The Up and Down buttons translate the displayed plot vertically.

        :param refresh_delay: Time (second) between each update of the plot, defaults to 0.1
        :type refresh_delay: float | int, optional
        :param plotbuffersize: define attribute to store a size of the plot buffer that can be used inside 'plot_setup' method, defaults to 2000
        :type plotbuffersize: int, optional
        """
        try:
            rQ = Queue()
            self.resultQs.append(rQ)
            
            A = direct_plotting(
                fb, updatetime=refresh_delay, plotbuffersize=plotbuffersize, nchannel=np.unique(self.adr_sig.values).size
            )

            A.stop_event = self.stop_event

            sleep(3)
            # A.dataqueue = self.mappedQ
            A.dataqueue = rQ
            A.update_plot_until_empty()
        except Exception as e:
            self.stop()
            raise e

    def runFillMappedQueue(self, interval : float = 0.1, time : float = math.inf, n_value = 50):
        """
        Tool function used for debugging output interfaces.
        """
        try:
            from random import random

            if self.mappedQ is None:
                self.mappedQ = Queue()

            stopwatch = 0
            logger.info("start filling Q")
            while not self.stop_event.is_set() and stopwatch < time:
                # qfill = np.array([(random()*1)/2] * 50)
                # qfill = np.array(np.sin(0.01*np.outer(np.arange(50),1+32*1.0)).flatten())
                # qfill = np.random.random(size=(50,NCHANNELS)) 
                qfill = np.asarray( [[random()+i for i,adr in enumerate(self.raw_adrs)]] * n_value)       

                # print("putting : ",np.unique(qfill))
                self.mappedQ.put(qfill)
                sleep(interval)
                stopwatch += interval
            logger.info("finished filling Q")
        except Exception as e:
            self.stop()
            raise e

    def runReadQueue(self, queue: Queue, timeout: int = 10, interval_time: float = 0.5):
        try:
            while (not self.stop_event.is_set()): # or ((data := self.Q.get(timeout=timeout)) is not None):
                data = queue.get(timeout=timeout) 
                print(f"data:\n{data}\ntype : {type(data)}\nlength : {len(data)}")
                sleep(interval_time)
            print("read ended")
        except Exception as e:
            self.stop()
            raise e

    def runReadSignal(self, timeout=10):
        logger.info("read started")
        mappedValues : dict = None
        while (not self.stop_event.is_set()): # or ((data := self.Q.get(timeout=timeout)) is not None):
            data = self.rawQ.get(timeout=timeout) 
            if mappedValues is None:
                mappedValues = mapValuesByAdr(
                    data,
                    self.adr_sig.values,
                    1
                )
            else:
                newMappedValues = mapValuesByAdr(data,self.adr_sig.values,1)
                for adr in mappedValues.keys():
                    mappedValues[adr].append(newMappedValues[adr])
            sigList = []
            for adr in np.unique(self.adr_sig.values):
                sigList.append(
                    Signal(
                        fs=self.fb,
                        desc=adr
                        # cal=sigValues.cal
                    )
                )
            resultSignal = Signal.pack(sigList)
            logger.info(resultSignal)
            sleep(0.5)
        logger.info("read ended")

    def readyToQuit(self):
        if self.stop_event is None:
            self.stop_event = Event()
        try:
            print("enter q to stop")
            while(not self.stop_event.is_set() and input() != "q"):
                sleep(0.1)
        finally:
            self.stop()

    def start(self, 
              flag_autosave     : bool = False, filepath : str = None, # runAutoSaveH5
              flag_directplot   : bool = False, refresh_delay : float = None, plotbuffersize : int = None, # runDirectPlot
              flag_sim_mapdata  : bool = False, interval : float = None, time : float|int = None, n_value : int = None # runFillMappedQueue
              ):
        """
        Start a measurement and create output management depending on the parameters set.

        Tools :
        - Run in a thread readyToQuit. Stop measurement when 'q' char is sent in console.
        - Run in a thread dispatch. Read mappedQ and put data in Queues from resultQs.

        Process :
        - Run in a thread runCbMeasurement if flag_sim_mapdata is False. Measure and put raw values in rawQ.  
        - Run in a thread runMapData if flag_sim_mapdata is False. Read rawQ and map data with their channel and put it in mappedQ.  
        - Run in a thread runFillMappedQ if flag_sim_mapdata is True. Fill mappedQ with random values. Associated parameters are : 
            - interval
            - time
            - n_value
        
        Post-process :
        - Run in a thread runAutoSaveH5 if flag_autosave is True. Regulary save mapped data in a file. Associated parameters are :
            - filepath
        - Run in main thread runDirectPlot if flag_directplot is True. Plot mapped data with a line for each channel. Associated parameters are :
            - refresh_delay
            - plotbuffersize
        
        """
        Trtq = Thread(target=self.readyToQuit)
        Trdq = Thread(target=dispatch, args=(self.mappedQ, self.resultQs, self.stop_event, 10))
        if flag_autosave:
            Trsh = Thread(target=self.runAutoSaveH5, args=(filepath,))
        if not flag_sim_mapdata:
            Trmc = Thread(target=self.runCbMeasurement)
            Trmd = Thread(target=self.runMapData)
        else:
            Trfq = Thread(target=self.runFillMappedQueue, args=(0.08,))

        Trtq.start()
        Trdq.start()
        if flag_autosave:
            Trsh.start()
        if not flag_sim_mapdata:
            Trmc.start()
            Trmd.start()
        else:
            Trfq.start()
        if flag_directplot:
            self.runDirectPlot(refresh_delay,plotbuffersize)

        Trtq.join()
        Trdq.join()
        if flag_autosave:
            Trsh.join()
        if not flag_sim_mapdata:
            Trmc.join()
            Trmd.join()
        else:
            Trfq.join()

    def stop(self):
        self.stop_event.set()
        if self.rawQ is not None:
            self.rawQ.put(None)
        if self.mappedQ is not None:
            self.mappedQ.put(None)

if __name__ == "__main__":
    print("start")

    t0 = time()

    fs = 10000 # 1MHz max
    fb = 100
    rate = 0.8

    # H5 Save
    filepath = "C:\\Users\\cleme\\OneDrive\\Documents\\Stage ENSTA\\measpy\\ContinousMeasure.hdf5"
    if os.path.exists(filepath):
        os.remove(filepath)

    # Direct plot
    refresh_delay = 0.2
    plotbuffersize = 500

    # addr = [0,1,2,4,8,16,32]
    addr = list(range(0,32))
    addr_sig = mp.Signal(fs=fs,raw=addr,type=SignalType.DIGITAL)

    t1 = time()

    presS = PressureScanner(addresses=addr, fs=fs, fb=fb, rate=rate)

    t2 = time()

    presS.createMeasurement(
        adr_port=1, 
        out_device="dev2", 
        dur=None, 
        in_sig=[mp.Signal(fs=fs, unit="V")],
        # ,dur=5
        # in_map=[1]
    )
    # TO CHANGE, SHOULD BE SET INSIDE, NOT BY USER
    presS.n_raw_value = 500

    t3 = time()

    presS.start(
        flag_autosave=False, filepath=filepath,
        flag_directplot=True, refresh_delay=refresh_delay, plotbuffersize=plotbuffersize,
        flag_sim_mapdata=True, interval=0.08, time=None, n_value=None)

    t4 = time()

    print(f"Total time : {t4-t0}s")
    print(f"Time stamps :")
    print(t1-t0)
    print(t2-t1)
    print(t3-t2)
    print(t4-t3)

