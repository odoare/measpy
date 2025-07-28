# %%

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:15:18 2025

@author: clement
"""
import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../../")

# import matplotlib

# matplotlib.use("TkAgg")

import measpy as mp
from measpy._plot_tools import basic_plot
from measpy._data_tools import Process_manager
from measpy.ni import ni_callback_measurement
from functools import partial
import time
from threading import Thread
from queue import Queue
"""
Plot data in Volt and PSD at the same time as measurment with ni card, with axis rescaling options.
"""

if __name__ == "__main__":
    # define a measurment
    fs = 5000
    Nchannel = 2
    dur = 150
    M = mp.Measurement(device_type="ni", in_sig=mp.Signal.pack([mp.Signal(fs=fs)]*Nchannel), dur=dur)
    filepath = "test.h5"
    # define plot parameter
    plot_time = 5
    refresh_delay = 0.1
    # size of plot buffer equal to the size of plot
    plotbuffersize = plot_time * fs
    # Create plot instance
    A = basic_plot(
        fs,
        nchannel=Nchannel,
        updatetime=refresh_delay,
        plotbuffersize=plotbuffersize,
        show_time0=True,
    )
    # read data every refresh delay or time to fill the data buffer
    n_values = min(int(fs * refresh_delay), A.databuffersize)

    # create the hdf5 file
    funcsav = M.create_hdf5(filepath,chunck_size=n_values)
    #Method wait for save_event then save into hdf5file
    def save_data(queuesave, save_event):
        while not save_event.is_set():
            D = queuesave.get(timeout=10)
            if D is None:
                return
        funcsav(queuesave)
    # create a queue to read data
    Qin = Queue()
    # create a queue to use data
    Qout = Queue()

    P = Process_manager(
        queue_in=Qin,
        Raw_output=[Qout, partial(M.in_sig.fill_from_queue,unit_in="V"),partial(save_data,save_event=A.save_event)]
    )

    # define the callback that fill up the queue

    def callback(buffer_in, n_values):
        Qin.put(buffer_in.copy())

    # use ni_callback_measurement to set up measrument
    with ni_callback_measurement(M) as NI:
        NI.set_callback(callback, n_values)

        # put the measurment into a thread
        def work(*args):
            NI.run(*args)
            print("measurment done")
            # Don't forget end flag for the Queue
            Qin.put(None)

        T = Thread(target=work, args=(A.stop_event,))
        # tstop = Thread(target=stop_after, args=(A.stop_event, 3))
        T.start()
        # tstop.start()

        try:
            # wait for first data chunk to arrive before starting process data
            time.sleep(1.5 * n_values / fs)
            P.start()
            # wait for first data chunk to arrive before giving the queue to the plot instance
            time.sleep(1.5 * n_values / fs)
            A.dataqueue = Qout
            # update the plot until end flag
            A.update_plot_until_empty()
            # wait for measurement to finish (should be already finished here)
            T.join()
            P.join()
        except Exception as e:
            # stop measurment in case of exeption
            A.stop_event.set()
            raise e

# %%
