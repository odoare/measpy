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
from measpy._plot_tools import basic_plot, basic_plot_with_fft
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
    fs = 2000
    Nchannel = 2
    dur = 150
    config = "DIFF"
    with_fft = True
    in_range = [5, 5]
    M = mp.Measurement(
        device_type="ni",
        in_sig=mp.Signal.pack([mp.Signal(fs=fs)] * Nchannel),
        dur=dur,
        in_sig_config=config,
        in_range=in_range,
    )
    filepath = "test.h5"
    # define plot parameter
    plot_time = 5
    refresh_delay = 0.1
    # size of plot buffer equal to the size of plot
    plotbuffersize = plot_time * fs

    # use ni_callback_measurement to set up measrument
    with ni_callback_measurement(M) as NI:
        bit_resolution = max(NI.ai_channels_bits_resolution)
        minvalue = [A[0] for A in NI.ai_channels_range]
        maxvalue = [A[1] for A in NI.ai_channels_range]
        # Create plot instance
        if with_fft:
            A = basic_plot_with_fft(
                fs,
                nchannel=Nchannel,
                updatetime=refresh_delay,
                plotbuffersize=plotbuffersize,
                show_time0=True,
                minvalue=minvalue,
                maxvalue=maxvalue,
                bit_resolution=bit_resolution,
            )
        else:
            A = basic_plot(
                fs,
                nchannel=Nchannel,
                updatetime=refresh_delay,
                plotbuffersize=plotbuffersize,
                show_time0=True,
            )
        # read data every refresh delay or time to fill the data buffer
        n_values = min(int(fs * refresh_delay), A.databuffersize)

        # Method wait for save_event then save into hdf5file
        def save_data(queuesave, save_event):
            # wait for button save push
            while not save_event.is_set():
                D = queuesave.get()
                # if no more data and button save never pushed, return
                if D is None:
                    return
            # create the hdf5 file
            funcsav = M.create_hdf5(filepath, chunck_size=n_values)
            funcsav(queuesave)

        # create a queue to read data
        Qin = Queue()
        # create a queue to use data
        Qout = Queue()

        P = Process_manager(
            queue_in=Qin,
            Raw_output=[
                Qout,
                partial(M.in_sig.fill_from_queue, unit_in="V"),
                partial(save_data, save_event=A.save_event),
            ],
        )

        data_wait_time = 1.5 * n_values / fs
        A.data_wait_time = data_wait_time
        # define the callback that fill up the queue

        def callback(buffer_in, n_values):
            Qin.put(buffer_in.copy())

        NI.set_callback(callback, n_values)

        # put the measurment into a thread
        def work(*args, **kwargs):
            NI.run(*args, **kwargs)
            # Don't forget end flag for the Queue
            Qin.put(None)

        T = Thread(
            target=work, kwargs={"stop": A.stop_event, "pause": A.pause_event}
        )
        # tstop = Thread(target=stop_after, args=(A.stop_event, 3))
        T.start()
        # tstop.start()

        try:
            # wait for first data chunk to arrive before starting process data
            time.sleep(data_wait_time)
            P.start()
            # wait for first data chunk to arrive before giving the queue to the plot instance
            time.sleep(data_wait_time)
            A.dataqueue = Qout
            # update the plot until end flag
            A.update_plot_until_empty()
            # wait for measurement to finish (should be already finished here)
            T.join()
            P.join()
        except (KeyboardInterrupt, Exception) as e:
            # stop measurment in case of exeption
            A.stop_event.set()
            raise e

# %%
