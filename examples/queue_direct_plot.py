# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:15:18 2025

@author: clement
"""

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import measpy as mp
from measpy._plot_tools import plot_data_from_queue
from measpy.ni import ni_callback_measurement
import time
from threading import Thread, Event
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Define class with custom plot configuration.
class inline_plotting(plot_data_from_queue):
    def plot_setup(self):
        # define x_data : list of numpy array : x axis of the plot
        self.x_data = [
            np.arange(0, self.plotbuffersize) * self.timeinterval,
            np.fft.rfftfreq(n=self.plotbuffersize, d=self.timeinterval),
        ]
        # define plotbuffer, list of numpy array : y axis of the plot
        self.plotbuffer = [np.zeros_like(x) for x in self.x_data]
        for buff in self.plotbuffer:
            buff[:] = np.nan
        # define figure and axes
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.subplots_adjust(bottom=0.2)
        # axes labesl
        self.axes[0].set_xlabel("Temps [s]", fontsize=15)
        self.axes[0].set_ylabel("Tension [V]", fontsize=15)
        self.axes[1].set_xlabel("Fréquence [Hz]", fontsize=15)
        self.axes[1].set_ylabel("Tension [V]", fontsize=15)
        self.axes[1].set_xlim([0.01, self.fs / 2])
        self.axes[1].set_ylim([0, 10])
        # Plot the buffer
        (linet,) = self.axes[0].plot(self.x_data[0], self.plotbuffer[0])
        (linef,) = self.axes[1].plot(
            self.x_data[1],
            self.plotbuffer[1] + 1,
        )

        # define lines : list of plt line updated with the buffer
        self.lines = [linet, linef]

        # No autoscale
        self.autoscale = [False, False]

        # Stop button
        self.stop_event = Event()

        def fstop(event):
            self.stop_event.set()

        axs = self.fig.add_axes([0.3, 0.01, 0.2, 0.075])
        self.bstop = Button(axs, "Stop")
        self.bstop.on_clicked(fstop)

        # Buttons that can rescale axis (after each plot update)
        self.tamp_plus = False

        def tamp_plus(event):
            self.tamp_plus = True

        atxp = self.fig.add_axes([0.02, 0.6, 0.04, 0.05])
        self.btplus = Button(atxp, "+")
        self.btplus.on_clicked(tamp_plus)

        self.tamp_moins = False

        def tamp_moins(event):
            self.tamp_moins = True

        atxm = self.fig.add_axes([0.02, 0.4, 0.04, 0.05])
        self.btmoins = Button(atxm, "-")
        self.btmoins.on_clicked(tamp_moins)

        self.famp_plus = False

        def famp_plus(event):
            self.famp_plus = True

        afxp = self.fig.add_axes([0.92, 0.6, 0.04, 0.05])
        self.bfplus = Button(afxp, "+")
        self.bfplus.on_clicked(famp_plus)

        self.famp_moins = False

        def famp_moins(event):
            self.famp_moins = True

        afxm = self.fig.add_axes([0.92, 0.4, 0.04, 0.05])
        self.bfmoins = Button(afxm, "-")
        self.bfmoins.on_clicked(famp_moins)

        self.freq_plus = False

        def freq_plus(event):
            self.freq_plus = True

        afreqp = self.fig.add_axes([0.6, 0.01, 0.04, 0.05])
        self.freqp = Button(afreqp, "+")
        self.freqp.on_clicked(freq_plus)

        self.freq_moins = False

        def freq_moins(event):
            self.freq_moins = True

        afreqm = self.fig.add_axes([0.7, 0.01, 0.04, 0.05])
        self.freqm = Button(afreqm, "-")
        self.freqm.on_clicked(freq_moins)

        self.fig.canvas.mpl_connect("close_event", fstop)

    def rescaling(self):
        # defines method to rescale axis (called after each new plot)
        # 'self.bm.change_axe = True' needed because changing axis impossible with fast plot
        # using slow plot method when axis are changed

        self.axes[0].set_xlim([self.x_data[0][0], self.x_data[0][-1]])
        if self.tamp_plus:
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) / 2)
            self.tamp_plus = False
            self.bm.change_axe = True

        if self.tamp_moins:
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 2)
            self.tamp_moins = False
            self.bm.change_axe = True

        if self.famp_plus:
            self.axes[1].set_ylim(np.array(self.axes[1].get_ylim()) / 2)
            self.famp_plus = False
            self.bm.change_axe = True

        if self.famp_moins:
            self.axes[1].set_ylim(np.array(self.axes[1].get_ylim()) * 2)
            self.famp_moins = False
            self.bm.change_axe = True

        if self.freq_plus:
            self.axes[1].set_xlim([0, np.array(self.axes[1].get_xlim())[1] / 2])
            self.freq_plus = False
            self.bm.change_axe = True

        if self.freq_moins:
            self.axes[1].set_xlim([0, np.array(self.axes[1].get_xlim())[1] * 2])
            self.freq_moins = False
            self.bm.change_axe = True

    def data_process(self):
        # Transfer and process data in data_buffer to plotbuffer
        self.plotbuffer[0][: -self.timesincelastupdate] = self.plotbuffer[0][
            self.timesincelastupdate :
        ]
        self.plotbuffer[0][-self.timesincelastupdate :] = self.data_buffer[
            -self.timesincelastupdate :
        ].copy()
        self.plotbuffer[1][:] = (
            np.abs(np.fft.rfft(self.plotbuffer[0], norm="ortho")) ** 2
        )


if __name__ == "__main__":
    # def stop_after(event,T):
    #     time.sleep(T)
    #     event.set()

    # define a measurment
    fs = 10000
    M = mp.Measurement(device_type="ni", in_sig=[mp.Signal(fs=fs)], dur=15)

    # define plot parameter
    plot_time = 5
    refresh_delay = 0.2
    plotbuffersize = plot_time * fs
    A = inline_plotting(
        fs,
        updatetime=refresh_delay,
        plotbuffersize=plotbuffersize,
        show_time0=False,
    )

    # create a queue
    Q = Queue()

    # define the callback that fill up the queue
    n_values = min(int(fs * refresh_delay / 2), plotbuffersize)

    def callback(buffer_in, n_values):
        Q.put(buffer_in.copy())

    # use ni_callback_measurement to set up measrument

    with ni_callback_measurement(M) as NI:
        NI.set_callback(callback, n_values)

        # put the measurment into a thread
        def work(*args):
            NI.run(*args)
            print("measurment done")
            Q.put(None)

        T = Thread(target=work, args=(A.stop_event, None))
        # tstop = Thread(target=stop_after, args=(A.stop_event, 3))
        T.start()
        # tstop.start()

        try:
            # wait for first data to update before update the plot
            time.sleep(1.5 * n_values / fs)
            A.dataqueue = Q
            A.update_plot_until_empty()
            T.join()
        except Exception as e:
            # stop measurment in case of exeption
            A.stop_event.set()
            raise e
