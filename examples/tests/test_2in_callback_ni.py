# %%

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:15:18 2025

@author: clement
"""
import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../../")

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

"""
Plot data in Volt and PSD at the same time as measurment with ni card, with axis rescaling options.
"""


# Define class with custom plot configuration.
class inline_plotting(plot_data_from_queue):
    def plot_setup(self):
        # define x_data : list of numpy array : x axis of the plot
        x_data = [
            np.arange(0, self.plotbuffersize) * self.timeinterval,
            np.fft.rfftfreq(n=self.plotbuffersize, d=self.timeinterval),
        ]
        # define plotbuffer, list of numpy array : y axis of the plot
        self.plotbuffer = [np.zeros_like(x) for x in x_data]
        # set defaults data to nan so it doesn't appear on the plot
        for buff in self.plotbuffer:
            buff[:] = np.nan
        # define figure and axes
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.subplots_adjust(bottom=0.2,left=0.2)
        # set axes labels
        self.axes[0].set_xlabel("Temps [s]", fontsize=15)
        self.axes[0].set_ylabel("Tension [V]", fontsize=15)
        self.axes[1].set_xlabel("Fréquence [Hz]", fontsize=15)
        self.axes[1].set_ylabel("Tension [V]", fontsize=15)
        # set axes limits
        self.axes[0].set_xlim([x_data[0][0], x_data[0][-1]])
        self.axes[0].set_ylim([-1, 1])
        self.axes[1].set_xlim([0.01, self.fs / 2])
        self.axes[1].set_ylim([10**-3, 10])
        # Plot the buffer to create lines objects
        (linet,) = self.axes[0].plot(x_data[0], self.plotbuffer[0], animated=True)
        (linef,) = self.axes[1].semilogy(
            x_data[1],
            np.ones_like(x_data[1]),
            animated=True,
        )
        #create animated artist to show some values
        self.std = self.axes[0].text(
            -0.5,
            0.8,
            "$standard \quad deviation = 0$",
            transform=self.axes[0].transAxes,
            va="bottom",
            ha="left",
            animated=True,
        )
        self.mean = self.axes[0].text(
            -0.5,
            0.9,
            "$mean = 0$",
            transform=self.axes[0].transAxes,
            va="bottom",
            ha="left",
            animated=True,
        )

        # define lines : list of line object that will be updated
        self.lines = [linet, linef]

        # displace nans to the right for first plot
        self.istimedata = [True, False]

        # Define a Stop button
        self.stop_event = Event()

        def fstop(event):
            self.stop_event.set()

        axs = self.fig.add_axes([0.3, 0.01, 0.2, 0.075])
        self.bstop = Button(axs, "Stop")
        self.bstop.on_clicked(fstop)

        # Define buttons to update a flag used to launch a method updating axis
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

        self.fMamp_plus = False

        def fMamp_plus(event):
            self.fMamp_plus = True

        afMxp = self.fig.add_axes([0.92, 0.9, 0.04, 0.05])
        self.bfMplus = Button(afMxp, "+")
        self.bfMplus.on_clicked(fMamp_plus)

        self.fMamp_moins = False

        def fMamp_moins(event):
            self.fMamp_moins = True

        afMxm = self.fig.add_axes([0.92, 0.8, 0.04, 0.05])
        self.bfMmoins = Button(afMxm, "-")
        self.bfMmoins.on_clicked(fMamp_moins)

        self.fmamp_plus = False

        def fmamp_plus(event):
            self.fmamp_plus = True

        afmxp = self.fig.add_axes([0.92, 0.3, 0.04, 0.05])
        self.bfmplus = Button(afmxp, "+")
        self.bfmplus.on_clicked(fmamp_plus)

        self.fmamp_moins = False

        def fmamp_moins(event):
            self.fmamp_moins = True

        afmxm = self.fig.add_axes([0.92, 0.2, 0.04, 0.05])
        self.bfmmoins = Button(afmxm, "-")
        self.bfmmoins.on_clicked(fmamp_moins)

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

        # set Stop event to stop measurment when the figure is closed.
        self.fig.canvas.mpl_connect("close_event", fstop)
        #this metho return list of animated artist
        return [self.std,self.mean]

    def rescaling(self):
        # defines method that rescale axis when a flag is set to True
        # the other flag : 'self.bm.change_axe = True' is needed because changing axis
        # is impossible with fast plot method, the axis are changed using slower plot method

        if self.tamp_plus:
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 0.5)
            self.tamp_plus = False
            self.bm.changed_axe = True

        if self.tamp_moins:
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 2)
            self.tamp_moins = False
            self.bm.changed_axe = True

        if self.fMamp_plus:
            self.axes[1].set_ylim(np.array(self.axes[1].get_ylim()) * [1, 0.1])
            self.fMamp_plus = False
            self.bm.changed_axe = True

        if self.fMamp_moins:
            self.axes[1].set_ylim(np.array(self.axes[1].get_ylim()) * [1, 10])
            self.fMamp_moins = False
            self.bm.changed_axe = True

        if self.fmamp_plus:
            self.axes[1].set_ylim(np.array(self.axes[1].get_ylim()) * [0.1, 1])
            self.fmamp_plus = False
            self.bm.changed_axe = True

        if self.fmamp_moins:
            self.axes[1].set_ylim(np.array(self.axes[1].get_ylim()) * [10, 1])
            self.fmamp_moins = False
            self.bm.changed_axe = True

        if self.freq_plus:
            self.axes[1].set_xlim([0, np.array(self.axes[1].get_xlim())[1] * 0.5])
            self.freq_plus = False
            self.bm.changed_axe = True

        if self.freq_moins:
            self.axes[1].set_xlim([0, np.array(self.axes[1].get_xlim())[1] * 2])
            self.freq_moins = False
            self.bm.changed_axe = True

    def data_process(self):
        # Transfert data from data_buffer to plotbuffer
        self.plotbuffer[0][: -self.timesincelastupdate] = self.plotbuffer[0][
            self.timesincelastupdate :
        ]
        self.plotbuffer[0][-self.timesincelastupdate :] = self.data_buffer[
            -self.timesincelastupdate :
        ].copy()
        # fft for the second plot, begin to be calculated only after enough data arrived
        if not any(np.isnan(self.plotbuffer[0])):
            self.plotbuffer[1][:] = np.abs(
                np.fft.rfft(self.plotbuffer[0], norm="ortho")
            )
        #update the animated artist text
        std = np.std(self.plotbuffer[0])
        self.std.set_text(
            f"$standard \quad deviation = {std:.2f}$"
        )
        mean = np.mean(self.plotbuffer[0])
        self.mean.set_text(
            f"$mean = {mean:.2f}$"
        )


if __name__ == "__main__":
    # def stop_after(event,T):
    #     time.sleep(T)
    #     event.set()

    # define a measurment
    fs = 10000
    M = mp.Measurement(device_type="ni", in_sig=[mp.Signal(fs=fs)], dur=15)

    # define plot pameaspyrameter
    plot_time = 5
    refresh_delay = 0.1
    # size of plot buffer equal to the size of plot
    plotbuffersize = plot_time * fs
    # Create plot instance
    A = inline_plotting(
        fs,
        updatetime=refresh_delay,
        plotbuffersize=plotbuffersize,
        show_time0=True,
    )

    # create a queue
    Q = Queue()

    # define the callback that fill up the queue

    def callback(buffer_in, n_values):
        Q.put(buffer_in.copy())

    # read data every refresh delay or time to fill the data buffer
    n_values = min(int(fs * refresh_delay), A.databuffersize)
    # use ni_callback_measurement to set up measrument
    with ni_callback_measurement(M) as NI:
        NI.set_callback(callback, n_values)

        # put the measurment into a thread
        def work(*args):
            NI.run(*args)
            print("measurment done")
            # Don't forget end flag for the Queue
            Q.put(None)

        T = Thread(target=work, args=(A.stop_event, None))
        # tstop = Thread(target=stop_after, args=(A.stop_event, 3))
        T.start()
        # tstop.start()

        try:
            # wait for first data chunk to arrive before giving the queue to the plot instance
            time.sleep(1.5 * n_values / fs)
            A.dataqueue = Q
            # update the plot until end flag
            A.update_plot_until_empty()
            # wait for measurement to finish (should be already finished here)
            T.join()
        except Exception as e:
            # stop measurment in case of exeption
            A.stop_event.set()
            raise e

# %%
