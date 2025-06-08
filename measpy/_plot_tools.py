# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:40:04 2025

@author: clement
"""
import numpy as np
from abc import ABC, abstractmethod
from queue import Empty

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000

import time


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(
                f.__name__, (time2 - time1) * 1000.0
            )
        )

        return ret

    return wrap


def justify(a, axis=0, side="left"):
    """
    Justifies a 2D array
    :param a: Input array to be justified
    :type a: ndarray
    :param axis: Axis along which justification is to be made, defaults to 0
    :type axis: int, optional
    :param side: Direction of justification. It could be 'left', 'right', 'up', 'down'
    It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0., defaults to 'left'
    :type side: str, optional

    """

    mask = ~np.isnan(a)
    justified_mask = np.sort(mask, axis=axis)
    if (side == "up") | (side == "left"):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(a.shape, np.nan)
    if axis == 1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []
        self.changed_axe = False

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if type(art) is list:
            for oneArt in art:
                if oneArt.figure != self.canvas.figure:
                    raise RuntimeError
                oneArt.set_animated(True)
                self._artists.append(oneArt)
        else:
            if art.figure != self.canvas.figure:
                raise RuntimeError
            art.set_animated(True)
            self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        if self.changed_axe:
            self.canvas.draw()
            self.changed_axe = False
            plt.pause(0.0001)
        else:
            cv = self.canvas
            fig = cv.figure
            # paranoia in case we missed the draw event,
            if self._bg is None:
                self.on_draw(None)
            else:
                # restore the background
                cv.restore_region(self._bg)
                # draw all of the animated artists
                self._draw_animated()
                # update the GUI state
                cv.blit(fig.bbox)
            # let the GUI event loop process anything it has to do
            cv.flush_events()


class plot_data_from_queue(ABC):
    """
    Abstract class used to analyse and plot data that are feed into a queue (by a measurment callback)
    """

    plot_attribute = ["plotbuffer", "axes", "lines", "istimedata"]

    def __init__(
        self, fs, updatetime=0.1, plotbuffersize=2000, nchannel=1, show_time0=True
    ):
        """
        :param fs: Frequency (Hz) of the signal, used to define 'timeinterval' attribute
        that manage the time dependencies of the plot.
        :type fs: float
        :param updatetime: Time (second) between each update of the plot, defaults to 0.1
        :type updatetime: flaot, optional
        :param plotbuffersize: define attribute to store a size of the plot buffer
        that can be used inside 'plot_setup' method, defaults to 2000
        :type plotbuffersize: int, optional
        :param nchannel: Number of channels inside the Queue, defaults to 1
        :type nchannel: int, optional
        :param show_time0: If true, show a indicator for t0, defaults to True
        :type show_time0: bool, optional
        :return: Initialise the instance, with attribute useful to define plot and mandatory to use the class
        :rtype: plot_data_from_queue

        """
        for x in self.plot_attribute:
            setattr(self, x, None)
        self.timesincelastupdate = 0
        self.plotbuffersize = plotbuffersize
        self.updatetime = updatetime
        self.timeout = 0.1 * updatetime
        self.fs = fs
        self.timeinterval = 1 / self.fs
        self.plot_duration = plotbuffersize * self.timeinterval
        self.databuffersize = max(int(updatetime * self.fs), plotbuffersize)
        self.nchannel = nchannel
        if nchannel > 1:
            self.data_buffer = np.zeros((self.databuffersize, nchannel))
        else:
            self.data_buffer = np.zeros((self.databuffersize))
        self.plot_setup()
        for x in self.plot_attribute + ["fig"]:
            if getattr(self, x) is None:
                raise TypeError(
                    f"Subclasses 'plot_setup' method must set {x} to a non-None value"
                )
        nlines = len(self.lines)
        for x in self.plot_attribute:
            if not len(getattr(self, x)) == nlines:
                raise ValueError(
                    f"The size of {x} is not the same as the number of lines = {nlines}"
                )

        animated_artists = self.lines
        self.tend = 0
        if show_time0:
            self.time0 = self.axes[0].text(
                0.05,
                0.05,
                "$t_0 = 0$",
                transform=self.axes[0].transAxes,
                va="bottom",
                ha="left",
                animated=True,
            )
            animated_artists += [self.time0]
        else:
            self.time0 = None
        self.bm = BlitManager(self.fig.canvas, animated_artists)

    @abstractmethod
    def plot_setup(self):
        """
        Create the plot and attributes used to modify the plot :
            - plotbuffer: list of Numpy array that contain data to be plotted
            - fig : matplotlib figure
            - axes : list of matplotlib axes
            - lines : list of  matplotlib lines
            - istimedata : list of boolean, if true the nan for the considered axis are displaced
            to the right (allow better display at begining)

        plotbuffer, axes, lines and istimedata should be lists of the same size,
        each element corresponds to one line of data plotted

        """
        pass

    @abstractmethod
    def data_process(self):
        """
        Process data to be plotted by updating plotbuffer using data_buffer
        data_buffer is a numpy array (dimensions = [databuffersize,nchannels])
        that contain the data from the Queue
        databuffersize is greater than plotbuffersize and updatetime * fs

        """
        pass

    def _plotting_buffer(self):
        self.data_process()
        for ax, line, data, istimedata in zip(
            self.axes, self.lines, self.plotbuffer, self.istimedata
        ):
            if istimedata:
                data = justify(data)
            line.set_ydata(data)
        self.rescaling()
        if self.time0 is not None:
            self.tend += self.timesincelastupdate * self.timeinterval
            self.time0.set_text(
                f"$t_0 = {max(0,self.tend-self.plot_duration):.2f}$"
            )
        self.bm.update()
        self.timesincelastupdate = 0

    def rescaling(self):
        """
        This method is called automatically to rescale the data after each plot
        By default, it does nothing.
        It has to set self.bm.change_axe to True when axes are changed

        """
        pass

    def _update_data_buffer(self, item):
        n_values = len(item)
        # item = np.asarray(item) * 0.001  #mv to V
        self.timesincelastupdate += n_values
        if n_values <= self.databuffersize:
            self.data_buffer[:-n_values] = self.data_buffer[n_values:]
            self.data_buffer[-n_values:] = item
        else:
            self.data_buffer[:] = item[-self.databuffersize :]

    def update_plot(self, updatetime=None):
        updatetime = self.updatetime if updatetime is None else updatetime
        try:
            if (item := self.dataqueue.get(timeout=self.timeout)) is not None:
                item = np.asarray(item).squeeze()
                self._update_data_buffer(item)
                if self.timesincelastupdate * self.timeinterval > updatetime:
                    self._plotting_buffer()
        except (Empty, AttributeError):
            pass

    def update_plot_until_empty(self):
        try:
            while (item := self.dataqueue.get(timeout=10)) is not None:
                item = np.asarray(item).squeeze()
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
        if (item := dataqueue.get(timeout=100 * self.timeout)) is not None:
            item = np.asarray(item).squeeze()
            if item[0].size == self.data_buffer[0].size:
                self._update_data_buffer(item)
                if self.timesincelastupdate * self.timeinterval > self.updatetime:
                    self._plotting_buffer()
                self._dataqueue = dataqueue
            else:
                raise ValueError("Invalid queue")