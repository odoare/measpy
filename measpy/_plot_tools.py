# measpy/_plot_tools.py
#
# ------------------------------------
# Utilities for ploting data real time
# ------------------------------------

import numpy as np
from abc import ABC, abstractmethod
from queue import Empty

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button, TextBox

from threading import Event

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000

import time
from ._tools import to_list


def float_inrange(number, rngs):
    try:
        num = float(number)
        if (V := rngs[0]) is not None:
            if num < V:
                raise ValueError
        if (V := rngs[1]) is not None:
            if num > V:
                raise ValueError
    except ValueError:
        print("\a")
        return
    return num


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
    # Faster real time plot, from matplotlib tutorial:
    # https://matplotlib.org/stable/users/explain/animations/blitting.html
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
                if plt.fignum_exists(
                    fig.number
                ):  # can be deleted for recent matplotlib version (merge  #25104)
                    cv.blit(fig.bbox)
            # let the GUI event loop process anything it has to do
            cv.flush_events()


class plot_data_from_queue(ABC):
    """
    Abstract class used to analyse and plot data that are feed into a queue (by a measurment callback)
    """

    plot_attribute = ["plotbuffer", "axes", "lines", "istimedata"]

    def __init__(
        self,
        fs,
        updatetime=0.1,
        plotbuffersize=2000,
        nchannel=1,
        show_time0=True,
        minvalue=-10,
        maxvalue=10,
        bit_resolution=16,
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
        :param minvalue: Give the min expected data value to set up plot, defaults to -10
        :type minvalue: float, optional
        :param maxvalue: Give the max expected data value to set up plot, defaults to 10
        :type maxvalue: float, optional
        :param bit_resolution: Resolution in bit of the data source, can be use to set plot with fft, 16
        :type bit_resolution: int, optional
        :return: Initialise the instance, with attribute useful to define plot and mandatory to use the class
        :rtype: plot_data_from_queue

        """
        for x in self.plot_attribute:
            setattr(self, x, None)
        self.timesincelastupdate = 0
        self.plotbuffersize = plotbuffersize
        self.updatetime = updatetime
        self.timeout = updatetime
        self.fs = fs
        self.minvalue = to_list(minvalue, nchannel)
        self.maxvalue = to_list(maxvalue, nchannel)
        self.Nbits = bit_resolution
        self.timeinterval = 1 / self.fs
        self.plot_duration = plotbuffersize * self.timeinterval
        self.databuffersize = max(int(updatetime * self.fs), plotbuffersize)
        self.nchannel = nchannel
        if nchannel > 1:
            self.data_buffer = np.zeros((self.databuffersize, nchannel))
        else:
            self.data_buffer = np.zeros((self.databuffersize))
        animated_artists = self.plot_setup()
        if not animated_artists:
            animated_artists = []
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

        animated_artists += self.lines
        self.tend = 0
        self.t0 = 0
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
        can return a animated artist to be updated inside data_process

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
        isnan = False
        for ax, line, data, istimedata in zip(
            self.axes, self.lines, self.plotbuffer, self.istimedata
        ):
            if istimedata and any(np.isnan(data)):
                data = justify(data)
                isnan = True
            line.set_ydata(data)
        self.rescaling()
        if self.time0 is not None:
            self.tend += self.timesincelastupdate * self.timeinterval
            if isnan:
                self.t0 = 0
            else:
                self.t0 = self.tend - self.plot_duration
            self.time0.set_text(f"$t_0 = {self.t0:.2f}$")
        self.bm.update()
        self.timesincelastupdate = 0

    def after_plot(self):
        """
        This method is called automatically when there is no more data to plot
        By default, it does nothing.

        """
        pass

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

    def update_plot(self, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        try:
            if (item := self.dataqueue.get(timeout=timeout)) is not None:
                item = np.asarray(item).squeeze()
                self._update_data_buffer(item)
                if (
                    self.timesincelastupdate * self.timeinterval
                    >= self.updatetime
                ):
                    self._plotting_buffer()
        except (Empty, AttributeError):
            pass

    def update_plot_until_empty(self):
        try:
            while (item := self.dataqueue.get(timeout=self.timeout)) is not None:
                item = np.asarray(item).squeeze()
                self._update_data_buffer(item)
                if (
                    self.timesincelastupdate * self.timeinterval
                    >= self.updatetime
                ):
                    self._plotting_buffer()
            if self.timesincelastupdate > 0:
                self._plotting_buffer()
        except (Empty, AttributeError):
            self.after_plot()

    def close(self, event):
        self.fig.canvas.stop_event_loop()

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
                if (
                    self.timesincelastupdate * self.timeinterval
                    >= self.updatetime
                ):
                    self._plotting_buffer()
                self._dataqueue = dataqueue
            else:
                raise ValueError(
                    f"Invalid queue : expected {self.nchannel} channels",
                    f"The queue seem to have {item[0].size} channels",
                )


class basic_plot(plot_data_from_queue):
    """
    Plot data as value (Volt) as function of time; supports multichannel data, plot all data in the same axe
    Stop button trigger self.stop_event and should stop the measurment thread
    Save button trigger self.save_event and should start the save data thread

    """

    def plot_setup(self):
        # define x_data : list of numpy array : x axis of the plot
        x_data = [
            np.arange(0, self.plotbuffersize) * self.timeinterval
        ] * self.nchannel
        # define plotbuffer, list of numpy array : y axis of the plot
        self.plotbuffer = [np.zeros_like(x) for x in x_data]
        # set defaults data to nan so it doesn't appear on the plot
        for buff in self.plotbuffer:
            buff[:] = np.nan
        # define figure and axes
        self.fig, axe = plt.subplots(1, 1, figsize=(8, 5))
        self.fig.subplots_adjust(bottom=0.2, left=0.2)
        self.axes = [axe] * self.nchannel
        # set axes labels
        axe.set_xlabel("Temps [s]", fontsize=15)
        axe.set_ylabel("Tension [V]", fontsize=15)
        # set axes limits
        axe.set_xlim([x_data[0][0], x_data[0][-1]])

        axe.set_ylim([min(self.minvalue), max(self.maxvalue)])
        # Plot the buffer to create lines objects
        linet = [
            axe.plot(x, y, animated=True)[0]
            for x, y in zip(x_data, self.plotbuffer)
        ]
        # define lines : list of line object that will be updated
        self.lines = linet
        plt.show(block=False)
        # to displace nans to the right for begining of plot
        self.istimedata = [True] * self.nchannel

        # Define a Stop button
        self.stop_event = Event()

        def fstop(event):
            self.stop_event.set()

        axs = self.fig.add_axes([0.3, 0.01, 0.2, 0.075])
        self.bstop = Button(axs, "Stop")
        self.cid_bstop = self.bstop.on_clicked(fstop)

        # Define a Pause button
        self.pause_event = Event()
        axs = self.fig.add_axes([0.5, 0.01, 0.2, 0.075])
        self.bpause = Button(axs, "Pause")
        self.cid_pause = self.bpause.on_clicked(self._fpause)

        # Define a Save button
        self.save_event = Event()

        def save(event):
            self.save_event.set()

        axs = self.fig.add_axes([0.7, 0.01, 0.2, 0.075])
        self.bsave = Button(axs, "Save")
        self.cid_bsave = self.bsave.on_clicked(save)

        # Define buttons to update a flag used to launch a method updating axis
        self.tamp_plus = False

        def tamp_plus(event):
            self.tamp_plus = True

        atxp = self.fig.add_axes([0.02, 0.6, 0.04, 0.05])
        self.btplus = Button(atxp, "+")
        self.cid_btplus = self.btplus.on_clicked(tamp_plus)

        self.tamp_moins = False

        def tamp_moins(event):
            self.tamp_moins = True

        atxm = self.fig.add_axes([0.02, 0.4, 0.04, 0.05])
        self.btmoins = Button(atxm, "-")
        self.cid_btmoin = self.btmoins.on_clicked(tamp_moins)

        # set Stop event to stop measurment when the figure is closed.
        # self.cid_stop is important to avoid memory leak
        self.cid_stop = self.fig.canvas.mpl_connect("close_event", self.close)

    def close(self, event):
        self.stop_event.set()
        super().close(event)

    def rescaling(self):
        # defines method that rescale axis when a flag is set to True
        # the other flag : 'self.bm.changed_axe = True' is needed because changing axis
        # is impossible with fast plot method, the axis are changed using slower plot method

        if self.tamp_plus:
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 0.5)
            self.tamp_plus = False
            self.bm.changed_axe = True

        if self.tamp_moins:
            self.axes[0].set_ylim(np.array(self.axes[0].get_ylim()) * 2)
            self.tamp_moins = False
            self.bm.changed_axe = True

    # methods used to pause and restart
    def _fpause(self, event):
        self.pause_event.set()
        self.bpause.label.set_text("...")
        self.bm.changed_axe = True
        self.bpause.disconnect(self.cid_pause)

    def after_plot(self):
        if self.pause_event.is_set():
            self.fig.waitforbuttonpress()
            self.restart()

    def restart(self):
        self.bpause.label.set_text("Pause")
        self.bm.changed_axe = True
        self.tend = 0
        self.pause_event.clear()
        self.cid_pause = self.bpause.on_clicked(self._fpause)
        self.update_plot(100 * self.timeout)
        self.update_plot_until_empty()

    def data_process(self):
        # Transfert data from data_buffer to plotbuffer
        for i, buff in enumerate(self.plotbuffer):
            buff[: -self.timesincelastupdate] = buff[self.timesincelastupdate :]
            try:
                data = self.data_buffer[-self.timesincelastupdate :, i].copy()
            except IndexError:
                data = self.data_buffer[-self.timesincelastupdate :].copy()
            buff[-self.timesincelastupdate :] = data


class rescal_axis:
    def __init__(
        self,
        axis_rescal,
        Tb_max,
        Tb_min,
        plotmanager,
        startvalues,
        minv=None,
        maxv=None,
    ):
        self.min, self.max = startvalues
        self.rescal = axis_rescal
        self.plotm = plotmanager
        self.changed = False

        def maxval(number):
            if (V := float_inrange(number, (self.min, maxv))) is not None:
                self.max = V
                self.changed = True
                self.plotm.bm.changed_axe = True

        self.Tb_max = Tb_max
        self.Tb_max.on_submit(maxval)

        def minval(number):
            if (V := float_inrange(number, (minv, self.max))) is not None:
                self.min = V
                self.changed = True
                self.plotm.bm.changed_axe = True

        self.Tb_min = Tb_min
        self.Tb_min.on_submit(minval)

    def rescale(self):
        if self.changed:
            self.rescal([self.min, self.max])
            self.changed = False


class basic_plot_with_fft(plot_data_from_queue):
    """
    Plot data as value (Volt) as function of time and fft of the data in a second plot;
    supports multichannel data, plot all data in the same axe
    Stop button trigger self.stop_event and should stop the measurment thread
    Save button trigger self.save_event and should start the save data thread

    """

    def plot_setup(self):
        # define x_data : list of numpy array : x axis of the plot
        x_data = [[], []]
        x_data[0] = [
            np.arange(0, self.plotbuffersize) * self.timeinterval
        ] * self.nchannel

        x_data[1] = [
            np.fft.rfftfreq(n=self.plotbuffersize, d=self.timeinterval),
        ] * self.nchannel
        # define plotbuffer, list of numpy array : y axis of the plot
        self.plotbuffer = [np.zeros_like(x) for x in x_data[0]] + [
            np.zeros_like(x) for x in x_data[1]
        ]
        # set defaults data to nan so it doesn't appear on the plot
        for buff in self.plotbuffer:
            buff[:] = np.nan
        # define figure and axes
        self.fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.subplots_adjust(bottom=0.2, left=0.2)
        # set axes labels
        axes[0].set_xlabel("Temps [s]", fontsize=15)
        axes[0].set_ylabel("Tension [V]", fontsize=15)
        axes[1].set_xlabel("Fréquence [Hz]", fontsize=15)
        axes[1].set_ylabel("Niveau du signal [dBu]", fontsize=15)
        # set axes limits
        axes[0].set_xlim([x_data[0][0][0], x_data[0][0][-1]])
        self.ymin, self.ymax = (min(self.minvalue), max(self.maxvalue))
        axes[0].set_ylim([self.ymin, self.ymax])

        self.fmin, self.fmax = (x_data[1][0][0], x_data[1][0][-1])
        axes[1].set_xlim([self.fmin, self.fmax])
        FS = [M - m for M, m in zip(self.maxvalue, self.minvalue)]
        dbFs = 20 * np.log10(max(FS))
        bits_noise_level = 20 * np.log10(min(FS)) - 6.02 * self.Nbits

        self.mindb, self.maxdb = bits_noise_level, dbFs
        axes[1].set_ylim([self.mindb, self.maxdb])
        self.axes = [axes[0]] * self.nchannel + [axes[1]] * self.nchannel
        # set axes limits
        # Plot the buffer to create lines objects
        linet = [
            axes[0].plot(x, y, animated=True, label=f"Channel {i+1}")[0]
            for i, (x, y) in enumerate(
                zip(x_data[0], self.plotbuffer[: self.nchannel])
            )
        ]
        axes[0].legend()
        linef = [
            axes[1].plot(x, np.ones_like(x), animated=True)[0] for x in x_data[1]
        ]
        # define lines : list of line object that will be updated

        self.lines = linet + linef
        # to displace nans to the right for begining of plot
        self.istimedata = [True] * self.nchannel + [False] * self.nchannel

        # create animated artist to show some values
        self.std_str = "$Std\ :$\n" + "\n".join(
            [f"$Channel\ {i+1}= {{:.2f}}$" for i in range(self.nchannel)]
        )
        self.std = self.axes[0].text(
            -0.5,
            0.4,
            self.std_str.format(*(0,) * self.nchannel),
            transform=self.axes[0].transAxes,
            va="bottom",
            ha="left",
            animated=True,
        )
        self.mean_str = "$Mean\ :$\n" + "\n".join(
            [f"$Channel\ {i+1}= {{:.2f}}$" for i in range(self.nchannel)]
        )
        self.mean = self.axes[0].text(
            -0.5,
            0.6,
            self.mean_str.format(*(0,) * self.nchannel),
            transform=self.axes[0].transAxes,
            va="bottom",
            ha="left",
            animated=True,
        )

        # Define a Stop button
        self.stop_event = Event()
        self.pause_event = Event()

        def fstop(event):
            self.stop_event.set()

        axs = self.fig.add_axes([0.25, 0.9, 0.2, 0.075])
        self.bstop = Button(axs, "Stop")
        self.cid_btstop = self.bstop.on_clicked(fstop)

        axs = self.fig.add_axes([0.45, 0.9, 0.2, 0.075])
        self.bpause = Button(axs, "Pause")
        self.cid_pause = self.bpause.on_clicked(self._fpause)

        # Define a Save button
        self.save_event = Event()

        def save(event):
            self.save_event.set()

        axs = self.fig.add_axes([0.65, 0.9, 0.2, 0.075])
        self.bsave = Button(axs, "Save")
        self.cid_btsave = self.bsave.on_clicked(save)

        # Define textbox to set y axis limit
        atxp = self.fig.add_axes([0.12, 0.8, 0.04, 0.05])
        Vmax = TextBox(atxp, "Max value")

        atxm = self.fig.add_axes([0.12, 0.2, 0.04, 0.05])
        Vmin = TextBox(atxm, "Min value")

        self.yrescal = rescal_axis(
            self.axes[0].set_ylim, Vmax, Vmin, self, (self.ymin, self.ymax)
        )

        # Define textbox to set dBu limit
        adbM = self.fig.add_axes([0.92, 0.8, 0.04, 0.05])
        Maxdbu = TextBox(adbM, "dBu max")
        label = Maxdbu.ax.get_children()[0]
        label.set_position([0.5, 1.5])
        label.set_verticalalignment("top")
        label.set_horizontalalignment("center")

        adbm = self.fig.add_axes([0.92, 0.2, 0.04, 0.05])
        Mindbu = TextBox(adbm, "dBu min")
        label = Mindbu.ax.get_children()[0]
        label.set_position([0.5, 1.5])
        label.set_verticalalignment("top")
        label.set_horizontalalignment("center")

        self.dburescal = rescal_axis(
            self.axes[self.nchannel].set_ylim,
            Maxdbu,
            Mindbu,
            self,
            (self.mindb, self.maxdb),
        )

        # Define textbox to set frequency limit
        afreqma = self.fig.add_axes([0.85, 0.08, 0.04, 0.05])
        Tb_max = TextBox(afreqma, "f max")

        afreqmi = self.fig.add_axes([0.6, 0.08, 0.04, 0.05])
        Tb_min = TextBox(afreqmi, "f min")
        self.freq_rescal = rescal_axis(
            self.axes[self.nchannel].set_xlim,
            Tb_max,
            Tb_min,
            self,
            (self.fmin, self.fmax),
            minv=self.fmin,
            maxv=self.fmax,
        )

        # set Stop event to stop measurment when the figure is closed.
        self.cid_stop = self.fig.canvas.mpl_connect("close_event", self.close)
        plt.show(block=False)
        return [self.std, self.mean]

    def close(self, event):
        self.stop_event.set()
        super().close(event)

    def rescaling(self):
        # defines method that rescale axis when a flag is set to True
        # should check if the flag 'self.bm.changed_axe is True', changing axis
        # is impossible with fast plot method, the axis are changed using a slower plot method
        if self.bm.changed_axe:
            self.yrescal.rescale()
            self.dburescal.rescale()
            self.freq_rescal.rescale()

    def _fpause(self, event):
        self.pause_event.set()
        self.bpause.label.set_text("...")
        self.bm.changed_axe = True
        self.bpause.disconnect(self.cid_pause)

    def after_plot(self):
        if self.pause_event.is_set():
            self.fig.waitforbuttonpress()
            self.restart()

    def restart(self):
        self.bpause.label.set_text("Pause")
        self.bm.changed_axe = True
        self.tend = 0
        self.t0 = 0
        self.pause_event.clear()
        self.cid_pause = self.bpause.on_clicked(self._fpause)
        for fft in self.plotbuffer[self.nchannel :]:
            fft[:] = 1
        self.update_plot(100 * self.timeout)
        self.update_plot_until_empty()

    def data_process(self):
        # Transfert data from data_buffer to plotbuffer
        for i, (buff, fftbuff) in enumerate(
            zip(
                self.plotbuffer[: self.nchannel], self.plotbuffer[self.nchannel :]
            )
        ):
            buff[: -self.timesincelastupdate] = buff[self.timesincelastupdate :]
            try:
                data = self.data_buffer[-self.timesincelastupdate :, i].copy()
            except IndexError:
                data = self.data_buffer[-self.timesincelastupdate :].copy()
            buff[-self.timesincelastupdate :] = data
            # fft for the second plot, begin to be calculated only after enough data arrived
            if not any(np.isnan(buff)) and self.t0 > 0:
                fftbuff[:] = 20 * np.log10(
                    np.abs(np.fft.rfft(buff, norm="ortho"))
                )
        # update the animated artist text
        std = np.std(self.plotbuffer[: self.nchannel], axis=1)
        self.std.set_text(self.std_str.format(*std))
        mean = np.mean(self.plotbuffer[: self.nchannel], axis=1)
        self.mean.set_text(self.mean_str.format(*mean))
