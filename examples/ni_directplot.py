import measpy as mp
import tkinter as tk
from measpy.ni import ni_callback_measurement
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from threading import Thread, Event


# Define Tkinter app using mp.Measurement and mp.ni.ni_callback_measurement
class direct_plot(tk.Tk):
    def __init__(self, NI, refresh_delay):
        super().__init__()
        # close all when windows closed
        self.protocol("WM_DELETE_WINDOW", self._close)
        # Signal to stop measurment
        self.stop_event = Event()

        self.M = NI.M
        self.NI = NI

        # Number of data read at each callback call
        self.n_values = int(refresh_delay * M.fs)
        # Duration of plotted data
        plotbuffersize = int(M.dur * M.fs)

        # Set up plot
        figure = Figure(figsize=(6, 4), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(figure, self)
        self.figure_canvas.get_tk_widget().grid()
        ax = figure.add_subplot()
        x = np.arange(plotbuffersize)
        self.plotbuffer = np.sin(x * 100 / plotbuffersize)
        ax.set_xlabel("Temps [s]", fontsize=15)
        ax.set_ylabel("Tension [V]", fontsize=15)
        ax.set_ylim([-10, 10])
        self.linet = ax.plot(x / fs, self.plotbuffer)

        # Start/Stop button
        self.textstatus = tk.StringVar(self, value="Start")
        start_stop_button = tk.Button(
            self, textvariable=self.textstatus, command=self.start_stop
        )
        start_stop_button.grid()

        # Set up callback method
        self.NI.set_callback(self.callback, self.n_values)

    def _close(self):
        self.stop_event.set()
        self.quit()
        self.destroy()

    def start_stop(self):
        if self.textstatus.get() == "Start":  # Start measurment
            # Button change to "Stop"
            self.textstatus.set("Stop")
            # Reset event
            self.stop_event.clear()
            self.run()
        else:  # Stop measurment
            # Send stop signal in the thread
            self.stop_event.set()
            # Button change to "Start"
            self.textstatus.set("Start")

    # Define a callback that update the plot
    def callback(self, buffer, n_values):
        # Copy the buffer
        data = buffer.copy()
        # Plot new data
        self.plotbuffer[:-n_values] = self.plotbuffer[n_values:]
        self.plotbuffer[-n_values:] = data
        self.linet[0].set_ydata(self.plotbuffer)
        # Draw figure
        self.figure_canvas.draw()

    def run(self):
        # Run measurment in thread
        work = self.NI.run
        # Argument : stop signal, no duration
        T = Thread(target=work, args=(self.stop_event, None))
        T.start()


if __name__ == "__main__":
    fs = 40000
    plot_time = 5
    refresh_delay = 0.2

    M = mp.Measurement(device_type="ni", in_sig=[mp.Signal(fs=fs)], dur=plot_time)

    with ni_callback_measurement(M) as NI:
        app = direct_plot(NI, refresh_delay)
        app.mainloop()
