import sounddevice as sd
from measpy.measurement import (Signal,
                    Spectral,
                    Measurement,
                    Weighting,
                    ms)

import numpy as np

import tkinter.ttk as ttk
import tkinter as tk

from datetime import datetime
from time import time, sleep

def run_audio_measurement(M,progress=True):
    if M.device_type!='audio':
        print("Warning: deviceType != 'audio'. Changing to 'audio'.")
        M.device_type='audio'
    if M.in_device=='':
        print("Warning: no device specified, changing to None")
        M.in_device=None
    if M.out_device=='':
        print("Warning: no device specified, changing to None")
        M.out_device=None
    now = datetime.now()
    M.date = now.strftime("%Y/%m/%d")
    M.time = now.strftime("%H:%M:%S")

    sd.default.device=(M.in_device,M.out_device)
    
    # Now done at initialization
    # M.create_output()

    if progress:
        root = tk.Tk()
        root.title('Recording...')
        # Progress bar widget
        progress = ttk.Progressbar(root, orient = tk.HORIZONTAL, 
                    length = 500, mode = 'determinate') 
        progress.pack(pady = 10) 
        start = time()

    if M.out_sig==None:
        y = sd.rec(int(M.dur * M.fs),
                    samplerate=M.fs,
                    mapping=M.in_map,
                    blocking=False)
    else:
        y = sd.playrec(M.x,
                    samplerate=M.fs,
                    input_mapping=M.in_map,
                    output_mapping=M.out_map,
                    blocking=False)     

    if progress:
        elapsed = time() - start
        durtot = M.dur+M.extrat[0]+M.extrat[1]
        while (elapsed)<durtot:
            sleep(0.1)
            elapsed = time() - start
            root.update_idletasks()
            root.update()
            progress['value'] = 100*elapsed/durtot

    sd.wait()

    if progress:
        root.destroy()
    
    for ii in range(len(M.in_map)):
        M.data[M.in_name[ii]].raw=np.array(y[:,ii],dtype=float)

Measurement.run_measurement=run_audio_measurement
