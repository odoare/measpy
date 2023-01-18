import sounddevice as sd

import measpy.signal as ms

import numpy as np
from numpy.matlib import repmat

import tkinter.ttk as ttk
import tkinter as tk

from datetime import datetime
from time import time, sleep

def audio_run_measurement(M,progress=True):
    if M.device_type!='audio':
        print("Warning: deviceType != 'audio'. Changing to 'audio'.")
        M.device_type='audio'
    if M.in_device=='':
        print("Warning: no device specified, changing to None")
        M.in_device=None
    if M.out_sig!=None:
        if M.out_device=='':
            print("Warning: no device specified, changing to None")
            M.out_device=None

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Set the audio devices to use
    if M.out_sig!=None:
        sd.default.device=(M.in_device,M.out_device)
    else:
        sd.default.device=M.in_device

    # Insert a synchronization peak at the begining of the output signals
    if M.out_sig==None:
        dursync=0
        effsync=False
    elif M.io_sync>0:
        if M.io_sync in M.in_map:
            nout = M.x_raw.shape[1]
            peaks = repmat(ms.picv(M.fs),nout,1).T
            zers = repmat(np.zeros(int(M.fs)),nout,1).T
            outx = np.block([[peaks],[M.x_raw],[zers]])
            effsync = True
            dursync=4
            indsearch=M.in_map.index(M.io_sync)
        else:
            print('io_sync channel not present in in_map, no sync is done')
            outx=M.x_raw
            dursync=0
            effsync=False
    else:
        outx=M.x_raw
        dursync=0
        effsync=False
        
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
        y = sd.playrec(outx,
                    samplerate=M.fs,
                    input_mapping=M.in_map,
                    output_mapping=M.out_map,
                    blocking=False)

    if progress:
        elapsed = time() - start
        durtot = M.dur+M.extrat[0]+M.extrat[1]+dursync
        while (elapsed)<durtot:
            sleep(0.1)
            elapsed = time() - start
            root.update_idletasks()
            root.update()
            progress['value'] = 100*elapsed/durtot

    sd.wait()

    if progress:
        root.destroy()

    if effsync:
        posmax = int( np.argmax(y[int(0.25*M.fs*2):int(0.75*M.fs*2),indsearch]) + 0.75*M.fs*2 )
        print(posmax)
        y = y[posmax:posmax+M.fs*M.dur,:]

    for ii in range(len(M.in_map)):
        M.data[M.in_name[ii]].raw=np.array(y[:,ii],dtype=float)

def audio_get_devices():
   return sd.query_devices()
 