# measpy/pico.py
#
# ---------------------------------------
# Data acquisition with Picoscope devices
# ---------------------------------------
#
# Part of measpy package for signal acquisition and processing
# (c) OD - 2021 - 2023
# https://github.com/odoare/measpy


import numpy as np

from datetime import datetime
from scipy.signal import decimate

import ctypes
import numpy as np
from picosdk.ps4000 import ps4000
from picosdk.functions import adc2mV, assert_pico_ok

from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.PicoDeviceEnums import picoEnum

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
#plt.style.use('seaborn-v0_8')

def findindex(l,e):
    try:
        a=l.index(e)
    except:
        a=None
    return a

def ps2000_run_measurement(M):
    """
    This function needs M to contain the following properties:
        - in_range : a list of strings specifying the voltage range.
            Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
            "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
        - upsampling_factor : upsampling factor
        - in_coupling : Coupling configuration of the channels.
            Can be "ac" or "dc"
    """

    import time

    CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
        None,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_int16,
        ctypes.c_uint32,
        ctypes.c_int16,
        ctypes.c_int16,
        ctypes.c_uint32
    )

    if M.device_type != 'pico':
        raise Exception("Error: device_type must be 'pico'.")

    if type(M.out_sig) != type(None):
        print('Warning: out_sig property ignored with picoscopes')

    # Effective sampling frequency
    # If upsampling_factor is > 1, the actual data acquisition is
    # performed at fs*upsampling_factor, and then decimated to
    # the desired frequency

    effective_fs = M.fs * M.upsampling_factor
    duree_ns = M.dur*1e9

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments

    si = round(1e9/effective_fs)
    if effective_fs != (1e9/si):
        effective_fs = (1e9/si)
        print("Warning : Sampling frequency fs changed to nearest possible value of "+str(effective_fs)+" Hz")
    
    print("Effective sampling frequency: "+str(effective_fs))

    sampleInterval = int(si)

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps2000.PS2000_VOLTAGE_RANGE['PS2000_'+M.in_range[indA]]
        adc_valuesA = []
        print('Channel A: enabled with range '+'PS2000_'+M.in_range[indA]+' ('+str(rangeA)+')')
        if M.in_coupling[indA].capitalize()=='Dc':
           couplingA = 'PICO_DC'
        elif M.in_coupling[indA].capitalize()=='Ac':
            couplingA = 'PICO_AC'
        else:
            print("Input A coupling not recognized, set to 'dc'")
            couplingA = 'PICO_DC'
            M.in_coupling[indA]='dc'
    else:
        enabledA = False
        rangeA = ps2000.PS2000_VOLTAGE_RANGE['PS2000_10V']
        print('Channel A: disabled')

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps2000.PS2000_VOLTAGE_RANGE['PS2000_'+M.in_range[indB]]
        adc_valuesB = []
        if M.in_coupling[indB].capitalize()=='Dc':
           couplingB = 'PICO_DC'
        elif M.in_coupling[indB].capitalize()=='Ac':
            couplingB = 'PICO_AC'
        else:
            print("Input B coupling not recognized, set to 'dc'")
            M.in_coupling[indB]='dc'
            couplingB = 'PICO_DC'
        print('Channel B: enabled with range '+'PS2000_'+M.in_range[indB]+' ('+str(rangeB)+')')
    else:
        enabledB = False
        rangeB = ps2000.PS2000_VOLTAGE_RANGE['PS2000_10V']
        couplingB = 'PICO_DC'
        print('Channel B: disabled')

    def get_overview_buffers(buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values):
        if enabledA:
            adc_valuesA.extend(buffers[0][0:n_values])
            if enabledB:
                adc_valuesB.extend(buffers[2][0:n_values])
        else:
            if enabledB:
                adc_valuesB.extend(buffers[2][0:n_values])

    callback = CALLBACK(get_overview_buffers)

    def adc_to_mv(values, range_, bitness=16):
        v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]
        return [(x * v_ranges[range_]) / (2**(bitness - 1) - 1) for x in values]

    with ps2000.open_unit() as device:
        print('Device info: {}'.format(device.info))

        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL['PICO_CHANNEL_A'],
            enabledA,
            picoEnum.PICO_COUPLING[couplingA],
            rangeA,
        )
        assert_pico2000_ok(res)        
        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL['PICO_CHANNEL_B'],
            enabledB,
            picoEnum.PICO_COUPLING[couplingB],
            rangeB,
        )
        assert_pico2000_ok(res)

        ########## Signal generator ##########
        if M.sig_gen:
            res = ps2000.ps2000_set_sig_gen_built_in(
                device.handle,
                int(M.offset*1e6),      # offset voltage in uV
                int(M.amp/2*1e6),       # peak-to-peak votage in uV
                M.wave,                 # type of waveform (0 = sine wave)
                M.freq_start,           # start frequency in Hz
                M.freq_stop,            # stop frequency in Hz
                M.freq_change,          # frequency change per interval in Hz
                M.freq_int,             # interval of frequency change in seconds
                M.sweep_dir,            # sweep direction (0 = up)
                M.sweep_number          # number of times to sweep
            )
            assert_pico2000_ok(res)
        #######################################

        res = ps2000.ps2000_run_streaming_ns(
            device.handle,
            sampleInterval,
            2,
            100_000,
            False,
            1,
            20_000
        )
        assert_pico2000_ok(res)

        now = datetime.now()
        M.date = now.strftime("%Y-%m-%d")
        M.time = now.strftime("%H:%M:%S")

        print('Start')
        start_time = time.time_ns()

        while time.time_ns() - start_time < duree_ns*1.1:
            ps2000.ps2000_get_streaming_last_values(
                device.handle,
                callback
            )

        print('Done')

        if M.fs!=effective_fs/M.upsampling_factor:
            M.fs = effective_fs/M.upsampling_factor
            print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
            for i in range(len(M.in_map)):
                M.in_sig[i].fs = M.fs

        for i in range(len(M.in_map)):
            if M.in_map[i] == 1:
                if M.upsampling_factor > 1:
                    M.in_sig[i].raw = decimate(np.double(adc_to_mv(adc_valuesA, rangeA)[:])/1000,M.upsampling_factor,ftype='fir')[0:int(round(M.dur*M.fs))]
                else:
                    M.in_sig[i].raw = (np.double(adc_to_mv(adc_valuesA, rangeA)[:])/1000)[0:int(round(M.dur*M.fs))]
            elif M.in_map[i] == 2:
                if M.upsampling_factor > 1:
                    M.in_sig[i].raw = decimate(np.double(adc_to_mv(adc_valuesB, rangeB)[:])/1000,M.upsampling_factor,ftype='fir')[0:int(round(M.dur*M.fs))]
                else:
                    M.in_sig[i].raw = (np.double(adc_to_mv(adc_valuesB, rangeB)[:])/1000)[0:int(round(M.dur*M.fs))]

def ps2000_plot(M,plotbuffersize=2000,updatetime=0.1):
    """
    Realtime plot of signal and spectrum

    This function needs M to contain the following properties:
        - in_range : a list of strings specifying the voltage range.
            Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
            "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
        - upsampling_factor : upsampling factor
        - in_coupling : Coupling configuration of the channels.
            Can be "ac" or "dc"
    """

    import time

    global plotbuffer, ax, x, line, stop, timesincelastupdate, updtime

    updtime = updatetime

    timesincelastupdate = 0.0

    stop = False

    def fstop(event):
        global stop
        stop = True

    def tamp_plus(event):
        ax[0].set_ylim(np.array(ax[0].get_ylim())/2)
    def tamp_moins(event):
        ax[0].set_ylim(np.array(ax[0].get_ylim())*2)
    def famp_plus(event):
        ax[1].set_ylim(np.array(ax[1].get_ylim())/2)
    def famp_moins(event):
        ax[1].set_ylim(np.array(ax[1].get_ylim())*2)
    def on_close(event):
        global stop
        stop = True

    x = np.arange(plotbuffersize)
    plotbuffer = np.sin(x)
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    fig.subplots_adjust(bottom=0.2)
    ax[0].set_xlabel('Temps [s]',fontsize=15)
    ax[0].set_ylabel('Tension [V]',fontsize=15)
    ax[1].set_xlabel('Fréquence [Hz]',fontsize=15)
    ax[1].set_ylabel('Tension [V]',fontsize=15)
    ax[1].set_xlim([0.01,M.fs/2])
    ax[1].set_ylim([0,10])
    linet, = ax[0].plot(x/M.fs,plotbuffer)
    linef, = ax[1].plot(np.fft.fftfreq(n=plotbuffersize,d=1/M.fs),np.fft.fft(plotbuffer,norm='ortho'))
    axs = fig.add_axes([0.4, 0.01, 0.2, 0.075])
    bstop = Button(axs, 'Stop')
    bstop.on_clicked(fstop)
    atxp = fig.add_axes([0.02, 0.6, 0.04, 0.05])
    btplus = Button(atxp, '+')
    btplus.on_clicked(tamp_plus)
    atxm = fig.add_axes([0.02, 0.4, 0.04, 0.05])
    btmoins = Button(atxm, '-')
    btmoins.on_clicked(tamp_moins)
    afxp = fig.add_axes([0.92, 0.6, 0.04, 0.05])
    bfplus = Button(afxp, '+')
    bfplus.on_clicked(famp_plus)
    afxm = fig.add_axes([0.92, 0.4, 0.04, 0.05])
    bfmoins = Button(afxm, '-')
    bfmoins.on_clicked(famp_moins)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.pause(0.0001)

    CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
        None,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_int16,
        ctypes.c_uint32,
        ctypes.c_int16,
        ctypes.c_int16,
        ctypes.c_uint32
    )

    # Effective sampling frequency
    # If upsampling_factor is > 1, the actual data acquisition is
    # performed at fs*upsampling_factor, and then decimated to
    # the desired frequency

    effective_fs = M.fs * M.upsampling_factor
    duree_ns = M.dur*1e9

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments

    si = round(1e9/effective_fs)
    if effective_fs != (1e9/si):
        effective_fs = (1e9/si)
        print("Warning : Sampling frequency fs changed to nearest possible value of "+str(effective_fs)+" Hz")
    
    print("Effective sampling frequency: "+str(effective_fs))

    sampleInterval = int(si)

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps2000.PS2000_VOLTAGE_RANGE['PS2000_'+M.in_range[indA]]
        adc_valuesA = []
        print('Channel A: enabled with range '+'PS2000_'+M.in_range[indA]+' ('+str(rangeA)+')')
        if M.in_coupling[indA].capitalize()=='Dc':
           couplingA = 'PICO_DC'
        elif M.in_coupling[indA].capitalize()=='Ac':
            couplingA = 'PICO_AC'
        else:
            print("Input A coupling not recognized, set to 'dc'")
            couplingA = 'PICO_DC'
            M.in_coupling[indA]='dc'
    else:
        enabledA = False
        rangeA = ps2000.PS2000_VOLTAGE_RANGE['PS2000_10V']
        print('Channel A: disabled')

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps2000.PS2000_VOLTAGE_RANGE['PS2000_'+M.in_range[indB]]
        adc_valuesB = []
        if M.in_coupling[indB].capitalize()=='Dc':
           couplingB = 'PICO_DC'
        elif M.in_coupling[indB].capitalize()=='Ac':
            couplingB = 'PICO_AC'
        else:
            print("Input B coupling not recognized, set to 'dc'")
            M.in_coupling[indB]='dc'
            couplingB = 'PICO_DC'
        print('Channel B: enabled with range '+'PS2000_'+M.in_range[indB]+' ('+str(rangeB)+')')
    else:
        enabledB = False
        rangeB = ps2000.PS2000_VOLTAGE_RANGE['PS2000_10V']
        couplingB = 'PICO_DC'
        print('Channel B: disabled')

    def get_overview_buffers(buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values):
        global plotbuffer, timesincelastupdate, updtime
        timesincelastupdate += n_values/M.fs
        plotbuffer=np.roll(plotbuffer,int(-n_values),axis=0)
        plotbuffer[-n_values:] = (np.array(adc_to_mv(buffers[0][0:n_values],rangeA))/1000)[:]
        if timesincelastupdate > updtime:
            linet.set_ydata(plotbuffer)
            linef.set_ydata(np.abs(np.fft.fft(plotbuffer*np.hanning(plotbuffersize),norm='ortho')))
            plt.pause(0.0001)
            timesincelastupdate = 0.0
        # if enabledA:
        #     adc_valuesA.extend(buffers[0][0:n_values])
        #     if enabledB:
        #         adc_valuesB.extend(buffers[2][0:n_values])
        # else:
        #     if enabledB:
        #         adc_valuesB.extend(buffers[2][0:n_values])

    callback = CALLBACK(get_overview_buffers)

    def adc_to_mv(values, range_, bitness=16):
        v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]
        return [(x * v_ranges[range_]) / (2**(bitness - 1) - 1) for x in values]

    with ps2000.open_unit() as device:
        print('Device info: {}'.format(device.info))

        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL['PICO_CHANNEL_A'],
            enabledA,
            picoEnum.PICO_COUPLING[couplingA],
            rangeA,
        )
        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL['PICO_CHANNEL_B'],
            enabledB,
            picoEnum.PICO_COUPLING[couplingB],
            rangeB,
        )
        assert_pico2000_ok(res)

        res = ps2000.ps2000_run_streaming_ns(
            device.handle,
            sampleInterval,
            2,
            100_000,
            False,
            1,
            20_000
        )
        assert_pico2000_ok(res)

        now = datetime.now()
        M.date = now.strftime("%Y-%m-%d")
        M.time = now.strftime("%H:%M:%S")

        print('Start')
        start_time = time.time_ns()

        while (time.time_ns() - start_time < duree_ns*1.1) and (stop==False):
            ps2000.ps2000_get_streaming_last_values(
                device.handle,
                callback
            )

        print('Done')

        # if M.fs!=effective_fs/M.upsampling_factor:
        #     M.fs = effective_fs/M.upsampling_factor
        #     print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
        #     for i in range(len(M.in_map)):
        #         M.data[M.in_name[i]].fs = M.fs

        # for i in range(len(M.in_map)):
        #     if M.in_map[i] == 1:
        #         if M.upsampling_factor > 1:
        #             M.data[M.in_name[i]].raw = decimate(np.double(adc_to_mv(adc_valuesA, rangeA)[:])/1000,M.upsampling_factor,ftype='fir')[0:int(round(M.dur*M.fs))]
        #         else:
        #             M.data[M.in_name[i]].raw = (np.double(adc_to_mv(adc_valuesA, rangeA)[:])/1000)[0:int(round(M.dur*M.fs))]
        #     elif M.in_map[i] == 2:
        #         if M.upsampling_factor > 1:
        #             M.data[M.in_name[i]].raw = decimate(np.double(adc_to_mv(adc_valuesB, rangeB)[:])/1000,M.upsampling_factor,ftype='fir')[0:int(round(M.dur*M.fs))]
        #         else:
        #             M.data[M.in_name[i]].raw = (np.double(adc_to_mv(adc_valuesB, rangeB)[:])/1000)[0:int(round(M.dur*M.fs))]

def ps4000_plot(M,plotbuffersize=2000,updatetime=0.1):
    """
    This function needs M to contain the following properties:
        - in_range : a list of strings specifying the voltage range.
            Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
            "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
        - upsampling_factor : upsampling factor
        - in_coupling : Coupling configuration of the channels.
            Can be "ac" or "dc"
    """

    import time


    # Plotting part

    global plotbuffer, ax, x, line, stop, timesincelastupdate, updtime

    updtime = updatetime

    timesincelastupdate = 0.0

    stop = False

    def fstop(event):
        global stop
        stop = True

    def tamp_plus(event):
        ax[0].set_ylim(np.array(ax[0].get_ylim())/2)
    def tamp_moins(event):
        ax[0].set_ylim(np.array(ax[0].get_ylim())*2)
    def famp_plus(event):
        ax[1].set_ylim(np.array(ax[1].get_ylim())/2)
    def famp_moins(event):
        ax[1].set_ylim(np.array(ax[1].get_ylim())*2)
    def on_close(event):
        global stop
        stop = True

    x = np.arange(plotbuffersize)
    plotbuffer = np.sin(x)
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    fig.subplots_adjust(bottom=0.2)
    ax[0].set_xlabel('Temps [s]',fontsize=15)
    ax[0].set_ylabel('Tension [V]',fontsize=15)
    ax[1].set_xlabel('Fréquence [Hz]',fontsize=15)
    ax[1].set_ylabel('Tension [V]',fontsize=15)
    ax[1].set_xlim([0.01,M.fs/2])
    ax[1].set_ylim([0,10])
    linet, = ax[0].plot(x/M.fs,plotbuffer)
    linef, = ax[1].plot(np.fft.fftfreq(n=plotbuffersize,d=1/M.fs),np.fft.fft(plotbuffer,norm='ortho'))
    axs = fig.add_axes([0.4, 0.01, 0.2, 0.075])
    bstop = Button(axs, 'Stop')
    bstop.on_clicked(fstop)
    atxp = fig.add_axes([0.02, 0.6, 0.04, 0.05])
    btplus = Button(atxp, '+')
    btplus.on_clicked(tamp_plus)
    atxm = fig.add_axes([0.02, 0.4, 0.04, 0.05])
    btmoins = Button(atxm, '-')
    btmoins.on_clicked(tamp_moins)
    afxp = fig.add_axes([0.92, 0.6, 0.04, 0.05])
    bfplus = Button(afxp, '+')
    bfplus.on_clicked(famp_plus)
    afxm = fig.add_axes([0.92, 0.4, 0.04, 0.05])
    bfmoins = Button(afxm, '-')
    bfmoins.on_clicked(famp_moins)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.pause(0.0001)

    # End of plotting part

    # Find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16(32767)

    global nextSample, autoStopOuter, wasCalledBack

    # Buffer size fixed to 20k samples
    sizeOfOneBuffer = 20_000

    # Effective sampling frequency
    # If upsampling is > 1, the actual data acquisition is
    # performed at fs*upsampling, and then decimated to
    # the desired frequency
    effective_fs = M.fs * M.upsampling_factor

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments
    si = round(1e9/effective_fs)
    if effective_fs != (1e9/si):
        effective_fs = (1e9/si)
        print("Warning : Sampling frequency fs changed to nearest possible value of "+str(effective_fs)+" Hz")
    
    print("Effective sampling frequency: "+str(effective_fs))

    numdesiredsamples = int(round(effective_fs*M.dur))
    numBuffersToCapture = int(np.ceil(numdesiredsamples/sizeOfOneBuffer))
    sampleInterval = ctypes.c_int32(si)
    totalSamples = sizeOfOneBuffer * numBuffersToCapture
    chandle = ctypes.c_int16()
    status = {}

    # Open PicoScope 4000 Series device
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps4000.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indA]]
        if M.in_coupling[indA].capitalize()=='Dc':
           couplingA = 1
        elif M.in_coupling[indA].capitalize()=='Ac':
            couplingA = 0
        else:
            print("Input A coupling not recognized, set to 'dc'")
            couplingA = 1
            M.in_coupling[indA]='dc'
        # Create buffers ready for assigning pointers for data collection
        bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel A: enabled with range '+'PS4000_'+M.in_range[indA]+' ('+str(rangeA)+')')
    else:
        enabledA = False
        rangeA = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel A: disabled')

    # Set up channel A
    channel_range = ps4000.PS4000_RANGE['PS4000_50MV']
    status["setChA"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                            int(enabledA),
                                            couplingA,
                                            rangeA)
    assert_pico_ok(status["setChA"])

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indB]]
        # Create buffers ready for assigning pointers for data collection
        bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel B: enabled with range '+'PS4000_'+M.in_range[indB]+' ('+str(rangeB)+')')
        if M.in_coupling[indB].capitalize()=='Dc':
           couplingB = 1
        elif M.in_coupling[indB].capitalize()=='Ac':
            couplingB = 0
        else:
            print("Input B coupling not recognized, set to 'dc'")
            couplingB = 1
            M.in_coupling[indB]='dc'
    else:
        enabledB = False
        couplingB = 1
        rangeB = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel B: disabled')

    # Set up channel B
    status["setChB"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                            int(enabledB),
                                            couplingB,
                                            rangeB)
    assert_pico_ok(status["setChB"])

    # Set data buffer location for data collection from channel A
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_A = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledA:
        status["setDataBuffersA"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                                        bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersA"])

    # Set data buffer location for data collection from channel B
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_B = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledB:
        status["setDataBuffersB"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                                        bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersB"])

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Begin streaming mode:
    sampleUnits = ps4000.PS4000_TIME_UNITS['PS4000_NS']
    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1
    # No downsampling:
    downsampleRatio = 1
    status["runStreaming"] = ps4000.ps4000RunStreaming(chandle,
                                                    ctypes.byref(sampleInterval),
                                                    sampleUnits,
                                                    maxPreTriggerSamples,
                                                    totalSamples,
                                                    autoStopOn,
                                                    downsampleRatio,
                                                    sizeOfOneBuffer)
    assert_pico_ok(status["runStreaming"])

    actualSampleInterval = sampleInterval.value
    actualSampleIntervalNs = actualSampleInterval * 1000

    print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False

    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):

        global plotbuffer, timesincelastupdate, updtime
        global nextSample, autoStopOuter, wasCalledBack
        wasCalledBack = True
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        timesincelastupdate += noOfSamples/M.fs
        plotbuffer=np.roll(plotbuffer,int(-noOfSamples),axis=0)
        plotbuffer[-noOfSamples:] = (np.array(adc2mV(bufferAMax[startIndex:sourceEnd],rangeA,maxADC))/1000)
        if timesincelastupdate > updtime:
            linet.set_ydata(plotbuffer)
            linef.set_ydata(np.abs(np.fft.fft(plotbuffer*np.hanning(plotbuffersize),norm='ortho')))
            plt.pause(0.0001)
            timesincelastupdate = 0.0
        # if enabledA:
        #     bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
        # if enabledB:
        #     bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
        nextSample += noOfSamples
        if autoStop:
            autoStopOuter = True

    # Convert the python function into a C function pointer.
    cFuncPtr = ps4000.StreamingReadyType(streaming_callback)

    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples and not autoStopOuter and not stop:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps4000.ps4000GetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    print("Done.")

    # # Convert ADC counts data to mV
    # if enabledA:
    #     adc2mVChAMax = adc2mV(bufferCompleteA, rangeA, maxADC)
    # if enabledB:
    #     adc2mVChBMax = adc2mV(bufferCompleteB, rangeB, maxADC)

    # # Create time data
    # time = np.linspace(0, (totalSamples - 1) * actualSampleIntervalNs, totalSamples)

    # Stop the scope
    # handle = chandle
    status["stop"] = ps4000.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    # handle = chandle
    status["close"] = ps4000.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    # for i in range(len(M.in_map)):
    #     if M.in_map[i] == 1:
    #         M.data[M.in_name[i]].raw = decimate(np.double(adc2mVChAMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)
    #     elif M.in_map[i] == 2:
    #         M.data[M.in_name[i]].raw = decimate(np.double(adc2mVChBMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)

    # if M.fs!=effective_fs/M.upsampling_factor:
    #     M.fs = effective_fs/M.upsampling_factor
    #     print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
    #     for i in range(len(M.in_map)):
    #         M.data[M.in_name[i]].fs = M.fs
    

def ps4000_run_measurement(M):
    """
    This function needs M to contain the following properties:
        - in_range : a list of strings specifying the voltage range.
            Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
            "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
        - upsampling_factor : upsampling factor
        - in_coupling : Coupling configuration of the channels.
            Can be "ac" or "dc"
    """

    import time
    global nextSample, autoStopOuter, wasCalledBack

    # Buffer size fixed to 20k samples
    sizeOfOneBuffer = 100_000

    # Effective sampling frequency
    # If upsampling is > 1, the actual data acquisition is
    # performed at fs*upsampling, and then decimated to
    # the desired frequency
    effective_fs = M.fs * M.upsampling_factor

    # Sample interval is the duration between two consecutive samples
    # As it is the sampling frequency that is specified,
    # and the sampling interval can only take integer increments
    # of 1 nanoseconds, the actual sampling frequency might
    # necessitate adjustments
    si = round(1e9/effective_fs)
    if effective_fs != (1e9/si):
        effective_fs = (1e9/si)
        print("Warning : Sampling frequency fs changed to nearest possible value of "+str(effective_fs)+" Hz")
    
    print("Effective sampling frequency: "+str(effective_fs))

    numdesiredsamples = int(round(effective_fs*M.dur))
    numBuffersToCapture = int(np.ceil(numdesiredsamples/sizeOfOneBuffer))
    sampleInterval = ctypes.c_int32(si)
    totalSamples = sizeOfOneBuffer * numBuffersToCapture
    chandle = ctypes.c_int16()
    status = {}

    # Open PicoScope 4000 Series device
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps4000.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indA]]
        if M.in_coupling[indA].capitalize()=='Dc':
           couplingA = 1
        elif M.in_coupling[indA].capitalize()=='Ac':
            couplingA = 0
        else:
            print("Input A coupling not recognized, set to 'dc'")
            couplingA = 1
            M.in_coupling[indA]='dc'
        # Create buffers ready for assigning pointers for data collection
        bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel A: enabled with range '+'PS4000_'+M.in_range[indA]+' ('+str(rangeA)+')')
    else:
        enabledA = False
        rangeA = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel A: disabled')

    # Set up channel A
    channel_range = ps4000.PS4000_RANGE['PS4000_50MV']
    status["setChA"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                            int(enabledA),
                                            couplingA,
                                            rangeA)
    assert_pico_ok(status["setChA"])

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps4000.PS4000_RANGE['PS4000_'+M.in_range[indB]]
        # Create buffers ready for assigning pointers for data collection
        bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        # Total buffer
        bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel B: enabled with range '+'PS4000_'+M.in_range[indB]+' ('+str(rangeB)+')')
        if M.in_coupling[indB].capitalize()=='Dc':
           couplingB = 1
        elif M.in_coupling[indB].capitalize()=='Ac':
            couplingB = 0
        else:
            print("Input B coupling not recognized, set to 'dc'")
            couplingB = 1
            M.in_coupling[indB]='dc'
    else:
        enabledB = False
        rangeB = ps4000.PS4000_RANGE['PS4000_10V']
        print('Channel B: disabled')

    # Set up channel B
    status["setChB"] = ps4000.ps4000SetChannel(chandle,
                                            ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                            int(enabledB),
                                            couplingB,
                                            rangeB)
    assert_pico_ok(status["setChB"])

    # Set data buffer location for data collection from channel A
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_A = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledA:
        status["setDataBuffersA"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                                        bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersA"])

    # Set data buffer location for data collection from channel B
    # Parameters :  handle = chandle
    #               source = PS4000_CHANNEL_B = 0
    #               pointer to buffer max = ctypes.byref(bufferAMax)
    #               pointer to buffer min = ctypes.byref(bufferAMin)
    #               buffer length = maxSamples
    #               segment index = 0
    #               ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledB:
        status["setDataBuffersB"] = ps4000.ps4000SetDataBuffers(chandle,
                                                        ps4000.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                                        bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersB"])

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Begin streaming mode:
    sampleUnits = ps4000.PS4000_TIME_UNITS['PS4000_NS']
    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1
    # No downsampling:
    downsampleRatio = 1
    status["runStreaming"] = ps4000.ps4000RunStreaming(chandle,
                                                    ctypes.byref(sampleInterval),
                                                    sampleUnits,
                                                    maxPreTriggerSamples,
                                                    totalSamples,
                                                    autoStopOn,
                                                    downsampleRatio,
                                                    sizeOfOneBuffer)
    assert_pico_ok(status["runStreaming"])

    actualSampleInterval = sampleInterval.value
    actualSampleIntervalNs = actualSampleInterval * 1000

    print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False

    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        global nextSample, autoStopOuter, wasCalledBack
        wasCalledBack = True
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        if enabledA:
            bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
        if enabledB:
            bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
        nextSample += noOfSamples
        if autoStop:
            autoStopOuter = True
            print("auto")

    # if M.sig_gen:
    #     res=ps4000.ps4000SetSigGenBuiltIn(chandle,
    #             int(M.offset*1e6),      # offset voltage in uV
    #             int(M.amp/2*1e6),       # peak-to-peak votage in uV
    #             M.wave,                 # type of waveform (0 = sine wave)
    #             M.freq_start,           # start frequency in Hz
    #             M.freq_stop,            # stop frequency in Hz
    #             M.freq_change,          # frequency change per interval in Hz
    #             M.freq_int,             # interval of frequency change in seconds
    #             M.sweep_dir,            # sweep direction (0 = up)
    #             0,
    #             1,
    #             0,
    #             0,
    #             0,
    #             0
    #         )
        
    #     print("config")
    #     assert_pico_ok(res)
    #     print("fin assert")
    # print("suite")

    # Convert the python function into a C function pointer.
    cFuncPtr = ps4000.StreamingReadyType(streaming_callback)

    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples:  #and not autoStopOuter:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps4000.ps4000GetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    print("Done.")

    # Find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16(32767)

    # Convert ADC counts data to mV
    if enabledA:
        adc2mVChAMax = adc2mV(bufferCompleteA, rangeA, maxADC)
    if enabledB:
        adc2mVChBMax = adc2mV(bufferCompleteB, rangeB, maxADC)

    # Create time data
    time = np.linspace(0, (totalSamples - 1) * actualSampleIntervalNs, totalSamples)

    # Stop the scope
    # handle = chandle
    status["stop"] = ps4000.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    # handle = chandle
    status["close"] = ps4000.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    for i in range(len(M.in_map)):
        if M.in_map[i] == 1:
            M.in_sig[i].raw = decimate(np.double(adc2mVChAMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)
        elif M.in_map[i] == 2:
            M.in_sig[i].raw = decimate(np.double(adc2mVChBMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)

    if M.fs!=effective_fs/M.upsampling_factor:
        M.fs = effective_fs/M.upsampling_factor
        print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
        for i,s in enumerate(M.in_sig):
            s.fs = M.fs
    