import measpy.signal as ms

import numpy as np
from numpy.matlib import repmat

from datetime import datetime

import ctypes
import numpy as np
from picosdk.ps4000 import ps4000 as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
from scipy.signal import decimate

def ps4000_run_measurement(M):
    """
    This function needs M to contain the following properties:
        - in_range : a list of strings specifying the voltage range.
            Possible voltage ranges are "10MV", "20MV", "50MV", "100MV",
            "200MV", "500MV", "1V", "2V", "5V", "10V", "20V", "50V", "100V"
        - upsampling_factor : upsampling factor
    """

    import time
    global nextSample, autoStopOuter, wasCalledBack
        
    def findindex(l,e):
        try:
            a=l.index(e)
        except:
            a=None
        return a

    # Buffer size fixed to 20k samples
    sizeOfOneBuffer = 100_000

    # Effective sampling frequency (if upsampling is made)
    effective_fs = M.fs * M.upsampling_factor

    # Sample interval
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
    status["openunit"] = ps.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])

    enabled = 1
    disabled = 0
    analogue_offset = 0.0

    # Setup channel A
    indA = findindex(M.in_map,1)
    if indA!=None:
        enabledA = True
        rangeA = ps.PS4000_RANGE['PS4000_'+M.in_range[indA]]
        # Create buffers ready for assigning pointers for data collection
        bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel A: enabled with range '+'PS4000_'+M.in_range[indA]+' ('+str(rangeA)+')')
    else:
        enabledA = False
        rangeA = ps.PS4000_RANGE['PS4000_10V']
        print('Channel A: disabled')

    # Set up channel A
    channel_range = ps.PS4000_RANGE['PS4000_50MV']
    status["setChA"] = ps.ps4000SetChannel(chandle,
                                            ps.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                            int(enabledA),
                                            1,
                                            rangeA)
    assert_pico_ok(status["setChA"])

    # Setup channel B
    indB = findindex(M.in_map,2)
    if indB!=None:
        enabledB = True
        rangeB = ps.PS4000_RANGE['PS4000_'+M.in_range[indB]]
        # Create buffers ready for assigning pointers for data collection
        bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
        print('Channel B: enabled with range '+'PS4000_'+M.in_range[indB]+' ('+str(rangeB)+')')
    else:
        enabledB = False
        rangeB = ps.PS4000_RANGE['PS4000_10V']
        print('Channel B: disabled')

    # Set up channel B
    status["setChB"] = ps.ps4000SetChannel(chandle,
                                            ps.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                            int(enabledB),
                                            1,
                                            rangeB)
    assert_pico_ok(status["setChB"])

    memory_segment = 0

    # Set data buffer location for data collection from channel A
    # handle = chandle
    # source = PS4000_CHANNEL_A = 0
    # pointer to buffer max = ctypes.byref(bufferAMax)
    # pointer to buffer min = ctypes.byref(bufferAMin)
    # buffer length = maxSamples
    # segment index = 0
    # ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledA:
        status["setDataBuffersA"] = ps.ps4000SetDataBuffers(chandle,
                                                        ps.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                                                        bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersA"])

    # Set data buffer location for data collection from channel B
    # handle = chandle
    # source = PS4000_CHANNEL_B = 1
    # pointer to buffer max = ctypes.byref(bufferBMax)
    # pointer to buffer min = ctypes.byref(bufferBMin)
    # buffer length = maxSamples
    # segment index = 0
    # ratio mode = PS4000_RATIO_MODE_NONE = 0
    if enabledB:
        status["setDataBuffersB"] = ps.ps4000SetDataBuffers(chandle,
                                                        ps.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                                                        bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        None,
                                                        sizeOfOneBuffer)
        assert_pico_ok(status["setDataBuffersB"])

    now = datetime.now()
    M.date = now.strftime("%Y-%m-%d")
    M.time = now.strftime("%H:%M:%S")

    # Begin streaming mode:
    sampleUnits = ps.PS4000_TIME_UNITS['PS4000_NS']
    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1
    # No downsampling:
    downsampleRatio = 1
    status["runStreaming"] = ps.ps4000RunStreaming(chandle,
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

    # We need a big buffer, not registered with the driver, to keep our complete capture in.
    #bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
    #bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
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


    # Convert the python function into a C function pointer.
    cFuncPtr = ps.StreamingReadyType(streaming_callback)

    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples and not autoStopOuter:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps.ps4000GetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    print("Done grabbing values.")

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
    status["stop"] = ps.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    # handle = chandle
    status["close"] = ps.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    for i in range(len(M.in_map)):
        if M.in_map[i] == 1:
            M.data[M.in_name[i]].raw = decimate(np.double(adc2mVChAMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)
        elif M.in_map[i] == 2:
            M.data[M.in_name[i]].raw = decimate(np.double(adc2mVChBMax[0:round(M.dur*effective_fs)])/1000,M.upsampling_factor)

    if M.fs!=effective_fs/M.upsampling_factor:
        M.fs = effective_fs/M.upsampling_factor
        print('Warning : Sampling frequency fs changed to nearest possible value of '+str(M.fs)+' Hz')
        for i in range(len(M.in_map)):
            M.data[M.in_name[i]].fs = M.fs
    