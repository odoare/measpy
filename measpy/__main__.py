# __main__.py

import matplotlib.pyplot as plt
import numpy as np
import sys
import measpy as mp

def main():
    if len(sys.argv)>1:
        if sys.argv[1]=='audio':
            from measpy.audio import audio_run_measurement as run_measurement
        elif sys.argv[1]=='ni':
            from measpy.ni import ni_run_measurement as run_measurement
        elif sys.argv[1]=='ps2000':
            from measpy.pico import ps2000_run_measurement as run_measurement
        elif sys.argv[1]=='ps4000':
            from measpy.pico import ps4000_run_measurement as run_measurement
    M1 = mp.Measurement(out_sig=None,
                    in_map=[1,2],
                    in_desc=['In1','In2'],
                    in_cal=[1.0,1.0],
                    in_unit=['V','V'],
                    in_dbfs=[1.0,1.0],
                    extrat=[0,0],
                    dur=2)
    run_measurement(M1)
    M1.plot()
    M1.to_csvwav(str(sys.argv[3]))

if __name__ == '__main__':
    main()
