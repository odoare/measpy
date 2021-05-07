# __main__.py

import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    if len(sys.argv)>1:
        if sys.argv[1]=='audio':
            import measpy.measpyaudio as mp
        elif sys.argv[1]=='ni':
            import measpy.measpyni as mp
    M1 = mp.Measurement(out_sig='noise',
                    out_map=[1],
                    out_desc=['Out1'],
                    out_dbfs=[1.0],
                    in_map=[1,2],
                    in_desc=['In1','In2'],
                    in_cal=[1.0,1.0],
                    in_unit=['V','V'],
                    in_dbfs=[1.0,1.0],
                    extrat=[1.0,2.0],
                    out_sig_fades=[500,500],
                    dur=2)
    M1.run_measurement()
    M1.plot_with_cal()
    plt.show()

if __name__ == '__main__':
    main()
