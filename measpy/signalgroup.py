known_group_len = {
    'Stereo' : 2,
    'Binau' : 2,
    'XY' : 2,
    'AmbiA' : 4,
    'AmbiB' : 4,
    '5.1' : 6
    }

# known_functions = ['iir','rms_smooth']

from .signal import Signal
import copy
from types import SimpleNamespace

class Signal_group:
    def __init__(self,**kwargs):

        # if 'sigs' not in kwargs:
        #     raise Exception('No signal list provided')
        if 'grouptype' in kwargs:
            self.grouptype = kwargs['grouptype']
            if kwargs['grouptype'] in known_group_len:
                if 'sigs' in kwargs:
                    if known_group_len[kwargs['grouptype']]!=len(kwargs['sigs']):
                        raise Exception('Number of signals in siglist does not correspond to group type "'+kwargs['grouptype']+'"')
        else:
            self.grouptype=None
    
        if 'sigs' in kwargs:
            self.sigs = kwargs['sigs']
        else:
            if 'grouptype' in kwargs:
                if kwargs['grouptype'] in known_group_len:
                    self.sigs = list(Signal() for i in range(known_group_len[kwargs['grouptype']]))
                elif 'ns' in kwargs:
                    self.sigs = list(Signal() for i in range(kwargs['ns']))
                else:
                    self.sigs={}

        # funcs = {}
        # for i,f in enumerate(known_functions):
        #     # print(f)
        #     # a = copy.deepcopy(f)
        #     funcs[known_functions[i]] = staticmethod(lambda **kwargs : self.apply_function(known_functions[i],**kwargs))
        #     setattr(self,known_functions[i], funcs[known_functions[i]])
        #     setattr(self,known_functions[i]+"_name",known_functions[i])

    def apply_function(self,func,**kwargs):
        out = copy.deepcopy(self)
        for i,s in enumerate(self.sigs):
            fun = getattr(Signal,func)
            out.sigs[i] = fun(s,**kwargs)
        return out

