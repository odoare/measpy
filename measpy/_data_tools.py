# measpy/_data_tools.py
#
# --------------------------------------------
# Utilities for data management in threading
# --------------------------------------------

from threading import Thread
from queue import Queue
import numpy as np

maxtimeout = 10

def use_queues(min_chunksize_processed):
    """
    decorator that transform f(list)->list into f(queue)->queue
    :param min_chunksize_processed: minimum number of data point cumulated before applying f
    :type min_chunksize_processed: int

    """
    if min_chunksize_processed>0:
        def decorator(f):
            def wrap(queue_in,queue_out,*args, **kwargs):
                chunk_data = queue_in.get(timeout=maxtimeout)
                while (item := queue_in.get(timeout=maxtimeout)) is not None:
                    if len(chunk_data[0]) > min_chunksize_processed:
                        queue_out.put(f(chunk_data, *args, **kwargs))
                        chunk_data = item
                    else:
                        try:
                            _ = [c.extend(it) for (c, it) in zip(chunk_data, item)]
                        except AttributeError: #numpy array
                            chunk_data = np.concatenate([chunk_data,item],axis = 1)
                if len(chunk_data[0]) > 0:
                    queue_out.put(f(chunk_data, *args, **kwargs))
                queue_out.put(None)
            return wrap
    else:
        def decorator(f):
            def wrap(queue_in,queue_out,*args, **kwargs):
                while (item := queue_in.get(timeout=maxtimeout)) is not None:
                    queue_out.put(f(item, *args, **kwargs))
                queue_out.put(None)
            return wrap
    return decorator

def dispatch(q_in, qs_out):
    """
    Dispatch one queue into a list of queue
    :param q_in: Input data
    :type q_in: Queue.queue
    :param qs_out: Copies of input data
    :type qs_out: list of Queue.queue

    """
    while (item := q_in.get(timeout=maxtimeout)) is not None:
        for q_out in qs_out:
            q_out.put(item)
    for q_out in qs_out:
        q_out.put(None)

def setup_threads(queue_in, process_dict):
    """
    Create a list of thread using methods inside a dictionnary
    :param queue_in: queue that contain datas
    :type queue_in: queue.Queue()
    :param process_dict: Dictionnary that contain methods and represent a pipeline
    example : process_dict = {Method1(queue)->None:None, Method2(queue_in)->queue_out:process_dict2, Queue_out:None}
    apply Method1 and Method2 with argument queue_in; copy queue_in into Queue_out and 
    use the queue_out from Method2 to continue a pipeline defined with process_dict2
    :type process_dict: dict
    :return: List of threading.Thread ordered according to process_dict structure
    :rtype: List
    """

    ret = []
    if len(process_dict)>1:
        queues = []
        for targ in process_dict.keys():
            if isinstance(targ, Queue):
                queues.append(targ)
            else:
                queues.append(Queue())
        ret.extend([Thread(target=dispatch,args=(queue_in, queues))])  
    else:
        queues = [queue_in]

    for (targ, NEXT), Q in zip(process_dict.items(),queues):
        if NEXT is None:
            if callable(targ):
                ret.extend([Thread(target=targ,args=(Q,))])
        else:
            if not len(NEXT)>1 and isinstance((Next_targ := next(iter(NEXT))), Queue):
                queue_out = Next_targ
            else:
                queue_out = Queue()
            ret.extend([Thread(target=targ,args=(Q,queue_out))])
            ret.extend(setup_threads(queue_out, NEXT))
    return ret

class Pipeline_manager(Thread):
    def __init__(self, queue_in, process_dict):
        super().__init__()
        self.queue_in = queue_in
        self.process_dict = process_dict
        self.process = setup_threads(self.queue_in, process_dict)

    def run(self):
        for P in self.process:
            P.start()
        print("Data processing...")
        for P in self.process:
            P.join()
        print("Data processing finished")


class Process_manager(Pipeline_manager):
    def __init__(self,
                 queue_in,
                 Raw_output=None,
                 Data_process=None,
                 Processed_output=None,
                 min_chunksize_processed=0,
                 ):
        """
        :param queue_in: Queue containing raw data
        :type queue_in: queue.Queue()
        :param Raw_output: methods to save raw data into container, defaults to None
        :type Raw_output: method(queue.Queue()) or list of methods(queue.Queue()), optional
        :param Data_process: Method(list)->list that transform raw data, defaults to None
        :type Data_process: method, optional
        :param Processed_output: methods to save processed data into container, defaults to None
        :type Processed_output: method(queue.Queue()) or list of methods(queue.Queue()), optional

        """
        self.Raw_output=Raw_output
        self.Data_process=Data_process
        self.Processed_output=Processed_output
        self.min_chunksize_processed=min_chunksize_processed
        process_dict = self.create_process_dict()
        super().__init__(queue_in, process_dict)
       
    def create_process_dict(self):
        process_dict = {}
        if self.Raw_output is not None:
            if isinstance(self.Raw_output, list):
                process_dict.update({Raw_output: None for Raw_output in self.Raw_output if Raw_output is not None})
            elif self.Raw_output is not None:
                process_dict.update({self.Raw_output:None})

        data_output = {}
        if self.Processed_output is not None:
            if isinstance(self.Processed_output, list):
                data_output.update({Processed_output: None for Processed_output in self.Processed_output if Processed_output is not None})
            elif self.Processed_output is not None:
                data_output.update({self.Processed_output: None})

        if self.Data_process is not None:
            process_dict.update({self.Process_queue():data_output})
        else:
            process_dict.update(data_output)
        return process_dict
            
    def Process_queue(self):
        return use_queues(self.min_chunksize_processed)(self.Data_process)

def Queue2prealocated_array(q_in, array):
    datasize = array.shape[0]
    nextSample = 0
    while (item := q_in.get(timeout=maxtimeout)) is not None:
        noOfSamples = item.shape[0]
        lastsample = nextSample + noOfSamples
        try:
            array[nextSample:lastsample] = item
            nextSample = lastsample
        except ValueError:
            N = datasize - nextSample
            if N > 0:
                array[nextSample:] = item[:N]
            break
    return array


def Queue2array(q_in):
    data = q_in.get(timeout=maxtimeout)
    while (item := q_in.get(timeout=maxtimeout)) is not None:
        data =  np.concatenate([data,item],axis = 0)
    return data