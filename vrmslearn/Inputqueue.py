#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Input queue to feed example when training with Tensorflow
"""

from multiprocessing import Process, Queue, Event, Value
import queue

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

class DataGenerator(Process):
    """
    Created a new process that will generate new examples with the generator_fun
    and will put them in the data Queue data_q.
    """
    def __init__(self,
                 data_q: Queue,
                 stop_event: Event,
                 generator_fun=lambda x: 1):
        """
        This is the constructor for the class.

        @params:
        data_q (Queue): A Queue in which to put generated examples
        stop_event (Event): An Event to kill the DataGenerator process
        generator_fun (callable): A callable that generate the data. This could
                                  be for example a function that reads a file

        @returns:
        """
        super().__init__()
        self.done_q = data_q
        self.stop_event = stop_event
        self.generator_fun = generator_fun

    def run(self):
        """
        When called, starts the DataGenerator process. Stop by setting the stop
        event.

        @returns:
        """
        while not self.stop_event.is_set():
            if not self.done_q.full():
                try:
                    batch = self.generator_fun()
                    self.done_q.put(batch)
                except FileNotFoundError:
                    pass

class DataAggregator(Process):
    """
    Create a new process that will aggregate examples created by DataGenerator
    processes into a batch of examples
    """
    def __init__(self,
                 data_q: Queue,
                 batch_q: Queue,
                 stop_event: Event,
                 batch_size: int,
                 n_in_queue: Counter=None,
                 postprocess_fun = None):
        """
        This is the constructor for the class.

        @params:
        data_q (Queue): A Queue in which to get new examples
        batch_q (Queue): A Queue in which to put new batches of examples
        stop_event (Event): An Event to kill the DataAggregator process
        batch_size (int): The number of examples in a batch
        n_in_queue (Counter): A counter sharable accross processes counting the
                              number of batches
        max_capacity (int): Maximum number of batches in the queue

        @returns:
        """
        super().__init__()
        self.pending_q = data_q
        self.done_q = batch_q
        self.stop_event = stop_event
        self.batch_size = batch_size
        self.batch = []
        if n_in_queue is None:
            self.n_in_queue = Counter()
        else:
            self.n_in_queue = n_in_queue
        self.nsaved = 0
        self.postprocess_fun = postprocess_fun

    def run(self):
        """
        When called, starts the DataGenerator process. Stop by setting the stop
        event.

        @returns:
        """
        while not self.stop_event.is_set():
            if not self.done_q.full():
                batch = self.pending_q.get()
                self.batch.append(batch)
                
                if len(self.batch) == self.batch_size:
                    batch = self.batch
                    if self.postprocess_fun is not None:
                        batch = self.postprocess_fun(batch)
                    self.done_q.put(batch)
                    self.batch = []
                    self.n_in_queue.increment()

class BatchManager:
    """
    Creates an input queue for Tensorflow, managing example creation and
    examples aggregation on multiple processes.
    """
    def __init__(self,
                 MAX_CAPACITY: int=10,
                 batch_size: int=3,
                 generator_fun=[lambda: 1],
                 postprocess_fun=None,
                 timeout: int=360):
        """
        Creates the DataGenerator and DataAggregator processes and starts them.
        Use with a with statement, as it will close processes automatically.

        @params:
        MAX_CAPACITY (int): Maximum number of batches or examples in
                            DataGenerator and DataAggregator queues
        batch_size (int): The number of examples in a batch
        generator_fun (list): List of callables that generates an example. One
                              DataGenerator process per element in the list will
                              be created.
        timeout (int):      Maximum time to retrieve a batch. Default to 60s,
                            change if generating a batch takes longer.

        @returns:
        """
        self.timeout = timeout
        self.generator_fun = generator_fun
        self.MAX_CAPACITY = MAX_CAPACITY
        self.batch_size = batch_size
        self.postprocess_fun = postprocess_fun
        self.stop_event = None
        self.data_q = None
        self.batch_q = None
        self.n_in_queue = None
        self.data_aggregator = None
        self.data_generators = None
    
        self.init()
    
    def init(self):
        self.stop_event = Event()
        self.data_q = Queue(self.MAX_CAPACITY)
        self.batch_q = Queue(self.MAX_CAPACITY)
        self.n_in_queue = Counter()
        self.data_aggregator = DataAggregator(self.data_q,
                                              self.batch_q,
                                              self.stop_event,
                                              self.batch_size,
                                              n_in_queue=self.n_in_queue,
                                              postprocess_fun=self.postprocess_fun)

        self.data_generators = [DataGenerator(self.data_q,
                                         self.stop_event,
                                         generator_fun=self.generator_fun[ii])
                                for ii in range(len(self.generator_fun))]
                                
        for w in self.data_generators:
            w.start()
        self.data_aggregator.start()

    def next_batch(self):
        """
        Ouput the next batch of examples in the queue

        @returns:
        """
        batch = None
        while batch is None:
            try:
                self.n_in_queue.increment(-1)
                batch = self.batch_q.get(timeout=self.timeout)
            except queue.Empty:
                print("Restarting data_generators")
                self.close()
                self.init()
        
        return batch

    def put_batch(self, batch):
        """
        Puts back a batch of examples in the queue

        @returns:
        """
        if not self.batch_q.full():
            self.batch_q.put(batch)
            self.n_in_queue.increment(1)

    def close(self, timeout: int = 5):
        """
        Terminate running processes

        @returns:
        """
        self.stop_event.set()

        for w in self.data_generators:
            w.join(timeout=timeout)
            while w.is_alive():
                w.terminate()
        self.data_aggregator.join(timeout=timeout)
        while self.data_aggregator.is_alive():
            self.data_aggregator.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


