import torch
import numpy as np

""" Store data for different loop
Design consideration: change the original "has-a" relationship of LT->rollout->transition to parallel.
Transition stores all the needed data for every step and will be cleared/updated every step and passed to rollout and longterm
Rollout store data for every iteration, generate batch for learning
Longterm stores needed data from transition, and probably also generate batches
"""

class BaseStorage:
    """ This class is a skeleton for an arbitrary storage type. """

    class Transition:
        """ Transition storage class.
        i.e. store data for each STEP of ALL agents
        """
        def __init__(self):
            """ Define all the data you need to store in __init__
            """
            raise NotImplementedError
    
        def clear(self):
            self.__init__()

    def __init__(self, max_storage, device='cpu'):
        self.device = device
        self.max_storage = max_storage
        # fill_count keeps track of how much storage is filled with actual data
        # anything after the fill_count is stale and should be ignored
        self.fill_count = 0

    def add_transitions(self, transition: Transition):
        """ Add current transition to LT storage
        Store variables according to the __init__ 
        """
        self.fill_count += 1
        raise NotImplementedError

    def clear(self):
        self.fill_count = 0


    def mini_batch_generator(self):
        """ Generate mini batch for learning
        """
        raise NotImplementedError
