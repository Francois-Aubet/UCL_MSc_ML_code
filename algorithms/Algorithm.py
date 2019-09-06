
from abc import ABCMeta, abstractmethod

import numpy as np



class Algorithm():
    """
    The abstract parent class for all the algorithms.
    """
    # we define the class as an abstract class:
    __metaclass__ = ABCMeta


    def __init__(self, dataset, meta_data):
        """ Constructor """

        self._dataset = dataset
        self._meta_data = meta_data
        




    @abstractmethod
    def extract_neural_path(self):
        """ 
        
        @:return: 
        """

        raise NotImplementedError("Must override methodB")














