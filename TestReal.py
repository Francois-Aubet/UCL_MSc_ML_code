
from MultiAlgoHandler import MultiAlgoHandler
from MultiAlgoHandler12 import MultiAlgoHandler12

from generating import generate_dataset
from generating_G import generate_dataset_G

from utilis import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from math import isnan

from tqdm import tqdm 




# csv_resutls = "results/test_12_14_with_R.csv"
tester = ()



class TestReal():
    """ 
    
    """

    
    def __init__(self):
        """

        """

        result_file_location = "results/test_15.csv"

        self._result_file_location = result_file_location

        # varied parameters: 

        self._initial_types = [1]

        self._inducing_point_numb = [15]#,30,45]#np.arange(15,18,5)
        self._em_iterations = 2405


        self._extracted_latent_numb = [3]#2,3,4]  #np.arange(2,6)

        self._bin_width = 30

        self._numb_test_trials = 3

        # open file:
        #self._result_file = open(result_file_location,'w')

        self._dataset_name = "datasets/movement-data/MotorHam3.mat"
        self._dataset_id = 3

        # threading vaiables:
        self._numb_opened_thread = 0
        self._max_opened_threads = 2
        self._semaphore_one_file = False

        self._duration_small_wait = 0.09


    def run_test_random(self):
        """
        Runs some tests.

        We launch one thread for each sensible configuration and write all the results in a file.
        """

        # print("Will do ", len(self._inducing_point_numb)," different configuration tries per test.")
        numb_indu = self._inducing_point_numb[0]
        init_type = 1
        extra_lat = 6

        numb_trials = 80
        numb_bins = 136
        numb_neurons = 80

        result_start = str(self._dataset_id) + "," + str(-1) + "," + str(numb_trials) + "," + str(numb_bins) + "," + str(-1) + "," + str(numb_neurons)

        result_start += "," + str(extra_lat) + "," + str(numb_indu) + "," + str(init_type)

        algo_handler = MultiAlgoHandler12(self._dataset_name, 0, 0, extra_lat, self._bin_width, self._em_iterations,
            numb_indu, init_type, 0, result_start, self._result_file_location, self._numb_test_trials, False, False)



        algo_handler.run_EM_iterations()


        end = time.time()
        print("Test number ",self._dataset_id," took ", (end - start) // 60 , " minutes and ", (end - start) - ((end - start) // 60)*60 , " seconds." )
            



print("starting")

tester = TestReal()

tester.run_test_random()