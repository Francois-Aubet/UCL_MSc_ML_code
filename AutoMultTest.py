
from MultiAlgoHandler import MultiAlgoHandler
from MultiAlgoHandler9 import MultiAlgoHandler9
from MultiAlgoHandler10 import MultiAlgoHandler10
from MultiAlgoHandler12 import MultiAlgoHandler12
from MultiAlgoHandler13 import MultiAlgoHandler13

from generating import generate_dataset
from generating_G import generate_dataset_G

from utilis import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from math import isnan

from tqdm import tqdm 







class AutoMultTest():
    """ 
    
    """

    
    def __init__(self, result_file_location):
        """

        """

        self._result_file_location = result_file_location

        # varied parameters: 

        self._initial_types = [0,2,4]

        self._inducing_point_numb = [50]#,30,45]#np.arange(15,18,5)
        self._em_iterations = 1005

        self._generating_latent_numb = [2] #np.arange(2,6)
        self._extracted_latent_numb = [2]#2,3,4]  #np.arange(2,6)

        self._bin_width = 11

        self._numb_tests = 5
        self._numb_test_trials = 1

        # open file:
        #self._result_file = open(result_file_location,'w')

        self._dataset_name = "auto_test_9_dataset"
        self._dataset_id = 0

        # threading vaiables:
        self._numb_opened_thread = 0
        self._max_opened_threads = 3
        self._semaphore_one_file = False

        self._duration_small_wait = 0.09


    def run_test_random(self):
        """
        Runs some tests.

        We launch one thread for each sensible configuration and write all the results in a file.
        """

        # print("Will do ", len(self._inducing_point_numb)," different configuration tries per test.")

        for test_i in tqdm(range(0, self._numb_tests)):
            numb_bins = 50 # 50
            numb_trials = 25 #25
            R_mean = 1/6
            numb_neurons = 100 #10 #100

            start = time.time()
            # if np.random.rand() < new_data_set_chance:
            small_mean = float('nan')
            self._dataset_id += 1
            while isnan(small_mean):
                self._dataset_name = "auto_test_9_dataset" + str(np.mod(test_i,self._max_opened_threads))
                numb_generating_lat = self._generating_latent_numb[np.random.randint(0, len(self._generating_latent_numb))]
                generating_C = generate_dataset_G(self._dataset_name, numb_trials, numb_neurons, 500, 500//numb_bins, 1, 0, 0, numb_generating_lat, R_mean, False, False)

                # start the parameters in the results:
                mean, small_mean, big_mean = get_orthogonality_score(generating_C, False)
                print("\n",mean)

            result_start = str(self._dataset_id) + "," + str(numb_generating_lat) + "," + str(numb_trials) + "," + str(numb_bins) + "," + str(R_mean) + "," + str(numb_neurons)

            # run the test for all settings:
            for extra_lat in self._extracted_latent_numb:
                for init_type in self._initial_types:
                    for index,numb_indu in enumerate(self._inducing_point_numb):
                        #extra_lat = numb_generating_lat

                        # a loop to make sure that it is fine to open a new thread and that waits before doing it
                        while self._numb_opened_thread >= self._max_opened_threads:
                            time.sleep(self._duration_small_wait)

                        thread = threading.Thread(target=self._fit_and_get_res, args=(extra_lat, numb_indu, self._em_iterations, init_type, mean, self._dataset_name, result_start))
                        thread.daemon = True
                        thread.start() # Start the execution
                        self._numb_opened_thread += 1
                        
            # be sure that this test is done before going to the next one:
            # while self._numb_opened_thread >= 0:
            #     time.sleep(self._duration_small_wait)

            end = time.time()
            print("Test number ",self._dataset_id," took ", (end - start) // 60 , " minutes and ", (end - start) - ((end - start) // 60)*60 , " seconds." )
            


    def _fit_and_get_res(self, extra_lat, numb_indu, numb_em_iter, init_type,  orig_mean, dataset_name, result_start):
        """
        This method gets an algorithm handler, fits it, and reports the results in the cvs file
        """

        result_start += "," + str(extra_lat) + "," + str(numb_indu) + "," + str(init_type)

        algo_handler = MultiAlgoHandler12(dataset_name, 0, 0, extra_lat, self._bin_width, numb_em_iter,
            numb_indu, init_type, 0, result_start, self._result_file_location, self._numb_test_trials, False, False)

        algo_handler.run_EM_iterations()
        

        self._numb_opened_thread -= 1
        # print("saved one")



print("starting")

csv_resutls = "results/test_12_ind_loc_1.csv"
# csv_resutls = "results/test_12_14_with_R.csv"
tester = AutoMultTest(csv_resutls)

tester.run_test_random()