
from AlgoHandler import AlgoHandler
# we import all the algorithms
from algorithms.PCA import PCA
from algorithms.FA import FA
from algorithms.GPFA import GPFA
from algorithms.GPFA_sv import GPFA_sv
from algorithms.GPFA_sv_lr import GPFA_sv_lr
from algorithms.GPFA_sv_mc import GPFA_sv_mc

from generating import generate_dataset
from generating_GC import generate_dataset_GC

from utilis import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from math import isnan

from tqdm import tqdm 








class AutoTest():
    """ 
    This class creates datasets with different settings and fits the different models on them.
    Then does stats on it.



    """

    
    def __init__(self, result_file_location):
        """

        """

        self._result_file_location = result_file_location

        # varied parameters: 
        self._generating_latent_numb = np.arange(2,6)
        self._extracted_latent_numb = np.arange(2,7)
        self._inducing_point_numb = np.arange(5,21,5)
        self._em_iterations = np.arange(30,51,20) # just testing two values here for now
        # self._em_iterations = np.arange(3,5,2) # to test the whole thingy quickly

        self._inter_correlation_values = [0,0.1]

        self._bin_width = 21

        self._numb_tests = 100000
        self._numb_test_trials = 1
        self._numb_trials = 25

        # open file:
        #self._result_file = open(result_file_location,'w')

        self._dataset_name = "auto_test_6_dataset"
        self._dataset_id = 29

        # self._numb_random_init = 2

        # threading vaiables:
        self._numb_opened_thread = 0
        self._max_opened_threads = 2
        self._semaphore_one_file = False

        self._duration_small_wait = 0.009


    def run_test_random(self):
        """
        Runs some tests.

        We launch one thread for each sensible configuration and write all the results in a file.
        """
        # new_data_set_chance = 0.1
        # # generate dataset:
        # numb_generating_lat = self._generating_latent_numb[np.random.randint(0, len(self._generating_latent_numb))]
        # generate_dataset(self._dataset_name, 10, 100, 400, 1, numb_generating_lat, 0, 0, False, False)

        # print("Will do ", len(self._extracted_latent_numb) * len(self._em_iterations)," different configuration tries per test.")

        for test_i in tqdm(range(0, self._numb_tests)):

            numb_indu = self._inducing_point_numb[np.mod(test_i,len(self._inducing_point_numb))]
            inter_corre_val = self._inter_correlation_values[np.random.randint(len(self._inter_correlation_values))]

            start = time.time()
            # if np.random.rand() < new_data_set_chance:
            small_mean = float('nan')
            while isnan(small_mean):
                numb_generating_lat = self._generating_latent_numb[np.random.randint(0, len(self._generating_latent_numb))]
                generating_C = generate_dataset_GC(self._dataset_name, self._numb_trials, 100, 400, 20, 1, 0, 0, numb_generating_lat, \
                    False, numb_indu, False, inter_corre_val)
                self._dataset_id += 1

                # start the parameters in the results:
                result_start = str(self._dataset_id) + "," + str(numb_generating_lat) + "," + str(inter_corre_val)
                mean, small_mean, big_mean = get_orthogonality_score(generating_C, False)
                #result_start += "," + str(mean) + "," + str(small_mean) + "," + str(big_mean)
                print(small_mean)

            # run the test for all settings:
            # for  extra_lat in self._extracted_latent_numb:
            extra_lat = numb_generating_lat
            for numb_em_iter in self._em_iterations:
                #for init_i in range(0, self._numb_random_init):

                # a loop to make sure that it is fine to open a new thread and that waits before doing it
                while self._numb_opened_thread >= self._max_opened_threads:
                    time.sleep(self._duration_small_wait)

                thread = threading.Thread(target=self._fit_and_get_res, args=(extra_lat, numb_indu, numb_em_iter, result_start))
                thread.daemon = True
                thread.start() # Start the execution
                self._numb_opened_thread += 1
                        
            # be sure that this test is done before going to the next one:
            # while self._numb_opened_thread >= 0:
            #     time.sleep(self._duration_small_wait)

            end = time.time()
            print("Test number ",self._dataset_id," took ", (end - start) // 60 , " minutes and ", (end - start) - ((end - start) // 60)*60 , " seconds." )



    def _fit_and_get_res(self, extra_lat, numb_indu, numb_em_iter, result_start):
        """
        This method gets an algorithm handler, fits it, and reports the results in the cvs file
        """

        try:
            results = result_start + "," + str(extra_lat) + "," + str(numb_indu) + "," + str(numb_em_iter) + ","

            # first, GPFA: not interested by GPFA in test4
            # algo_hanlder = AlgoHandler(GPFA, self._dataset_name, 0, 0, extra_lat, self._bin_width, numb_em_iter,
            #     numb_indu, False, self._numb_test_trials, 0, False)

            # algo_hanlder.extract_path()

            # mean, small_mean, big_mean = algo_hanlder.get_orthogonality_scores()
            # error_C_orig, error_C, error_S, error_posterior = algo_hanlder.compute_dataset_errors_from_C()

            # results += str(mean) + "," + str(small_mean) + "," + str(big_mean) + ","
            # results += str(error_C_orig) + "," + str(error_C) + "," + str(error_S) + "," + str(error_posterior) + ","

            # then sv_GPFA:
            algo_handler = AlgoHandler(GPFA_sv_lr, self._dataset_name, 0, 0, extra_lat, self._bin_width, numb_em_iter,
                numb_indu, False, self._numb_test_trials, 0, -1, False)

            algo_handler.initialise_algo_to_gen_param()

            algo_handler.extract_path()

            free, like, kl, free_full, like_full, kl_full = algo_handler.get_current_Free_KL_like()

            results += str(free) + "," + str(like) + "," + str(kl) + "," + str(free_full) + "," + str(like_full) + "," + str(kl_full) + ","

            # finally, lr-sv-GPFA:
            # illegal manipulation, but saves the refitting time:
            algo_handler._algo._set_S_to_generating()
            algo_handler._algo._M_step()
            algo_handler._algo.E_step_update_m_only()

            free, like, kl, free_full, like_full, kl_full = algo_handler.get_current_Free_KL_like()

            results += str(free) + "," + str(like) + "," + str(kl) + "," + str(free_full) + "," + str(like_full) + "," + str(kl_full)


            while self._semaphore_one_file:
                time.sleep(self._duration_small_wait)
            self._semaphore_one_file = True
            with open(self._result_file_location ,'a') as f:
                f.write(results)
                f.write("\n")
            # self._result_file.write(results)
            # self._result_file.write("\n")
            self._semaphore_one_file = False
        except:
            ouerhugh = 0
            print("got error....")

        self._numb_opened_thread -= 1


print("starting")

csv_resutls = "results/test_6.csv"
tester = AutoTest(csv_resutls)

tester.run_test_random()