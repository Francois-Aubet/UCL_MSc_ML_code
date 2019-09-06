
from AlgoHandler import AlgoHandler
# we import all the algorithms
from algorithms.PCA import PCA
from algorithms.FA import FA
from algorithms.GPFA import GPFA
from algorithms.GPFA_sv import GPFA_sv
from algorithms.GPFA_sv_lr import GPFA_sv_lr
from algorithms.GPFA_sv_mc import GPFA_sv_mc
from algorithms.GPFA_sv_mc_lr import GPFA_sv_mc_lr

from generating import generate_dataset
from generating_G import generate_dataset_G

from utilis import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from math import isnan

from tqdm import tqdm 







class AutoTest3_orth():
    """ 
    This class creates datasets with different settings and fits the different models on them.
    Then does stats on it.

    The idea here is to train the models for many iterations while 

    stats on the orthogonality of C when initialising at the generating parameters.

    So:
    - same number of latents
    - ranges of measurements, so not saving in pandas

    """

    
    def __init__(self):
        """

        """

        # self._result_file_location = result_file_location

        # varied parameters: 
        self._generating_latent_numb = np.arange(2,3)

        self._inducing_point_numb = np.arange(15,18,5)
        self._em_iterations = 2501

        self._measuring_interval = 20


        self._result_dic = []
        for i in self._inducing_point_numb:
            result_dic = {"original_C_orth_l": []}
            result_dic["GPFA_C_list"] = []
            result_dic["GPFA_sv_C_list"] = []
            result_dic["GPFA_lr_C_list"] = []
            result_dic["GPFA_diff_list"] = []
            result_dic["GPFA_sv_diff_list"] = []
            result_dic["GPFA_lr_diff_list"] = []

            result_dic["GPFA_angle_list"] = []
            result_dic["GPFA_sv_angle_list"] = []
            result_dic["GPFA_lr_angle_list"] = []

            # result_dic["GPFA_free_list"] = []
            result_dic["GPFA_sv_free_list"] = []
            result_dic["GPFA_lr_free_list"] = []

            result_dic["em_iterations"] = self._em_iterations
            result_dic["measuring_interval"] = self._measuring_interval
            result_dic["numb_induc"] = i
            self._result_dic.append(result_dic.copy())

        self._bin_width = 11

        self._numb_tests = 10000
        self._numb_test_trials = 1

        # open file:
        #self._result_file = open(result_file_location,'w')

        self._dataset_name = "auto_test_3_dataset"
        self._dataset_id = -1

        # threading vaiables:
        self._numb_opened_thread = 0
        self._max_opened_threads = 1
        self._semaphore_one_file = False

        self._duration_small_wait = 0.009


    def run_test_random(self):
        """
        Runs some tests.

        We launch one thread for each sensible configuration and write all the results in a file.
        """

        # print("Will do ", len(self._inducing_point_numb)," different configuration tries per test.")

        for test_i in tqdm(range(0, self._numb_tests)):

            start = time.time()
            # if np.random.rand() < new_data_set_chance:
            small_mean = float('nan')
            while isnan(small_mean):
                self._dataset_name = "auto_test_3_dataset" + str(np.mod(test_i,self._max_opened_threads))
                numb_generating_lat = self._generating_latent_numb[np.random.randint(0, len(self._generating_latent_numb))]
                generating_C = generate_dataset_G(self._dataset_name, 25, 100, 500, 30, 1, 0, 0, numb_generating_lat, False, False)
                self._dataset_id += 1

                # start the parameters in the results:
                mean, small_mean, big_mean = get_orthogonality_score(generating_C, False)
                print("\n",mean)

            for res_dic in self._result_dic:
                res_dic["original_C_orth_l"].append(mean)

            # run the test for all settings:
            for index,numb_indu in enumerate(self._inducing_point_numb):
                extra_lat = numb_generating_lat

                # a loop to make sure that it is fine to open a new thread and that waits before doing it
                while self._numb_opened_thread >= self._max_opened_threads:
                    time.sleep(self._duration_small_wait)

                thread = threading.Thread(target=self._fit_and_get_res, args=(extra_lat, numb_indu, self._em_iterations, index, mean, self._dataset_name))
                thread.daemon = True
                thread.start() # Start the execution
                self._numb_opened_thread += 1
                        
            # be sure that this test is done before going to the next one:
            # while self._numb_opened_thread >= 0:
            #     time.sleep(self._duration_small_wait)

            end = time.time()
            print("Test number ",self._dataset_id," took ", (end - start) // 60 , " minutes and ", (end - start) - ((end - start) // 60)*60 , " seconds." )
            


    def _fit_and_get_res(self, extra_lat, numb_indu, numb_em_iter, index, orig_mean, dataset_name):
        """
        This method gets an algorithm handler, fits it, and reports the results in the cvs file
        """

        if index == 0:

            # # first, GPFA:
            algo_hanlder = AlgoHandler(GPFA, dataset_name, 0, 0, extra_lat, self._bin_width, numb_em_iter,
                numb_indu, False, self._numb_test_trials, 0, self._measuring_interval, False)
            # algo_hanlder.initialise_algo_to_gen_param()

            algo_hanlder.extract_path()

            gpfa_orth_list, angle_list, free_list = algo_hanlder.get_orth_c_list()
            gpfa_orth = np.array(gpfa_orth_list)
            gpfa_diff = gpfa_orth - orig_mean
            # print(orig_mean,gpfa_orth,gpfa_diff)
            #temp = self._result_dic["GPFA_C_list"]

            for i in range(0,len(self._inducing_point_numb)):
                self._result_dic[i]["GPFA_C_list"].append(gpfa_orth)
                self._result_dic[i]["GPFA_diff_list"].append(gpfa_diff)
                self._result_dic[i]["GPFA_angle_list"].append(angle_list)


        # then sv_GPFA:
        algo_hanlder = AlgoHandler(GPFA_sv, dataset_name, 0, 0, extra_lat, self._bin_width, numb_em_iter,
            numb_indu, False, self._numb_test_trials, 0, self._measuring_interval, False)
        # algo_hanlder.initialise_algo_to_gen_param()

        algo_hanlder.extract_path()

        gpfa_orth_list, angle_list, free_list = algo_hanlder.get_orth_c_list()
        gpfa_orth = np.array(gpfa_orth_list)
        gpfa_diff = gpfa_orth - orig_mean
        self._result_dic[index]["GPFA_sv_C_list"].append(gpfa_orth)
        self._result_dic[index]["GPFA_sv_diff_list"].append(gpfa_diff)
        self._result_dic[index]["GPFA_sv_angle_list"].append(angle_list)
        self._result_dic[index]["GPFA_sv_free_list"].append(free_list)


        # then lr_sv_GPFA:
        algo_hanlder = AlgoHandler(GPFA_sv_lr, dataset_name, 0, 0, extra_lat, self._bin_width, numb_em_iter,
            numb_indu, False, self._numb_test_trials, 0, self._measuring_interval, False)
        # algo_hanlder.initialise_algo_to_gen_param()

        algo_hanlder.extract_path()
        
        gpfa_orth_list, angle_list, free_list = algo_hanlder.get_orth_c_list()
        gpfa_orth = np.array(gpfa_orth_list)
        gpfa_diff = gpfa_orth - orig_mean
        self._result_dic[index]["GPFA_lr_C_list"].append(gpfa_orth)
        self._result_dic[index]["GPFA_lr_diff_list"].append(gpfa_diff)
        self._result_dic[index]["GPFA_lr_angle_list"].append(angle_list)
        self._result_dic[index]["GPFA_lr_free_list"].append(free_list)


        while self._semaphore_one_file:
            time.sleep(self._duration_small_wait)
        self._semaphore_one_file = True
        save_obj(self._result_dic,"test_5.1_run3")
        # self._result_file.write(results)
        # self._result_file.write("\n")
        self._semaphore_one_file = False
        self._numb_opened_thread -= 1
        # print("saved one")



print("starting")

# csv_resutls = "results/third_test_1.csv"7
tester = AutoTest3_orth()

tester.run_test_random()