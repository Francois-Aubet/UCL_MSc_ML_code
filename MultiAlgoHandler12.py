""""

This is an algortihm handler of a new caliber, it runs multiple algorithms at the same time and compares their performances.


"""

import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import pickle
from algorithms.PCA import PCA
from algorithms.FA import FA
from algorithms.GPFA import GPFA
from algorithms.GPFA_sv import GPFA_sv
from algorithms.GPFA_sv_lr import GPFA_sv_lr
from algorithms.GPFA_sv_mc import GPFA_sv_mc
from algorithms.GPFA_sv_mc_lr import GPFA_sv_mc_lr

from AlgoHandler import AlgoHandler

from tqdm import tqdm 
from utilis import *

from copy import deepcopy



class MultiAlgoHandler12(AlgoHandler):
    """ 
    
    """


    def __init__(self, dataset_name, numb_shared_lat, numb_grouped_lat, numb_trial_lat, bin_width, max_EM_iterations,
        numb_inducing, init_type, hybrid_steps, result_start, res_file, numb_test_trials = 1, verbose = False, learn_kernel_params = False):

        # results:
        self._result_start = result_start 
        self._result_file_location = res_file

        # save the number of latents to extract:
        self._numb_shared_lat = numb_shared_lat
        self._numb_grouped_lat = numb_grouped_lat
        self._K_numb_trial_lat = numb_trial_lat

        self._bin_width = bin_width

        # read in dataset:
        self._numb_test_trials = numb_test_trials
        self._read_dataset(dataset_name)
        self._dataset_name = dataset_name

        self._bin_dataset(numb_test_trials)

        self._verbose = verbose
        self._init_type = init_type

        self._max_EM_iterations = max_EM_iterations

        # create the algorithms:
        ## GPFA:
        self._gpfa_full = GPFA(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._gpfa_full.use_MF_approximation = False

        self._gpfa_mf = GPFA(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._gpfa_mf.use_MF_approximation = True

        ## sparse exact:
        self._sv_gpfa_full = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._sv_gpfa_full.use_MF_approximation = False

        self._sv_gpfa_mf = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._sv_gpfa_mf.use_MF_approximation = True

        self._sv_LR = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._sv_LR._use_lr_E_steps = True

        #### sparse gradient:
        # self._sv_grad_full = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
        #     learn_kernel_params, 0, -1)
        # self._sv_grad_full.use_MF_approximation = False
        # self._sv_grad_full._use_gradient_E_steps = True

        self._sv_grad_mf = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._sv_grad_mf.use_MF_approximation = True
        self._sv_grad_mf._use_gradient_E_steps = True

        self._sv_grad_LR = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._sv_grad_LR._use_lr_E_steps = True
        self._sv_grad_LR._use_gradient_E_steps = True


        # self._algo_list = [self._gpfa_full, self._gpfa_mf, self._sv_gpfa_full, self._sv_gpfa_mf, self._sv_LR, self._sv_grad_mf, self._sv_grad_LR]#, self._sv_grad_full] #
        
        self._algo_list = [self._sv_gpfa_full]
        
        # for algo in self._algo_list:
        #     algo._learn_only_C = True
       
        # logging settings:
        self._logging_inter_start = 20
        self._logging_inter_normal = 100
        self._logging_break = 200

        # get orig_C_orth_mean and orig_C_err
        # orig_C_err = self._get_orig_C_error()
        # orig_C_orth_mean, small, big = get_orthogonality_score(self._meta_data["C_gamma"], False)
        # self._result_start += "," + str(orig_C_err) + "," + str(orig_C_orth_mean)



    def _initialise_algos(self):
        """
        We want to make sure that we initialise all algorithms in the same way, this function takes care of that.
        
        init type = 
        0 : initialize to original parameters
        1 : fa with the same orthogonality as the original parameters

        """
        # if self._init_type == 0:
            #self._gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())

        for algo in self._algo_list:
            algo.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        # else:
        #     for algo in self._algo_list:
        #         algo._fa_initialisation()

        for algo in self._algo_list:
            algo._specific_init()

            if self._init_type != 0:
                algo._z_induc_loc += (10.0)**(-1 * self._init_type)
                print(algo._z_induc_loc)

        



    def run_EM_iterations(self):
        """
        We run the EM iterations from all the algorithms from here, in order to have a hand on the whole process.
        """

        self._initialise_algos()

        for em_iter in tqdm(range(0,self._max_EM_iterations)):

            ## do the em updates of each algorithm:

            for algo in self._algo_list:
                algo._E_step()
                algo._M_step()
            
            
            ## if it is time, logs the stats:
            is_looging_iter = (2 == np.mod(em_iter,self._logging_inter_normal)) or ((2 == np.mod(em_iter,self._logging_inter_start)) and (em_iter < self._logging_break))

            if is_looging_iter:
                self._log_all_stats(em_iter)




    def _log_all_stats(self, iter_number):
        """
        Computes and logs all the stats that we want to access.

        """

        results = self._result_start + "," + str(iter_number)

        # get GPFA stats:

        # gpfa_covars = self._gpfa_full._temp_Sigma_x.copy()
        # full_sparse_covarse = self._sv_gpfa_full._S_full_matr.copy()


        for algo in self._algo_list:
            
            # free energy and so on:
            results += "," + self.get_current_Free_KL_like2(algo, True)

            # C statistics:
            C_mean_orth = get_orthogonality_score(algo._C_matrix, False)[0]
            # C_angle = get_angle_subspaces(self._meta_data["C_gamma"], algo._C_matrix, False)
            C_angle = get_angle_subspaces(self._gpfa_full._C_matrix, algo._C_matrix, False)
            results += "," + str(C_mean_orth) + "," + str(C_angle)
            results += "," + str(np.linalg.norm(algo._C_matrix[:,0])) + "," + str(np.linalg.norm(algo._C_matrix[:,1]))
            results += "," + str(np.mean(np.abs( self._gpfa_full._C_matrix - algo._C_matrix ))) + "," + str(fnorm( self._gpfa_full._C_matrix - algo._C_matrix ))

            # R statistics:
            # results += "," + str(np.linalg.norm(np.diag(algo._R_noise))) + "," + str(np.mean( np.diag(self._meta_data["gen_R"]) - np.diag(algo._R_noise) ))
            results += "," + str(np.linalg.norm(np.diag(algo._R_noise))) + "," + str(np.mean( np.diag(self._gpfa_full._R_noise) - np.diag(algo._R_noise) ))
            
            # Sigma statistics:
            # if type(algo) == GPFA:
            #     results += "," + str(np.linalg.norm(np.diag(algo._temp_Sigma_x))) + "," + str(np.mean( np.diag(gpfa_covars) - np.diag(algo._temp_Sigma_x) ))
            # else:
            #     algo_full_covars = algo.get_full_XX_covariance()
            #     results += "," + str(np.linalg.norm(np.diag(algo_full_covars))) + "," + str(np.mean( np.diag(gpfa_covars) - np.diag(algo_full_covars) ))
            
            # error statistic:
            # results += "," + self.compute_dataset_errors_from_C(algo, True)

            # S statistics:
            # if type(algo) == GPFA:
            #     results += "," + stats_of_covariance(algo._temp_Sigma_x - gpfa_covars, self._K_numb_trial_lat, True, True) # False, 
            # else:
            #     results += "," + stats_of_covariance(algo_full_covars - gpfa_covars, self._K_numb_trial_lat, True, True)
            #     results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True, True)


        results = results.replace('\n', ' ').replace('\r', '')

        with open(self._result_file_location ,'a') as f:
            f.write(results+"\n")






