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
from scipy.linalg import block_diag

from copy import deepcopy



class MultiAlgoHandler13(AlgoHandler):
    """ 
    

    """


    def __init__(self, dataset_name, numb_shared_lat, numb_grouped_lat, numb_trial_lat, bin_width, max_EM_iterations,
        numb_inducing, init_type, hybrid_steps, result_start, res_file, numb_test_trials = 1, verbose = False, learn_kernel_params = False):

        # convergences to be tested:
        self._convergence_criterias_l = np.array([1,2,3,4,5,6,7,8,9,10,11,13,15,17])

        # results:
        self._result_start = result_start
        self._result_file_location = res_file

        # save the number of latents to extract:
        self._numb_shared_lat = numb_shared_lat
        self._numb_grouped_lat = numb_grouped_lat
        self._K_numb_trial_lat = numb_trial_lat

        self._bin_width = bin_width

        # read in dataset:
        self._read_dataset(dataset_name)
        self._dataset_name = dataset_name

        self._bin_dataset(numb_test_trials)
        self._numb_test_trials = numb_test_trials

        self._verbose = verbose
        self._init_type = 0

        self._max_EM_iterations = max_EM_iterations

        self._numb_inducing = numb_inducing
        self._numb_trial_lat = numb_trial_lat

        # create the algorithms:
        self._gpfa_full = GPFA(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
            learn_kernel_params, 0, -1)
        self._gpfa_full.use_MF_approximation = False

        self._algo_list = [self._gpfa_full]
       
        # logging settings:
        # self._logging_inter_start = 14
        # self._logging_inter_normal = 100
        # self._logging_break = 200

        # get orig_C_orth_mean and orig_C_err
        orig_C_err = self._get_orig_C_error()
        orig_C_orth_mean, small, big = get_orthogonality_score(self._meta_data["C_gamma"], False)
        self._result_start += "," + str(orig_C_err) + "," + str(orig_C_orth_mean)



    def _initialise_algos(self):
        """
        We want to make sure that we initialise all algorithms in the same way, this function takes care of that.
        
        init type = 
        0 : initialize to original parameters
        1 : fa with the same orthogonality as the original parameters

        """
        # if self._init_type == 0:
        for algo in self._algo_list:
            algo.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        # else:

        #     for algo in self._algo_list:
        #         algo._fa_initialisation()

        for algo in self._algo_list:
            algo._specific_init()



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
            

        # after the EM iterations, log everything
        self._log_all_stats_quick(self._max_EM_iterations)



    def _log_all_stats_quick(self, iter_number):
        """
        Computes and logs all the stats that we want to access.

        """
        log_covariances = True

        results = self._result_start + "," + str(iter_number)

        # self._gpfa_full._E_step()
        gpfa_covars = self._gpfa_full._temp_Sigma_x.copy()


        ### the ground truth from the sv-full-GPFA

        sv_full_gpfa = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        sv_full_gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        sv_full_gpfa._specific_init()
        sv_full_gpfa.use_MF_approximation = False
        sv_full_gpfa._E_step()

        full_sparse_covarse = sv_full_gpfa._S_full_matr.copy()

        results += "," + self.get_current_Free_KL_like_quick(sv_full_gpfa, True)
        if log_covariances:
            results += "," + stats_of_covariance(full_sparse_covarse, self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(gpfa_covars - sv_full_gpfa.get_full_XX_covariance(), self._K_numb_trial_lat, True)

        ### the exact methods

        sv_gpfa = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        sv_gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        sv_gpfa._specific_init()
        sv_gpfa.use_MF_approximation = True
        sv_gpfa._E_step()

        lr_gpfa = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        lr_gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        lr_gpfa._specific_init()
        lr_gpfa.use_MF_approximation = True
        lr_gpfa._use_lr_E_steps = True
        lr_gpfa._E_step()


        # S statistics:
        for algo in [sv_gpfa, lr_gpfa]:
            results += "," + self.get_current_Free_KL_like_quick(algo, True)

            algo_full_covars = algo.get_full_XX_covariance()
            
            if log_covariances:
                results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
                results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
                results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)
            
            

        #### gradient methods:
        grad_full = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        grad_full.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        grad_full.use_MF_approximation = False
        grad_full._use_gradient_E_steps = True
        grad_full._specific_init()

        grad_mf = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        grad_mf.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        grad_mf.use_MF_approximation = True
        grad_mf._use_gradient_E_steps = True
        grad_mf._specific_init()


        for criteria in tqdm(self._convergence_criterias_l):
            convergence_crit = 10.0**(-1 * criteria)

            grad_full_temp = deepcopy(grad_full)
            grad_full_temp._convergence_criteria = convergence_crit
            grad_full_temp._E_step()

            grad_mf_temp = deepcopy(grad_mf)
            grad_mf_temp._convergence_criteria = convergence_crit
            grad_mf_temp._E_step()

            # grad_lr = deepcopy(template_algo)
            # grad_lr._use_lr_E_steps = True
            # grad_lr._use_gradient_E_steps = True
            # grad_lr._convergence_criteria = convergence_crit
            # grad_lr._E_step()

            grad_lr = deepcopy(grad_mf_temp)
            grad_lr.linear_response_correction()


            # perturbated_mf = deepcopy(sv_gpfa)
            # perturbated_mf._perturbate_S()

            # perturbated_lr = deepcopy(perturbated_mf)
            # perturbated_lr.linear_response_correction()


            # S statistics:
            for algo in [grad_full_temp, grad_mf_temp, grad_lr]:#, perturbated_mf, perturbated_lr]:
                results += "," + str(np.mean(algo._number_gradient_iterations_l))
                results += "," + self.get_current_Free_KL_like_quick(algo, True)

                results += "," + str(is_pos_def(algo._S_full_matr))

                algo_full_covars = algo.get_full_XX_covariance()
                
                if log_covariances:
                    results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
                    results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
                    results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)



        results = results.replace('\n', ' ').replace('\r', '')

        with open(self._result_file_location ,'a') as f:
            f.write(results+"\n")




    def _log_all_stats(self, iter_number):
        """
        Computes and logs all the stats that we want to access.

        """

        results = self._result_start + "," + str(iter_number)

        # self._gpfa_full._E_step()
        gpfa_covars = self._gpfa_full._temp_Sigma_x.copy()


        ### the ground truth from the sv-full-GPFA

        sv_full_gpfa = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        sv_full_gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        sv_full_gpfa._specific_init()
        sv_full_gpfa.use_MF_approximation = False
        sv_full_gpfa._E_step()

        full_sparse_covarse = sv_full_gpfa._S_full_matr.copy()

        results += "," + self.get_current_Free_KL_like2(sv_full_gpfa, True)
        results += "," + stats_of_covariance(full_sparse_covarse, self._K_numb_trial_lat, True)
        results += "," + stats_of_covariance(gpfa_covars - sv_full_gpfa.get_full_XX_covariance(), self._K_numb_trial_lat, True)

        ### the exact methods

        sv_gpfa = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        sv_gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        sv_gpfa._specific_init()
        sv_gpfa.use_MF_approximation = True
        sv_gpfa._E_step()

        lr_gpfa = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        lr_gpfa.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
        lr_gpfa._specific_init()
        lr_gpfa.use_MF_approximation = True
        lr_gpfa._use_lr_E_steps = True
        lr_gpfa._E_step()


        # S statistics:
        for algo in [sv_gpfa, lr_gpfa]:
            results += "," + self.get_current_Free_KL_like2(algo, True)

            algo_full_covars = algo.get_full_XX_covariance()
            
            # results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
            # results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
            # results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)
            
            results += "," + stats_of_covariance(np.linalg.inv(algo._S_full_matr), self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(np.linalg.inv(algo._S_full_matr) - np.linalg.inv(full_sparse_covarse), self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(np.linalg.inv(gpfa_covars) - np.linalg.inv(algo_full_covars), self._K_numb_trial_lat, True)



        #### gradient methods:

        for criteria in tqdm(self._convergence_criterias_l):
            # convergence_crit = 10.0**(-1 * criteria)

            grad_full = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
            grad_full.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
            grad_full._specific_init()
            grad_full.use_MF_approximation = False
            grad_full._use_gradient_E_steps = True
            # grad_full._convergence_criteria = convergence_crit
            grad_full._E_step()

            grad_mf = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
            grad_mf.set_model_params_to(self._meta_data["C_gamma"].copy(), self._meta_data["d_bias"].copy(), self._meta_data["gen_R"].copy())
            grad_mf._specific_init()
            grad_mf.use_MF_approximation = True
            grad_mf._use_gradient_E_steps = True
            # grad_mf._convergence_criteria = convergence_crit
            grad_mf._E_step()

            # grad_lr = deepcopy(template_algo)
            # grad_lr._use_lr_E_steps = True
            # grad_lr._use_gradient_E_steps = True
            # grad_lr._convergence_criteria = convergence_crit
            # grad_lr._E_step()

            # grad_lr = deepcopy(grad_mf)
            # grad_lr.linear_response_correction()


            # S statistics:
            for algo in [grad_full, grad_mf]:
                results += "," + str(np.mean(algo._number_gradient_iterations_l))
                results += "," + self.get_current_Free_KL_like2(algo, True)

                results += "," + str(is_pos_def(algo._S_full_matr))

                algo_full_covars = algo.get_full_XX_covariance()
                
                results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
                results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
                results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)

            algo = grad_mf

            algo.linear_response_correction()

            results += "," + str(np.mean(algo._number_gradient_iterations_l))
            results += "," + self.get_current_Free_KL_like2(algo, True)

            results += "," + str(is_pos_def(algo._S_full_matr))

            algo_full_covars = algo.get_full_XX_covariance()
            
            results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)



        results = results.replace('\n', ' ').replace('\r', '')

        with open(self._result_file_location ,'a') as f:
            f.write(results+"\n")




    def _log_all_stats_old(self, iter_number):
        """
        Computes and logs all the stats that we want to access.

        """

        results = self._result_start + "," + str(iter_number)

        gpfa_covars = self._gpfa_full._temp_Sigma_x.copy()

        C = self._gpfa_full._C_matrix.copy()
        d = self._gpfa_full._d_bias.copy()
        R = self._gpfa_full._R_noise.copy()


        template_algo = GPFA_sv_lr(self._dataset_binned, self._meta_data, self._bin_times, self._numb_trial_lat, 0, self._numb_inducing, False, 0, -1)
        template_algo.use_MF_approximation = False
        template_algo._specific_init()
        template_algo.set_model_params_to(C, d, R)

        ### the ground truth from the sv-full-GPFA

        sv_full_gpfa = deepcopy(template_algo)
        sv_full_gpfa.use_MF_approximation = False
        sv_full_gpfa._E_step()
        full_sparse_covarse = sv_full_gpfa._S_full_matr.copy()

        results += "," + self.get_current_Free_KL_like2(sv_full_gpfa, True)
        results += "," + stats_of_covariance(full_sparse_covarse, self._K_numb_trial_lat, True)
        results += "," + stats_of_covariance(gpfa_covars - sv_full_gpfa.get_full_XX_covariance(), self._K_numb_trial_lat, True)

        ### the exact methods

        sv_gpfa = deepcopy(template_algo)
        sv_gpfa.use_MF_approximation = True
        sv_gpfa._E_step()

        lr_gpfa = deepcopy(template_algo)
        lr_gpfa._use_lr_E_steps = True
        lr_gpfa._E_step()


        # S statistics:
        for algo in [sv_gpfa, lr_gpfa]:
            results += "," + self.get_current_Free_KL_like2(algo, True)

            algo_full_covars = algo.get_full_XX_covariance()
            
            results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
            results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)



        #### gradient methods:

        for criteria in tqdm(self._convergence_criterias_l):
            convergence_crit = 10.0**(-1 * criteria)

            grad_full = deepcopy(template_algo)
            grad_full.use_MF_approximation = False
            grad_full._use_gradient_E_steps = True
            grad_full._convergence_criteria = convergence_crit
            grad_full._E_step()

            grad_mf = deepcopy(template_algo)
            grad_mf.use_MF_approximation = True
            grad_mf._use_gradient_E_steps = True
            grad_mf._convergence_criteria = convergence_crit
            grad_mf._E_step()

            # grad_lr = deepcopy(template_algo)
            # grad_lr._use_lr_E_steps = True
            # grad_lr._use_gradient_E_steps = True
            # grad_lr._convergence_criteria = convergence_crit
            # grad_lr._E_step()

            grad_lr = deepcopy(grad_mf)
            grad_lr.linear_response_correction()


            # S statistics:
            for algo in [grad_full, grad_mf, grad_lr]:
                results += "," + str(np.mean(algo._number_gradient_iterations_l))
                results += "," + self.get_current_Free_KL_like2(algo, True)

                results += "," + str(is_pos_def(algo._S_full_matr))

                algo_full_covars = algo.get_full_XX_covariance()
                
                results += "," + stats_of_covariance(algo._S_full_matr, self._K_numb_trial_lat, True)
                results += "," + stats_of_covariance(algo._S_full_matr - full_sparse_covarse, self._K_numb_trial_lat, True)
                results += "," + stats_of_covariance(gpfa_covars - algo_full_covars, self._K_numb_trial_lat, True)

        results = results.replace('\n', ' ').replace('\r', '')

        with open(self._result_file_location ,'a') as f:
            f.write(results+"\n")






