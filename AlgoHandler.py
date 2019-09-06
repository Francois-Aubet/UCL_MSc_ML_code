
from algorithms.PCA import PCA
from algorithms.FA import FA
from algorithms.GPFA import GPFA
from algorithms.GPFA_sv import GPFA_sv
from algorithms.GPFA_sv_lr import GPFA_sv_lr
from algorithms.GPFA_sv_mc import GPFA_sv_mc
from algorithms.GPFA_sv_mc_lr import GPFA_sv_mc_lr
#from algorithms.GPFA_beta import GPFA_beta


from utilis import get_orthogonality_score, get_angle_subspaces, load_obj, make_K

import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import pickle
# from utilis import *
from scipy.linalg import block_diag

from scipy.io import loadmat



class AlgoHandler():
    """ 
    This class takes a dataset and an algorithm class and can use it to:
    - extract the trajectories
    - plot them
    - do cross validation


    """


    def __init__(self, algoClass, dataset_name, numb_shared_lat, numb_grouped_lat, numb_trial_lat, bin_width, max_EM_iterations,
            numb_inducing, learn_kernel_params, numb_test_trials = 0, plot_free_energy = 0, save_ortho_step = -1, verbose = True):
        """
        Creates everything in the class.

        """
        #  define the algo class
        self._algoClass = algoClass

        # save the number of latents to extract:
        self._numb_shared_lat = numb_shared_lat
        self._numb_grouped_lat = numb_grouped_lat
        self._numb_trial_lat = numb_trial_lat

        self._numb_test_trials = numb_test_trials

        self._bin_width = bin_width

        # read in dataset:
        self._read_dataset(dataset_name)
        self._dataset_name = dataset_name

        self._verbose = verbose

        # bin it:
        self._bin_dataset(numb_test_trials)

        # create an instance of the algorithm class
        if algoClass is FA or algoClass is PCA:
            self._algo = self._algoClass(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat)
        elif algoClass is GPFA_sv_mc or algoClass is GPFA_sv_mc_lr:
            self._algo = self._algoClass(self._dataset_binned, self._meta_data, self._bin_times, numb_shared_lat, numb_grouped_lat, numb_trial_lat,\
                 max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step)
        else:
            self._algo = self._algoClass(self._dataset_binned, self._meta_data, self._bin_times, numb_trial_lat, max_EM_iterations, numb_inducing,\
                 learn_kernel_params, plot_free_energy, save_ortho_step)


    def initialise_algo_to_gen_param(self):
        """
        Initialises the algorithm with the generating parameters.

        """

        self._algo._intialise_to_gen_param = True
        

    def extract_path(self):
        """

        """
        if self._algoClass is GPFA_sv_mc  or self._algoClass is GPFA_sv_mc_lr:
            self._found_shared, self._found_cond, self._found_latents = self._algo.extract_neural_path(self._verbose)
        else:
            self._found_latents = self._algo.extract_neural_path(self._verbose)
            self._found_shared = None
            self._found_cond = None


    def get_orthogonality_scores(self):
        """
        Get the orthogonality scores of the learnt C matrix.
        """

        return get_orthogonality_score(self._algo._C_matrix, self._verbose)


    def compared_generating_and_found(self):
        """
        This function's aim is compare the found parameters with the ones used to generate the dataset.

        """
        
        # comparing the bias:
        if not self._algoClass is FA or self._algoClass is GPFA_sv_mc_lr:
            diff = self._algo._d_bias - self._meta_data["d_bias"]

            mean_1 = np.mean(np.abs(diff))
            std_1 = np.std(np.abs(diff))
            mean_2 = np.mean(np.square(diff))
            std_2 = np.std(np.square(diff))

            print("Comparing the means of the d_bias: generating: ", np.mean(np.abs(self._meta_data["d_bias"])), " found: ", np.mean(np.abs(self._algo._d_bias)))
            print("mean and std of l1 error: ", mean_1, std_1, "mean and std of l2 error: ", mean_2, std_2)

        # first shared latents:
        if self._numb_shared_lat > 0 or self._meta_data["K_alpha"]:
            """
            Could do things specific to alpha.
            """
            #print("\nAlpha: ")
            #self._comparing_C_matrices(self._meta_data["C_alpha"], self._algo._C_alpha)

        # then group latents:
        if  self._numb_grouped_lat > 0 or self._meta_data["K_beta"] > 0: 
            """
            Could do things specific to beta.
            """
            #print("\nBeta: ")
            #self._comparing_C_matrices(self._meta_data["C_beta"], self._algo._C_beta)
            
        # finally the trial latents:
        if  self._numb_trial_lat > 0 or self._meta_data["K_gamma"] > 0: 
            """
            Could do things specific to gamma.
            """
            #print("\nGamma: ")
            #self._comparing_C_matrices(self._meta_data["C_gamma"], self._algo._C_matrix)


        found_big_C = self._algo._C_matrix
        gener_big_C = np.concatenate([self._meta_data["C_alpha"], self._meta_data["C_beta"], self._meta_data["C_gamma"]], axis=1)

        print("\n Big, C: ")
        self._comparing_C_matrices(gener_big_C, found_big_C)
        

    def _comparing_C_matrices(self, original_C, found_C):
        """
        Here we compare the C matrices.

        """
        print("          othogonality stats of the generating C:")
        get_orthogonality_score(original_C)
        print("          othogonality stats of the learnt C:")
        get_orthogonality_score(found_C)

        print("The angle between the subspaces spanned by the two matrices is:",get_angle_subspaces(original_C,found_C,False))
        

        # the SVD projections:
        if False and original_C.shape == found_C.shape:

            U, S, V = np.linalg.svd(original_C)

            m, n = original_C.shape

            Smat = np.zeros_like(original_C)
            Smat[:n,:n] = np.diag(S)

            print(np.mean(np.abs(found_C - U.dot(Smat))))

            # we take the abs of the matrices because they can have flipped axis
            print("Diff between found C and UxS: ",np.mean(np.abs(np.abs(found_C) - np.abs(U.dot(Smat))) ))
            print("Diff between found C and gen: ",np.mean(np.abs(np.abs(found_C) -  np.abs(original_C) )) )

    def compare_found_covars_with_GPFA(self):
        """
        Compare the full covars found by the algorithm with the one found by GPFA.

        """

        if self._algo._gpfa_covariance is None:
            # get the covariance from the GPFA initialised at the generating params
            
            algo_handler = AlgoHandler(GPFA, self._dataset_name, 0, 0, self._numb_trial_lat, 20, 10,
                        10, False, 0, 0, -1)

            algo_handler.initialise_algo_to_gen_param()
            algo_handler.extract_path()

            self._algo._gpfa_covariance = algo_handler._algo._full_covars

        return self._algo.compare_S_with_GPFA(self._verbose)



    def get_orth_c_list(self):
        """
        Returns the list of measurements done during the fitting of the algorithm.
        """
        return self._algo._orth_c_list, self._algo._angle_list, self._algo._free_tupple_list

    def get_current_Free_KL_like(self, algo = None, return_as_string = False):
        """
        """
        if algo is None:
            algo = self._algo
        self._algoClass = type(algo)


        # free_full, like_full, kl_full, true_likelihood, post_kl = algo._compute_free_energy_full()
        free_full, like_full, kl_full = algo._compute_free_energy_full()

        if self._algoClass is not GPFA:
            free = algo._compute_free_energy()
            like = algo._compute_likelihood_term()
            kl = algo._compute_KL_term()
        else:
            free = free_full
            like = like_full
            kl = kl_full

        #  expected_likelihood = algo.compute_the_expected_log_marginal()
        true_likelihood = algo.compute_likelihood()
        post_kl = true_likelihood - free_full

        if self._verbose:
            print("Not full: ", free, like, kl)
            print("    Full: ", free_full, like_full, kl_full)
            print(" estimat: ", free_full, true_likelihood, post_kl)
            # print("\n")

        if return_as_string:
            return str(free)+","+str(like)+","+str(kl)+","+str(free_full)+","+str(like_full)+","+str(kl_full)+","+str(true_likelihood)+","+str(post_kl)
        else:
            return free, like, kl, free_full, like_full, kl_full

    def get_current_Free_KL_like2(self, algo = None, return_as_string = False):
        """
        """
        if algo is None:
            algo = self._algo

        # free_full, like_full, kl_full, true_likelihood, post_kl = algo._compute_free_energy_full()
        free_full, like_full, kl_full = algo._compute_free_energy_full()

        true_likelihood = algo.compute_likelihood()
        post_kl = true_likelihood - free_full

        if return_as_string:
            return str(free_full)+","+str(like_full)+","+str(kl_full)+","+str(true_likelihood)+","+str(post_kl)
        else:
            return free_full, like_full, kl_full, true_likelihood, post_kl
            

    def get_current_Free_KL_like_quick(self, algo = None, return_as_string = False):
        """
        """
        if algo is None:
            algo = self._algo

        # free_full, like_full, kl_full, true_likelihood, post_kl = algo._compute_free_energy_full()
        free_full, like_full, kl_full = algo._compute_free_energy_full()


        if return_as_string:
            return str(free_full)+","+str(like_full)+","+str(kl_full)
        else:
            return free_full, like_full, kl_full,


    def get_current_likelihood_and_co(self):
        """
        """

        expected_likelihood = self._algo.compute_the_expected_log_marginal()

        true_likelihood = self._algo.compute_likelihood()

        if self._verbose:
            print(expected_likelihood, true_likelihood)

        return expected_likelihood, true_likelihood

    def _get_orig_C_error(self):
        """
        """
        norm_func = np.abs
        numb_trials_train, numb_neurons, numb_bins = self._dataset_binned.shape
        numb_trials_test = self._dataset_binned_test.shape[0]

        if "C_alpha" in self._meta_data:
            lno_cv_error_regression_orig = 0
            for neuron_i in range(0, self._numb_neurons):
                for trial_i in range(0, numb_trials_test):
                    all_but_one = np.arange(self._numb_neurons)!= neuron_i
                    C = np.concatenate([self._meta_data["C_alpha"] , self._meta_data["C_beta"] , self._meta_data["C_gamma"] ], axis=1)
                    d = self._meta_data["d_bias"].copy()
                    try:
                        # this is retriveing the latent without bayesian linear regression
                        latent = np.linalg.inv( np.matmul(np.transpose(C[all_but_one,:]), C[all_but_one,:] ) )
                        temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,:].transpose(), d[all_but_one].transpose() )
                        latent = latent.dot(np.matmul(np.transpose(C[all_but_one,:]), temp.transpose() ) )

                        pred = C[neuron_i,:].dot(latent) + d[neuron_i]

                        lno_cv_error_regression_orig += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,:] ) ) * (1 / (self._numb_neurons * numb_trials_test))
                    except np.linalg.LinAlgError:
                        print("lin alg error")
                        ei = 9+ self._meta_data["d_bias"][neuron_i]
                    except:
                        print("other error")

            return lno_cv_error_regression_orig
        else:
            return 0



    def compute_dataset_errors_from_C(self, algo = None, return_as_string = False):
        """
        Computes the train error and the test error in the case where one trial (or more) was left out.
        
        TODO: use a getters to get C and d from the algo class
        """
        if algo is None:
            algo = self._algo

        norm_func = np.abs
        
        numb_trials_train, numb_neurons, numb_bins = self._dataset_binned.shape
        numb_trials_test = self._dataset_binned_test.shape[0]

        lno_cv_error_fa = 0
        lno_cv_error_gp = 0
        lno_cv_error_sv = 0

        lno_cv_error_post = 0

        C = algo._C_matrix.copy()
        d = algo._d_bias.copy()
        R = algo._R_noise.copy()
        R_inv = np.linalg.inv(R)


        ###### get bayesian and posterior error  #############
        number_of_neurons_tested = min(4, self._numb_neurons)# // 10
        for neuron_i in range(0, number_of_neurons_tested):
            for trial_i in range(0, numb_trials_test):
                all_but_one = np.arange(self._numb_neurons)!= neuron_i

                C_temp = C[all_but_one,:]

                R_n = R[:,all_but_one]
                R_inv = np.linalg.inv(R_n[all_but_one,:])

                CRinv = np.matmul( np.transpose(C_temp) , R_inv )
                CRinvC = np.matmul( CRinv, C_temp )

                # first part of the computation:
                if type(algo) is GPFA_sv or type(algo) is GPFA_sv_lr:
                    m_sv = self._compute_first_part_posterior(algo, C_temp, R_inv, CRinvC, CRinv, d, numb_bins, trial_i ,all_but_one, "")

                x_gp = self._compute_first_part_posterior(algo, C_temp, R_inv, CRinvC, CRinv, d, numb_bins, trial_i ,all_but_one, "GPFA")

                for bin_i in range(0, numb_bins):
                    diff_i = np.subtract(self._dataset_binned_test[trial_i,all_but_one,bin_i].transpose(), d[all_but_one].transpose() )

                    # get the latent using the posterior formula:
                    latent_gp = x_gp[:,bin_i]
                    latent_fa = np.linalg.inv( CRinvC + np.eye(CRinvC.shape[0]) ).dot(np.matmul(CRinv, diff_i.transpose() ))
                    
                    pred_gp = C[neuron_i,:].dot(latent_gp) + d[neuron_i]
                    pred_fa = C[neuron_i,:].dot(latent_fa) + d[neuron_i]
                                        
                    lno_cv_error_fa += np.mean( norm_func( pred_fa - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) 
                    lno_cv_error_gp += np.mean( norm_func( pred_gp - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) 

                    if type(algo) is GPFA_sv or type(algo) is GPFA_sv_lr:
                        latent_sv = algo._temp_param_K_i_list[bin_i].dot(m_sv)
                        pred_sv = C[neuron_i,:].dot(latent_sv) + d[neuron_i]
                        lno_cv_error_sv += np.mean( norm_func( pred_sv - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) 

        lno_cv_error_fa *= (1 / (number_of_neurons_tested * numb_trials_test * numb_bins))
        lno_cv_error_gp *= (1 / (number_of_neurons_tested * numb_trials_test * numb_bins))
        lno_cv_error_sv *= (1 / (number_of_neurons_tested * numb_trials_test * numb_bins))
        # lno_cv_error_post *= (1 / (self._numb_neurons * numb_trials_test * numb_bins))

        if self._verbose:
            # print(lno_cv_error_bayes, lno_cv_error_bayes_1, lno_cv_error_bayes_2, lno_cv_error_posterior)
            # print("lno_cv_error_posterior: ", lno_cv_error_posterior)
            print(str(lno_cv_error_fa) + " , " + str(lno_cv_error_gp) +" , " + str(lno_cv_error_sv))

        if return_as_string:
            return str(lno_cv_error_fa) + "," + str(lno_cv_error_gp) +"," + str(lno_cv_error_sv)
        else:
            return lno_cv_error_fa, lno_cv_error_gp, lno_cv_error_sv



    def _compute_second_part_posterior(self, algo, C, R_inv, CRinvC, CRinv, d, numb_bins, trial_i ,all_but_one, bin_i, m):
        """
        Returns the time independent part of the posterior.
        """

        self._algoClass = type(algo)
        latent = 1
        if self._algoClass is FA:
            latent = 1
        elif self._algoClass is GPFA:
            latent = m[:,bin_i]
        elif self._algoClass is GPFA_sv or self._algoClass is GPFA_sv_lr:
            latent = algo._temp_param_K_i_list[bin_i].dot(m)
        elif self._algoClass is GPFA_sv_mc or self._algoClass is GPFA_sv_mc_lr:
            latent = 1

        return latent

    def _compute_first_part_posterior(self, algo, C, Rinv, CRinvC, CRinv, d, numb_bins, trial_i ,all_but_one, command):
        """
        Returns the time independent part of the posterior.
        """
        self._algoClass = type(algo)
        latent = 1
        if self._algoClass is FA:
            latent = 1
        elif self._algoClass is GPFA or command == "GPFA":
            
            # create C tilde:
            C_tilde = np.zeros((algo._numb_bins*(algo._numb_neurons-1),algo._K_numb_latents*algo._numb_bins))
            for lat_k in range(0,algo._K_numb_latents):
                C_k = np.kron(np.eye(algo._numb_bins), C[:,lat_k])
                # print( (C_k != 0).astype(int) )
                C_tilde[:,algo._numb_bins*lat_k:algo._numb_bins*(lat_k+1)] = C_k.transpose()

            Rinv_big = np.kron(np.eye(algo._numb_bins), Rinv)

            CRinv = np.matmul( np.transpose(C_tilde), Rinv_big )

            for lat_k in range(0,algo._K_numb_latents):
                K_xx = make_K(self._bin_times, self._bin_times, algo._kernel_func, algo._kernel_param, lat_k)
                inv_Kxx = np.linalg.inv(K_xx)
                if lat_k == 0:
                    inv_K_tilde = inv_Kxx
                else:
                    inv_K_tilde = block_diag(inv_K_tilde, inv_Kxx)

            Sigma_x_inv = inv_K_tilde + np.transpose(C_tilde).dot(Rinv_big).dot(C_tilde)

            Sigma_x = np.linalg.inv( Sigma_x_inv ) 

            algo._temp_Sigma_x = Sigma_x

            all_but_y = Sigma_x.dot(CRinv)
            d_big = np.kron(np.ones(algo._numb_bins), d[all_but_one])

            y_big = np.transpose(self._dataset_binned_test[trial_i,all_but_one,:]).reshape(algo._numb_bins*(algo._numb_neurons-1))

            big_diff = y_big - d_big

            X_big = np.matmul(all_but_y, big_diff)

            # X_big_by = np.matmul(all_but_y_by, big_diff)

            latent = np.transpose(X_big.reshape(algo._numb_bins, algo._K_numb_latents))
            latent = np.zeros((algo._K_numb_latents, algo._numb_bins))
            for lat_k in range(0,algo._K_numb_latents):
                latent[lat_k,:] = X_big[algo._numb_bins*lat_k:algo._numb_bins*(lat_k+1)]

        elif self._algoClass is GPFA_sv or self._algoClass is GPFA_sv_lr:
            first_term_Ti = 0
            second_term = 0
            for bin_i in range(0, numb_bins):

                first_term_Ti += np.transpose(algo._temp_param_K_i_list[bin_i]).dot(CRinvC).dot(algo._temp_param_K_i_list[bin_i])
                second_term += np.transpose(algo._temp_param_K_i_list[bin_i]).dot(CRinv).dot( self._dataset_binned_test[trial_i,all_but_one,bin_i] - d[all_but_one] )
            first_term_Ti = np.linalg.inv( first_term_Ti + np.linalg.inv(algo._temp_param_K_tilde) )

            latent = first_term_Ti.dot(second_term)
        elif self._algoClass is GPFA_sv_mc or self._algoClass is GPFA_sv_mc_lr:
            latent = 1

        return latent




    def compute_dataset_errors_from_C_old(self):
        """
        Computes the train error and the test error in the case where one trial (or more) was left out.
        
        TODO: use a getters to get C and d from the algo class
        """

        norm_func = np.abs
        
        numb_trials_train, numb_neurons, numb_bins = self._dataset_binned.shape
        numb_trials_test = self._dataset_binned_test.shape[0]

        lno_cv_error_regression_orig = 0
        lno_cv_error_regression = 0
        lno_cv_error_bayes_regression = 0
        lno_cv_error_posterior = 0


        # the leave one neuron out cross val error:
        if "C_alpha" in self._meta_data:
            lno_cv_error_regression_orig = 0
            for neuron_i in range(0, self._numb_neurons):
                for trial_i in range(0, numb_trials_test):
                    all_but_one = np.arange(self._numb_neurons)!= neuron_i
                    C = np.concatenate([self._meta_data["C_alpha"] , self._meta_data["C_beta"] , self._meta_data["C_gamma"] ], axis=1)
                    d = self._meta_data["d_bias"].copy()
                    try:
                        # this is retriveing the latent without bayesian linear regression
                        latent = np.linalg.inv( np.matmul(np.transpose(C[all_but_one,:]), C[all_but_one,:] ) )
                        temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,:].transpose(), d[all_but_one].transpose() )
                        latent = latent.dot(np.matmul(np.transpose(C[all_but_one,:]), temp.transpose() ) )

                        pred = C[neuron_i,:].dot(latent) + d[neuron_i]

                        lno_cv_error_regression_orig += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,:] ) ) * (1 / (self._numb_neurons * numb_trials_test))
                    except np.linalg.LinAlgError:
                        print("lin alg error")
                        ei = 9+ self._meta_data["d_bias"][neuron_i]
                    except:
                        print("other error")

            if self._verbose: 
                print("lno_cv_error_regression_orig: ",lno_cv_error_regression_orig)

        # the leave one neuron out cross val error:
        lno_cv_error_regression = 0
        for neuron_i in range(0, self._numb_neurons):
            for trial_i in range(0, numb_trials_test):
                all_but_one = np.arange(self._numb_neurons)!= neuron_i
                C = self._algo._C_matrix.copy()
                d = self._algo._d_bias.copy()
                # try:
                # this is retriveing the latent without bayesian linear regression
                latent = np.linalg.inv( np.matmul(np.transpose(C[all_but_one,:]), C[all_but_one,:] ) )
                temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,:].transpose(), d[all_but_one].transpose() )
                latent = latent.dot(np.matmul(np.transpose(C[all_but_one,:]), temp.transpose() ) )

                pred = C[neuron_i,:].dot(latent) + d[neuron_i]

                lno_cv_error_regression += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,:] ) ) * (1 / (self._numb_neurons * numb_trials_test))
                # except:
                #     ei = 9+ self._meta_data["d_bias"][neuron_i]
                #     print("bad")

        if self._verbose: 
            print("lno_cv_error_regression: ",lno_cv_error_regression)

        # the leave one neuron out cross val error:
        # if self._algoClass is not FA:
        #     lno_cv_error_bayes_regression = 0
        #     for neuron_i in range(0, self._numb_neurons):
        #         for trial_i in range(0, numb_trials_test):
        #             all_but_one = np.arange(self._numb_neurons)!= neuron_i
        #             C = self._algo._C_matrix.copy()
        #             d = self._algo._d_bias.copy()
        #             for bin_i in range(0, numb_bins):
        #                 # try:
        #                 # this is retriveing the latent with bayesian linear regression
        #                 if self._algoClass is GPFA:
        #                     matr_S = self._algo._Vsm[:,:,bin_i]
        #                 else:
        #                     matr_S = np.matmul( np.matmul(self._algo._temp_param_K_i_list[bin_i], self._algo._S_full_matr), np.transpose(self._algo._temp_param_K_i_list[bin_i]) )
                        
        #                 # print(self._algo._temp_param_K_i_list[bin_i].shape)
        #                 # print(matr_S.shape)
                        
        #                 latent = np.linalg.inv( np.matmul(np.transpose(C[all_but_one,:]), C[all_but_one,:] ) + (matr_S ) )
        #                 temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,bin_i].transpose(), d[all_but_one].transpose() )
        #                 latent = latent.dot(np.matmul(np.transpose(C[all_but_one,:]), temp.transpose() ) )

        #                 pred = C[neuron_i,:].dot(latent) + d[neuron_i]

        #                 lno_cv_error_bayes_regression += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) * (1 / (self._numb_neurons * numb_trials_test * numb_bins))
        #                 # except:
        #                 #     print("a...")
        #                 #     ei = 9
        
        #     if self._verbose: 
        #         print("lno_cv_error_bayes_regression: ", lno_cv_error_bayes_regression)
        if self._algoClass is not FA:
            lno_cv_error_bayes_regression = 0
            C = self._algo._C_matrix.copy()
            d = self._algo._d_bias.copy()
            R = self._algo._R_noise.copy()
            R_inv = np.linalg.inv(R)

            for neuron_i in range(0, self._numb_neurons):
                for trial_i in range(0, numb_trials_test):
                    all_but_one = np.arange(self._numb_neurons)!= neuron_i

                    C_temp = C[all_but_one,:]

                    R_n = R[:,all_but_one]
                    R_inv = np.linalg.inv(R_n[all_but_one,:])

                    CRinv = np.matmul( np.transpose(C_temp) , R_inv )
                    CRinvC = np.matmul( CRinv, C_temp )
                    
                    for bin_i in range(0, numb_bins):
                        
                        temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,bin_i].transpose(), d[all_but_one].transpose() )
                        # try:
                        # this is retriveing the latent with bayesian linear regression
                        if self._algoClass is GPFA:
                            matr_S = self._algo._Vsm[:,:,bin_i]
                        # elif True:
                            # matr_S = np.matmul( np.matmul(self._algo._temp_param_K_i_list[bin_i], self._algo._S_full_matr), np.transpose(self._algo._temp_param_K_i_list[bin_i]) )
                            # matr_S = np.linalg.inv( np.matmul(np.transpose(C[all_but_one,:]), C[all_but_one,:] ) + (matr_S ) )

                            # latent = matr_S.dot(np.matmul(np.transpose(C[all_but_one,:]), temp.transpose() ) )

                        else:
                            first_term_Ti = 0
                            second_term = 0
                            for bin_i_2 in range(0, numb_bins):
                                first_term_Ti += np.transpose(self._algo._temp_param_K_i_list[bin_i_2]).dot(CRinvC).dot(self._algo._temp_param_K_i_list[bin_i_2])
                                second_term += np.transpose(self._algo._temp_param_K_i_list[bin_i_2]).dot(CRinv).dot( self._dataset_binned_test[trial_i,all_but_one,bin_i_2] - d[all_but_one] )
                            first_term_Ti = np.linalg.inv( first_term_Ti + np.linalg.inv(self._algo._temp_param_K_tilde) )
                            m = first_term_Ti.dot(second_term)

                            matr_S = np.matmul( np.matmul(self._algo._temp_param_K_i_list[bin_i], self._algo._S_full_matr), np.transpose(self._algo._temp_param_K_i_list[bin_i]) )
                            #matr_S = np.linalg.inv( matr_S)

                            inv_map = np.linalg.inv( CRinvC + matr_S)
                            # inv_map = np.linalg.inv( CRinvC + 8.679375421852991e-06*np.eye(CRinvC.shape[0]) )

                            latent = inv_map.dot(np.matmul(CRinv, temp.transpose() ) )

                            #latent += np.matmul( matr_S, np.matmul(self._algo._temp_param_K_i_list[bin_i], m))

                            #print((matr_S!=0).astype(int))
                            #print(np.mean(matr_S))
                            
                        

                        pred = C[neuron_i,:].dot(latent) + d[neuron_i]

                        lno_cv_error_bayes_regression += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) * (1 / (self._numb_neurons * numb_trials_test * numb_bins))
                        # except:
                        #     print("a...")
                        #     ei = 9
        
            if self._verbose: 
                print("lno_cv_error_bayes_regression: ", lno_cv_error_bayes_regression)
        else:
            lno_cv_error = 0
            C = self._algo._C_matrix.copy()
            d = self._algo._d_bias.copy()
            R = self._algo._R_noise.copy()
            R_inv = np.linalg.inv(R)
            for neuron_i in range(0, self._numb_neurons):
                for trial_i in range(0, numb_trials_test):
                    all_but_one = np.arange(self._numb_neurons)!= neuron_i

                    R_n = R[:,all_but_one]
                    R_inv = np.linalg.inv(R_n[all_but_one,:])

                    CRinv = np.matmul( np.transpose(C[all_but_one,:]) , R_inv )
                    CRinvC = np.matmul( CRinv, C[all_but_one,:] )
                    matr_Sig = np.linalg.inv(  CRinvC) #np.eye(self._numb_trial_lat) +
                    for bin_i in range(0, numb_bins):
                        # try:
                        # this is retriveing the latent with the posterior formulation
                                                
                        temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,bin_i].transpose(), d[all_but_one].transpose() )
                        latent = matr_Sig.dot(np.matmul(CRinv, temp.transpose() ) )

                        pred = C[neuron_i,:].dot(latent) + d[neuron_i]

                        lno_cv_error += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) * (1 / (self._numb_neurons * numb_trials_test * numb_bins))
                        # except:
                        #     print("a...")
                        #     ei = 9
            if self._verbose: 
                print("lno_cv_error: ", lno_cv_error)

        # get the error using the posterior mean:
        lno_cv_error_posterior = 0
        C = self._algo._C_matrix.copy()
        d = self._algo._d_bias.copy()
        R = self._algo._R_noise.copy()
        R_inv = np.linalg.inv(R)

        for neuron_i in range(0, self._numb_neurons):
            for trial_i in range(0, numb_trials_test):
                all_but_one = np.arange(self._numb_neurons)!= neuron_i

                C_temp = C[all_but_one,:]

                R_n = R[:,all_but_one]
                R_inv = np.linalg.inv(R_n[all_but_one,:])

                CRinv = np.matmul( np.transpose(C_temp) , R_inv )
                CRinvC = np.matmul( CRinv, C_temp )

                # first part of the computation:
                if self._algoClass is FA:
                    latent = 1
                elif self._algoClass is GPFA:
                    latent = 1
                elif self._algoClass is GPFA_sv or self._algoClass is GPFA_sv_lr:
                    first_term_Ti = 0
                    second_term = 0
                    for bin_i in range(0, numb_bins):
                        first_term_Ti += np.transpose(self._algo._temp_param_K_i_list[bin_i]).dot(CRinvC).dot(self._algo._temp_param_K_i_list[bin_i])
                        second_term += np.transpose(self._algo._temp_param_K_i_list[bin_i]).dot(CRinv).dot( self._dataset_binned_test[trial_i,all_but_one,bin_i] - d[all_but_one] )
                    first_term_Ti = np.linalg.inv( first_term_Ti + np.linalg.inv(self._algo._temp_param_K_tilde) )

                    m = first_term_Ti.dot(second_term)
                elif self._algoClass is GPFA_sv_mc or self._algoClass is GPFA_sv_mc_lr:
                    latent = 1

                for bin_i in range(0, numb_bins):

                    latent = 0
                    # get the latent using the posterior formula:
                    if self._algoClass is FA:
                        latent = 1
                    elif self._algoClass is GPFA:
                        latent = 1
                    elif self._algoClass is GPFA_sv or self._algoClass is GPFA_sv_lr:
                        latent = self._algo._temp_param_K_i_list[bin_i].dot(m)
                    elif self._algoClass is GPFA_sv_mc or self._algoClass is GPFA_sv_mc_lr:
                        latent = 1

                    
                    # temp = np.subtract(self._dataset_binned_test[trial_i,all_but_one,bin_i].transpose(), d[all_but_one].transpose() )
                    # latent = matr_Sig.dot(np.matmul(CRinv, temp.transpose() ) )

                    pred = C[neuron_i,:].dot(latent) + d[neuron_i]

                    lno_cv_error_posterior += np.mean( norm_func( pred - self._dataset_binned_test[trial_i,neuron_i,bin_i] ) ) * (1 / (self._numb_neurons * numb_trials_test * numb_bins))
        
        if self._verbose:              
            print("lno_cv_error_posterior: ", lno_cv_error_posterior)

        # if self._verbose:
            # print(lno_cv_error_san_S, lno_cv_error_san_S_orig, lno_cv_error)
        return lno_cv_error_regression_orig, lno_cv_error_regression, lno_cv_error_bayes_regression, lno_cv_error_posterior




    def compute_dataset_errors_from_lat(self):
        """
        Computes the train error and the test error in the case where one trial (or more) was left out.
        
        TODO: use a getters to get C and d from the algo class
        """
        #self._found_latents = self._algo._recover_path_for_model()
        # self.extract_path()

        train_error = 0
        train_error_avr = 0
        test_error = 0

        numb_trials_train, numb_neurons, numb_bins = self._dataset_binned.shape
        numb_trials_test = self._dataset_binned_test.shape[0]

        # first the 'true' training error
        for trial_i in range(0, numb_trials_train):
            prediction = np.matmul(self._algo._C_matrix , self._found_latents[trial_i,:,:]) + np.reshape(self._algo._d_bias, (numb_neurons,1))
            #print(prediction.shape)
            train_error += np.mean(np.abs(prediction - self._dataset_binned[trial_i,:,:]))

        train_error *= 1 / numb_trials_train

        # then the error on the training set when averaging the latents
        avr_lat = np.mean(self._found_latents, 0)
        
        for trial_i in range(0, numb_trials_train):
            prediction = np.matmul(self._algo._C_matrix , avr_lat) + np.reshape(self._algo._d_bias, (numb_neurons,1))
            train_error_avr += np.mean(np.abs(prediction - self._dataset_binned[trial_i,:,:]))

        train_error_avr *= 1 / numb_trials_train

        # finally the error on the eventual test sets
        for trial_i in range(0, numb_trials_test):
            prediction = np.matmul(self._algo._C_matrix , avr_lat) + np.reshape(self._algo._d_bias, (numb_neurons,1))
            test_error += np.mean(np.abs(prediction - self._dataset_binned_test[trial_i,:,:]))
        test_error *= 1 / numb_trials_test


        if self._verbose:
            print(train_error, train_error_avr, test_error)
        return train_error, train_error_avr, test_error




    def plot_path_real_data(self):
        """
        Plot the found latents on top of another because real dataset.
        """

        latents = self._algo._recover_path_for_model()

        numb_plots_hor = ( self._numb_trial_lat )
        if numb_plots_hor == 1:
            numb_plots_hor += 1
        fig, axes = plt.subplots(1, numb_plots_hor)
        
        for trial_i in range(0,  self._numb_trials - self._numb_test_trials ):
            
            for lat_i in range(0,self._numb_trial_lat):
                axes[lat_i].plot(self._bin_times, latents[trial_i,lat_i,:], 'b', linewidth=0.3)
                axes[lat_i].set_ylim([-4,4])
                axes[lat_i].set_title("T:"+str(self._algo._get_tau_param()[lat_i]))
        fig.suptitle('Main title')

        latents = self._algo._recover_path_for_orthonorm_model()

        numb_plots_hor = ( self._numb_trial_lat )
        if numb_plots_hor == 1:
            numb_plots_hor += 1
        fig, axes = plt.subplots(1, numb_plots_hor)
        
        for trial_i in range(0,  self._numb_trials - self._numb_test_trials ):
            
            for lat_i in range(0,self._numb_trial_lat):
                axes[lat_i].plot(self._bin_times, latents[trial_i,lat_i,:], 'b', linewidth=0.3)
                axes[lat_i].set_ylim([-4,4])
                axes[lat_i].set_title("T:"+str(self._algo._taus_ordered[lat_i]))
        fig.suptitle('Orthonormalised latents')

        plt.show()



    def plot_path(self):
        """
        Plot the found latents (compared to the true ones in case they are known)

        """

        if self._dataset_is_binned: # this means reall data
            self.plot_path_real_data()
            return
        
        # self._numb_shared_lat = numb_shared_lat
        # self._numb_grouped_lat = numb_grouped_lat
        # self._numb_trial_lat = numb_trial_lat

        # first shared latents:
        if self._numb_shared_lat > 0 or self._meta_data["K_alpha"]:

                numb_plots_hor = max(self._numb_shared_lat,self._meta_data["K_alpha"])
                if numb_plots_hor == 1:
                    numb_plots_hor += 1

                fig, axes = plt.subplots(2, numb_plots_hor)

                for lat_i in range(0,self._meta_data["K_alpha"]):
                    axes[0, lat_i].plot(self._true_latent_alpha[lat_i,:])
                for lat_i in range(0,self._numb_shared_lat):
                    axes[1, lat_i].plot(self._bin_times,self._found_shared[lat_i,:])

                fig.suptitle("Shared latents : ")

        # then group latents:
        if  self._numb_grouped_lat > 0 or self._meta_data["K_beta"]: 
            
            numb_plots_hor = (max(self._numb_grouped_lat,self._meta_data["K_beta"]) + 1) * self._numb_conditions 
            if numb_plots_hor == 1:
                numb_plots_hor += 1
            fig, axes = plt.subplots(2, numb_plots_hor)
            counter = 0
            for cond_i in range(0, self._numb_conditions):
                for lat_i in range(0,self._meta_data["K_beta"]):
                    axes[0, counter+lat_i].plot(self._true_latent_beta[cond_i,lat_i,:])
                    #axes[1].set_ylim([-2.5, 2.5])
                for lat_i in range(0,self._numb_grouped_lat):
                    axes[1, counter+lat_i].plot(self._bin_times,self._found_cond[cond_i,lat_i,:])

                counter += max(self._numb_grouped_lat,self._meta_data["K_beta"]) + 1

            fig.suptitle("Group latents : ")

        # finally the trial latents:
        if  self._numb_trial_lat > 0 or self._meta_data["K_gamma"]: 
            
            numb_plots_hor = (max(self._numb_trial_lat,self._meta_data["K_gamma"]))
            if numb_plots_hor == 1:
                numb_plots_hor += 1
            fig, axes = plt.subplots(min(self._numb_trials - self._numb_test_trials, 20), numb_plots_hor)
            
            for trial_i in range(0, min(self._numb_trials - self._numb_test_trials, 20)):
                for lat_i in range(0,self._meta_data["K_gamma"]):
                    axes[trial_i, lat_i].plot(self._bin_times, self._true_latent_gamma[trial_i,lat_i,:])
                    #plt.hold(True)
                    #axes[1].set_ylim([-2.5, 2.5])
                for lat_i in range(0,self._numb_trial_lat):
                    axes[trial_i, lat_i].plot(self._bin_times,self._found_latents[trial_i,lat_i,:])

            fig.suptitle("Trial latents : ")


        plt.show()


    def plot_binned(self):
        """
        Just a small function to plot the binned dataset compared to the real one:

        """
        neuron = 33
        for trial_i in range(0, self._numb_trials):
            plt.plot(self._bin_times,self._dataset_binned[trial_i,neuron,:])
            #plt.plot(self._dataset[trial_i,neuron,:])
        plt.show()


    def _bin_dataset(self, numb_test_trials):
        """
        Creates a binned dataset from the dataset.

        """

        if "binned_with" in self._meta_data:
            bin_times = np.arange(0, self._meta_data["total_duration"], self._meta_data["binned_with"])
            self._bin_times = bin_times
            self._meta_data["bin_times"] = bin_times

            self._dataset_binned = self._dataset[0:self._numb_trials-numb_test_trials,:,:]
            self._dataset_binned_test = self._dataset[self._numb_trials-numb_test_trials:,:,:]

            self._true_latent_alpha = self._meta_data["latent_alpha"]
            self._true_latent_beta = self._meta_data["latent_beta"]
            self._true_latent_gamma = self._meta_data["latent_gamma"]

        elif self._dataset_is_binned:
            return
            
        else:
            # create the new dataset:
            numb_bins = self._total_duration // self._bin_width
            dataset_binned = np.zeros((self._numb_trials, self._numb_neurons, numb_bins))
            bin_times = np.zeros((numb_bins))

            for trial_i in range(0,self._numb_trials):
                bin_i = 0
                timestep_i = 0

                while timestep_i < self._total_duration:

                    dataset_binned[trial_i,:,bin_i] = np.sum(self._dataset[trial_i,:,timestep_i:(timestep_i+self._bin_width)],axis=1)
                    bin_times[bin_i] = timestep_i + (self._bin_width / 2)

                    bin_i += 1
                    timestep_i += self._bin_width


            self._dataset_binned = dataset_binned[0:self._numb_trials-numb_test_trials]
            self._dataset_binned_test = dataset_binned[self._numb_trials-numb_test_trials:]
            self._bin_times = bin_times
            self._meta_data["bin_times"] = bin_times


    def _read_dataset(self,dataset_name):
        """
        Reads the dataset and the meta information about the dataset.

        """

        if "mat" in dataset_name:
            self._read_mat_dataset(dataset_name)
            return


        self._dataset = load_obj(dataset_name)

        self._meta_data = load_obj(dataset_name + '_meta')

        if not "binned_with" in self._meta_data:
            self._true_latent_alpha = load_obj(dataset_name + '_latents')

        self._generated_data = self._meta_data["generated"]

        # save some key points of the latent information:
        self._numb_trials = self._meta_data["numb_trials"]
        self._numb_neurons = self._meta_data["numb_neurons"]
        self._total_duration = self._meta_data["total_duration"]
        self._numb_conditions = self._meta_data["numb_conditions"]
        self._trial_conditions = self._meta_data["trial_conditions"]


    def _read_mat_dataset(self,dataset_name):
        """
        Reads the dataset and the meta information about the dataset.

        """

        thresh_spike_per_sec = 10


        mat = loadmat(dataset_name)

        dataset = mat["S"]
        numb_neurons, total_duration, numb_trials = dataset.shape

        ### determining the number of valide neurons:
        thresh_spiking = thresh_spike_per_sec * (total_duration / 1000)

        sum_firing_avr = np.sum(dataset,1)
        #print(sum_firing_avr.shape)

        avr_per_ner = np.mean(sum_firing_avr,1)

        numb_selected_neur = np.sum( avr_per_ner > thresh_spiking)
        print(numb_selected_neur)

        dataset_chosen = np.zeros(( numb_selected_neur, total_duration, numb_trials ))
        chosen_count = 0
        for neu_i in range(0, numb_neurons):
            
            if avr_per_ner[neu_i] > thresh_spiking:
                dataset_chosen[chosen_count,:,:] = dataset[neu_i,:,:]
        #         print("happened",chosen_count)
                chosen_count += 1


        ### create binned dataset:
        numb_trials = numb_trials // 1

        numb_bins = total_duration // self._bin_width
        dataset_binned = np.zeros(( numb_trials,  numb_selected_neur, numb_bins))
        bin_times = np.zeros((numb_bins))


        for trial_i in range(0, numb_trials):
            bin_i = 0
            timestep_i = 0

            while bin_i < numb_bins: #timestep_i <  total_duration:

                if timestep_i >  total_duration-self._bin_width:
                    dataset_binned[trial_i,:,bin_i] = np.sum( dataset_chosen[:,timestep_i:,trial_i],axis=1)
                else:
                    dataset_binned[trial_i,:,bin_i] = np.sum( dataset_chosen[:,timestep_i:(timestep_i+self._bin_width),trial_i],axis=1)
                bin_times[bin_i] = timestep_i + ( self._bin_width / 2)

                bin_i += 1
                timestep_i +=  self._bin_width


        dataset_binned = np.sqrt(dataset_binned)
        print(dataset_binned.shape)

        self._numb_trials = numb_trials
        self._numb_neurons = numb_selected_neur
        self._total_duration = total_duration
        self._numb_conditions = 1
        self._trial_conditions = []

        self._dataset_binned = dataset_binned[0:self._numb_trials-self._numb_test_trials]
        self._dataset_binned_test = dataset_binned[self._numb_trials-self._numb_test_trials:]
        self._bin_times = bin_times
        self._meta_data = {"total_duration": total_duration}
        self._meta_data["bin_times"] = bin_times
        self._meta_data["numb_trials"] = numb_trials
        self._meta_data["K_alpha"] = 0
        self._meta_data["K_beta"] = 0
        self._meta_data["K_gamma"] = 0

        self._dataset_is_binned = True



    # def _get_n_norm(self, matrix):
    #     """
    #     does stuff
    #     """

    #     len_1, len_2 = matrix.shape

    #     dot_pro_mat = np.zeros((len_2, len_2))

    #     for i in range(0, len_2):
    #         for j in range(0, len_2):
    #             i_v = i if i < 1 else i+1
    #             j_v = j+1
    #             dot_pro_mat[i,j] = np.dot(matrix[:,i], matrix[:,j])

    #     det_mat = np.linalg.det(dot_pro_mat)

    #     return np.sqrt(det_mat)


    # def _get_angle_subspaces(self, U, V):
    #     """
    #     """

    #     U = orth(U)
    #     V = orth(V)

    #     len_1_u, len_2_u = U.shape
    #     len_1_v, len_2_v = V.shape

    #     if len_2_u > len_2_v:
    #         a = U.copy()
    #         U = V.copy()
    #         V = a.copy()

    #         len_1_u, len_2_u = U.shape
    #         len_1_v, len_2_v = V.shape

    #     # normalizing subspaces:
    #     for i in range(0, len_2_u):
    #         U[:,i] *= (1 / np.linalg.norm(U[:,i]) )
    #     for i in range(0, len_2_v):
    #         V[:,i] *= (1 / np.linalg.norm(V[:,i]) )


    #     # get the projection of every vector in U on V:
    #     proj_U_on_V = np.zeros_like(U)

    #     for i in range(0, len_2_u):
    #         proj = 0
    #         for j in range(0, len_2_v):
    #             proj += V[:,j] * (np.dot(V[:,j], U[:,i]) / np.linalg.norm(V[:,j])**2)
        
    #         proj_U_on_V[:,i] = proj


    #     n_norm_proj = self._get_n_norm(proj_U_on_V)
    #     n_norm_U = self._get_n_norm(U)

    #     cos_square_theta = n_norm_proj**2 / (n_norm_U**2 + 1e-14)

    #     theta = np.sqrt(cos_square_theta)

    #     print("angle",np.rad2deg(np.arccos(theta)) )
