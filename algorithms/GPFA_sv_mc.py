"""

This is a test class that does only the condition specific (beta) mc-sv-GPFA.
(In the same way that sv-GPFA actually does only gamma)


"""

from algorithms.Algorithm import Algorithm
from algorithms.IGPFA import IGPFA
import numpy as np
import time
from tqdm import tqdm 
from utilis import *
from scipy.linalg import block_diag
import sklearn.decomposition
import matplotlib.pyplot as plt

# the auto grad things:
#import autograd.numpy as np
#import autograd.numpy.random as npr
from autograd import value_and_grad, grad

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
# np.core.arrayprint._line_width = 180

SMALL = 1e-7


class GPFA_sv_mc(IGPFA):
    """ sparce variational GPFA 
    
    """

    algName = "mc_sv_GPFA"



    def __init__(self, dataset, meta_data, bin_times, numb_shared_lat, numb_grouped_lat, numb_latents, \
                 max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy = 0, save_ortho_step = -1):
        """ Preparation of all the needed variables. """

        super().__init__(dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step)

        self._K_alpha = numb_shared_lat
        self._K_beta = numb_grouped_lat
        self._K_gamma = numb_latents

        self._trial_conditions = self._meta_data["trial_conditions"].astype(int)
        self._unique_conditions = np.unique(self._trial_conditions)
        self._numb_conditions = self._meta_data["numb_conditions"]

        # print(self._trial_conditions)
        # self._numb_trials = 2*np.int(self._numb_trials//self._numb_conditions)
        # self._dataset = self._dataset[0:self._numb_trials,:,:].copy()
        # self._trial_conditions = (self._trial_conditions == 1).astype(int).copy()
        # self._trial_conditions = np.zeros_like(self._trial_conditions)


        # just for when the conditions are wrong because of dataset generation:
        if self._trial_conditions[0] == -1:
            self._trial_conditions = np.zeros_like(self._trial_conditions)

        # get the number of trials each condition:
        # cond_numb_list = []
        # for cond_i in range(0,self._numb_conditions):
        #     cond_numb_list.append(np.sum(cond_i == self._trial_conditions))
        # self._condi_trial_count = np.array(cond_numb_list)
        
        
        self._numb_trial_per_cond = []
        for con_i in np.unique(self._trial_conditions):
            self._numb_trial_per_cond.append(np.sum(con_i == self._trial_conditions))
        # print(self._numb_trial_per_cond)

        # we have multiple C matrices in this model:
        self._C_alpha = np.random.rand(self._numb_neurons, self._K_alpha) / 2
        self._C_beta = np.random.rand(self._numb_neurons, self._K_beta) / 2
        self._C_gamma = np.random.rand(self._numb_neurons, self._K_gamma) / 2

        # self._C_alpha = self._meta_data["C_alpha"]
        # self._C_beta = self._meta_data["C_beta"]
        # self._C_gamma = self._meta_data["C_gamma"]


        # we have multiple m categories in this model:
        self._m_alpha_induc_mean = np.zeros((self._K_alpha, self._numb_inducing))
        self._m_beta_induc_mean = np.zeros((self._numb_conditions, self._K_beta, self._numb_inducing))
        self._m_gamma_induc_mean = np.zeros((self._numb_trials, self._K_gamma, self._numb_inducing))
        
        self._temp_param_m_alpha = np.zeros((self._K_alpha*self._numb_inducing))
        self._temp_param_m_beta = np.zeros((self._numb_conditions,self._K_beta*self._numb_inducing))
        self._temp_param_m_gamma = np.zeros((self._numb_trials,self._K_gamma*self._numb_inducing))

        # same for S_k
        self._S_alpha_covars = np.zeros((self._K_alpha, self._numb_inducing, self._numb_inducing))
        self._S_beta_covars = np.zeros((self._numb_conditions, self._K_beta, self._numb_inducing, self._numb_inducing))
        self._S_gamma_covars = np.zeros((self._numb_trials, self._K_gamma, self._numb_inducing, self._numb_inducing))

        # initialise GPs parameters:
        self._kernel_func = se_kernel
        
        tau_scales = 100 * np.ones((self._K_alpha))
        sigma_ns = 0.001 * np.ones((self._K_alpha))
        self._kernel_param_alpha = {"tau" : tau_scales, "sigma_n" : sigma_ns}
        if "tau_s_alpha" in self._meta_data:
            for lat_k in range(0, min(self._K_alpha, self._meta_data["K_alpha"])):
                self._kernel_param_alpha["tau"][lat_k] = self._meta_data["tau_s_alpha"][lat_k]

        tau_scales = 100 * np.ones((self._K_beta))
        sigma_ns = 0.001 * np.ones((self._K_beta))
        self._kernel_param_beta = {"tau" : tau_scales, "sigma_n" : sigma_ns}
        if "tau_s_beta" in self._meta_data:
            for lat_k in range(0, min(self._K_beta, self._meta_data["K_beta"])):
                self._kernel_param_beta["tau"][lat_k] = self._meta_data["tau_s_beta"][lat_k]

        tau_scales = 100 * np.ones((self._K_gamma))
        sigma_ns = 0.001 * np.ones((self._K_gamma))
        self._kernel_param_gamma = {"tau" : tau_scales, "sigma_n" : sigma_ns}
        if "tau_s_gamma" in self._meta_data:
            for lat_k in range(0, min(self._K_gamma, self._meta_data["K_gamma"])):
                self._kernel_param_gamma["tau"][lat_k] = self._meta_data["tau_s_gamma"][lat_k]



    def _fa_initialisation(self):
        """
        Initializes the model parameters (C and R) using factor analysis.

        """

        if self._intialise_to_gen_param:
            self._C_alpha = self._meta_data["C_gamma"].copy()
            self._C_beta = self._meta_data["C_alpha"].copy()
            self._C_gamma = self._meta_data["C_beta"].copy()
            self._d_bias = self._meta_data["d_bias"].copy()

            if "gen_R" in self._meta_data:
                self._R_noise = self._meta_data["gen_R"].copy()

            return

        # create a big matrix with all the inputs:
        temp_data = self._dataset.transpose([0,2,1]).reshape((self._numb_trials*self._numb_bins,self._numb_neurons))

        # fit fa:
        fa = sklearn.decomposition.FactorAnalysis(n_components=(self._K_beta + self._K_gamma))
        fa.fit(temp_data)

        self._R_noise = np.diag(fa.noise_variance_)

        C_matrix = np.transpose(fa.components_)
        # #self._C_alpha = C_matrix[:,0:self._K_alpha]
        # self._C_beta = C_matrix[:,:self._K_beta]
        # self._C_gamma = C_matrix[:,self._K_beta:]
        # #self._C_beta = self._meta_data["C_beta"]





    def _rescale_C(self):
        """
        This function is used to rescale the C matrix/matrices to have comparable ones with the generating ones.
        
        """
        
        # for lat_k in range(0, self._K_alpha):
        #     self._C_alpha[:,lat_k] *= (np.linalg.norm(self._meta_data["C_alpha"][:,lat_k]) / np.linalg.norm(self._C_alpha[:,lat_k]))
        
        # for lat_k in range(0, self._K_beta):
        #     self._C_beta[:,lat_k] *= (np.linalg.norm(self._meta_data["C_beta"][:,lat_k]) / np.linalg.norm(self._C_beta[:,lat_k]))
        
        # for lat_k in range(0, self._K_gamma):
        #     self._C_gamma[:,lat_k] *= (np.linalg.norm(self._meta_data["C_gamma"][:,lat_k]) / np.linalg.norm(self._C_gamma[:,lat_k]))


        

    def _specific_init(self):
        """
        This method is called as an initialisation of model specific variables.

        """

        ## initialize inducing points locations and 
        #if not self._learn_inducing_locations:
        start = np.min(self._bin_times)
        end = np.max(self._bin_times)
        z_locations = np.linspace(start, end, self._numb_inducing)

        #for trial_i in range(0,self._numb_trials):
        #    for lat_i in range(0,self._numb_latents):
        self._z_induc_loc[:] = z_locations

        self._pre_compute_kernel_matrices()


    def _pre_compute_kernel_matrices(self):
        """
        This method is used to compute and save all the matrices that only 
            depend on the kernel parameters and the inducing points.

        If these are not updated, this method is only called once and then the values are saved.

        """

        self._kernel_mat_induc_alpha, self._inv_kernel_mat_induc_alpha, self._K_tilde_alpha, \
            self._inv_K_tilde_alpha, self._K_i_list_alpha = self._compute_part_of_kernel_mat(self._K_alpha, self._kernel_param_alpha)

        self._kernel_mat_induc_beta, self._inv_kernel_mat_induc_beta, self._K_tilde_beta, \
            self._inv_K_tilde_beta, self._K_i_list_beta = self._compute_part_of_kernel_mat(self._K_beta, self._kernel_param_beta)

        self._kernel_mat_induc_gamma, self._inv_kernel_mat_induc_gamma, self._K_tilde_gamma, \
            self._inv_K_tilde_gamma, self._K_i_list_gamma = self._compute_part_of_kernel_mat(self._K_gamma, self._kernel_param_gamma)

        # construct the list of full K_i s

        self._temp_param_K_i_list = []
        for bin_i in range(0, self._numb_bins):
            K_i = None
            if self._K_alpha > 0:
                if K_i == None:
                    K_i = self._K_i_list_alpha[bin_i]
                else:
                    K_i = block_diag(K_i, self._K_i_list_alpha[bin_i])

            if self._K_beta > 0:
                if K_i is None:
                    K_i = self._K_i_list_beta[bin_i]
                else:
                    K_i = block_diag(K_i, self._K_i_list_beta[bin_i])

            if self._K_gamma > 0:
                if K_i is None:
                    K_i = self._K_i_list_gamma[bin_i]
                else:
                    K_i = block_diag(K_i, self._K_i_list_gamma[bin_i])

            self._temp_param_K_i_list.append(K_i)


    def _compute_part_of_kernel_mat(self, K_numb_latents, kernel_param):
        
        # the kernel matrices and get their inverses:
        kernel_mat_induc = np.zeros((K_numb_latents, self._numb_inducing, self._numb_inducing))
        inv_kernel_mat_induc = np.zeros((K_numb_latents, self._numb_inducing, self._numb_inducing))

        for lat_k in range(0,K_numb_latents):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
            kernel_mat_induc[lat_k,:,:] = K_zz
            inv_kernel_mat_induc[lat_k,:,:] = np.linalg.inv(K_zz)

        # the K tilde as defined in the report
        K_tilde = 0
        for lat_k in range(0,K_numb_latents):
            if lat_k == 0:
                K_tilde = kernel_mat_induc[0,:,:]
            else:
                K_tilde = block_diag(K_tilde, kernel_mat_induc[lat_k,:,:])
        temp_param_K_tilde = K_tilde
        if K_numb_latents == 0:
            temp_param_inv_K_tilde = 0
        else:
            temp_param_inv_K_tilde = np.linalg.inv(K_tilde)

        # the list of the K_i as defined in the report
        K_i_list = []
        for bin_i in range(0,self._numb_bins):
            # create K_i
            K_i = np.zeros((0,0))
            for lat_k in range(0,K_numb_latents):
                if lat_k == 0:
                    K_i = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k).dot(inv_kernel_mat_induc[lat_k,:,:])
                else:
                    K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
                    K_i = block_diag(K_i, K_new.dot(inv_kernel_mat_induc[lat_k,:,:]))
            K_i_list.append(K_i)
        temp_param_K_i_list = K_i_list

        return kernel_mat_induc, inv_kernel_mat_induc, temp_param_K_tilde, temp_param_inv_K_tilde, temp_param_K_i_list




    def _E_step(self):
        """
        The E step of the EM algorithm to fit the model.

        Retrieves the extimate of the latent values.
        
        This is used to compute the updates of:
        - S_k the covariance matrices of the inducing points of the latent GPs
        - m_k the mean vector inducing points of the latent GPs

        """
        # some precomputations:
        if self._force_R_diag:
            Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
            #print(np.diag(self._R_noise))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise))

        CalRinv = np.matmul( np.transpose(self._C_alpha), Rinv )
        CalRinvCal = np.matmul( CalRinv, self._C_alpha )

        CbeRinv = np.matmul( np.transpose(self._C_beta), Rinv )
        CbeRinvCbe = np.matmul( CbeRinv, self._C_beta )

        CgaRinv = np.matmul( np.transpose(self._C_gamma), Rinv )
        CgaRinvCga = np.matmul( CgaRinv, self._C_gamma )

        ## Compute the update of S_k
        # alpha:
        for lat_k in range(0,self._K_alpha):
            sum_of_N = np.zeros((self._numb_inducing, self._numb_inducing))

            for bin_i in range(0,self._numb_bins):
                kappa1 = make_K(self._z_induc_loc, self._bin_times[bin_i], self._kernel_func, self._kernel_param_alpha, lat_k)
                kappa2 = np.transpose(kappa1)

                sum_of_N += np.outer(kappa1, kappa2)

            S_k_main = self._kernel_mat_induc_alpha[lat_k,:,:] + CalRinvCal[lat_k,lat_k] * sum_of_N

            S_k = self._kernel_mat_induc_alpha[lat_k,:,:].dot(np.linalg.inv(S_k_main)).dot(self._kernel_mat_induc_alpha[lat_k,:,:])

            self._S_alpha_covars[lat_k,:,:] = S_k

        # beta:
        for lat_k in range(0,self._K_beta):
            sum_of_N = np.zeros((self._numb_inducing, self._numb_inducing))

            for bin_i in range(0,self._numb_bins):
                kappa1 = make_K(self._z_induc_loc, self._bin_times[bin_i], self._kernel_func, self._kernel_param_beta, lat_k)
                kappa2 = np.transpose(kappa1)

                sum_of_N += np.outer(kappa1, kappa2)

            for cond_i in range(0,self._numb_conditions):

                S_k_main = self._kernel_mat_induc_beta[lat_k,:,:] + CbeRinvCbe[lat_k,lat_k] * sum_of_N * self._numb_trial_per_cond[cond_i]

                S_k = self._kernel_mat_induc_beta[lat_k,:,:].dot(np.linalg.inv(S_k_main)).dot(self._kernel_mat_induc_beta[lat_k,:,:])

                self._S_beta_covars[cond_i,lat_k,:,:] = S_k

        # gamma:
        for lat_k in range(0,self._K_gamma):
            sum_of_N = np.zeros((self._numb_inducing, self._numb_inducing))

            for bin_i in range(0,self._numb_bins):
                kappa1 = make_K(self._z_induc_loc, self._bin_times[bin_i], self._kernel_func, self._kernel_param_gamma, lat_k)
                kappa2 = np.transpose(kappa1)

                sum_of_N += np.outer(kappa1, kappa2)

            S_k_main = self._kernel_mat_induc_gamma[lat_k,:,:] + CgaRinvCga[lat_k,lat_k] * sum_of_N

            S_k = self._kernel_mat_induc_gamma[lat_k,:,:].dot(np.linalg.inv(S_k_main)).dot(self._kernel_mat_induc_gamma[lat_k,:,:])

            for trial_i in range(0,self._numb_trials):
                self._S_gamma_covars[trial_i,lat_k,:,:] = S_k


        ## Compute the update of m_k

        # first compute the inversed term, which is not trial dependent
        first_term_alpha = np.zeros((self._K_alpha*self._numb_inducing, self._K_alpha*self._numb_inducing))
        first_term_Ti_beta = 0
        first_term_Ti_gamma = 0

        for bin_i in range(0,self._numb_bins):

            first_term_alpha += np.transpose(self._K_i_list_alpha[bin_i]).dot(CalRinvCal).dot(self._K_i_list_alpha[bin_i])
            first_term_Ti_beta += np.transpose(self._K_i_list_beta[bin_i]).dot(CbeRinvCbe).dot(self._K_i_list_beta[bin_i])
            first_term_Ti_gamma += np.transpose(self._K_i_list_gamma[bin_i]).dot(CgaRinvCga).dot(self._K_i_list_gamma[bin_i])
        
        l_first_term_Ti_beta = []
        l_second_term_beta = []
        for cond_i in range(0, self._numb_conditions):
            l_first_term_Ti_beta.append( np.linalg.inv( first_term_Ti_beta * self._numb_trial_per_cond[cond_i] + self._inv_K_tilde_beta ))
            l_second_term_beta.append(0)
        #first_term_Ti = np.linalg.inv( first_term_Ti + self._temp_param_inv_K_tilde )

        first_term_alpha = np.linalg.inv(first_term_alpha * self._numb_trials + self._inv_K_tilde_alpha)
        
        first_term_Ti_gamma = np.linalg.inv( first_term_Ti_gamma + self._inv_K_tilde_gamma )

        second_term_alpha = np.zeros((self._K_alpha*self._numb_inducing))
        l_second_term_gamma = []


        for trial_i in range(0,self._numb_trials):
            # compute the second term for each trial:
            #second_term = 0

            this_cond = self._trial_conditions[trial_i]

            l_second_term_gamma.append(0)

            for bin_i in range(0,self._numb_bins):

                temp_term_alpha = self._C_alpha.dot(self._K_i_list_alpha[bin_i]).dot(self._temp_param_m_alpha)
                temp_term_beta = self._C_beta.dot(self._K_i_list_beta[bin_i]).dot(self._temp_param_m_beta[this_cond,:])
                temp_term_gamma = self._C_gamma.dot(self._K_i_list_gamma[bin_i]).dot(self._temp_param_m_gamma[trial_i,:])

                diff = self._dataset[trial_i,:,bin_i] - self._d_bias

                second_term_alpha += np.transpose(self._K_i_list_alpha[bin_i]).dot(CalRinv).dot(diff - temp_term_beta - temp_term_gamma)
                l_second_term_beta[this_cond] += np.transpose(self._K_i_list_beta[bin_i]).dot(CbeRinv).dot( diff - temp_term_gamma - temp_term_alpha)
                l_second_term_gamma[trial_i] += np.transpose(self._K_i_list_gamma[bin_i]).dot(CgaRinv).dot( diff - temp_term_beta - temp_term_alpha)


            m = first_term_Ti_gamma.dot(l_second_term_gamma[trial_i])
            self._temp_param_m_gamma[trial_i,:] = m

            count = 0
            for lat_k in range(0,self._K_gamma):
                self._m_gamma_induc_mean[trial_i,lat_k,:] = m[count:count+self._numb_inducing]
                count += self._numb_inducing

        # print(l_second_term)

        self._temp_param_m_alpha = first_term_alpha.dot(second_term_alpha)
        count = 0
        for lat_k in range(0,self._K_alpha):
            self._m_alpha_induc_mean[lat_k,:] = self._temp_param_m_alpha[count:count+self._numb_inducing]
            count += self._numb_inducing

        for cond_i in range(0, self._numb_conditions):
            m = l_first_term_Ti_beta[cond_i].dot(l_second_term_beta[cond_i])
            self._temp_param_m_beta[cond_i,:] = m

            count = 0
            for lat_k in range(0,self._K_beta):
                self._m_beta_induc_mean[cond_i,lat_k,:] = m[count:count+self._numb_inducing]
                count += self._numb_inducing
        
        # build S matrix:
        self._S_full_matr = None
        for lat_k in range(0, self._K_alpha):
            if self._S_full_matr is None:
                self._S_full_matr = self._S_alpha_covars[lat_k,:,:]
            else:
                self._S_full_matr = block_diag(self._S_full_matr, self._S_alpha_covars[lat_k,:,:])

        for lat_k in range(0, self._K_beta):
            if self._S_full_matr is None:
                self._S_full_matr = self._S_beta_covars[0,lat_k,:,:]
            else:
                self._S_full_matr = block_diag(self._S_full_matr, self._S_beta_covars[0,lat_k,:,:])

        for lat_k in range(0, self._K_gamma):
            if self._S_full_matr is None:
                self._S_full_matr = self._S_gamma_covars[0,lat_k,:,:]
            else:
                self._S_full_matr = block_diag(self._S_full_matr, self._S_gamma_covars[0,lat_k,:,:])


    # def _M_step(self):
    #     """
    #     The M step of the EM algorithm to fit the model.

    #     """

    def _M_step(self):
        """
        The M step of the EM algorithm to fit the model.

        """

        d_sum_term = 0

        C1_sum_term_alpha = 0
        C2_sum_term_alpha = 0

        C1_sum_term_beta = 0
        C2_sum_term_beta = 0

        C1_sum_term_gamma = 0
        C2_sum_term_gamma = 0

        R_sum_term = 0


        for bin_i in range(0,self._numb_bins):

            # get S_ii:
            # for gamma:
            S_ii_alpha = self._build_S_ii(bin_i, self._K_alpha, self._kernel_param_alpha, self._inv_kernel_mat_induc_alpha, self._S_alpha_covars[:,:,:])
                
            Sigma_ii_alpha = self._C_alpha.dot(S_ii_alpha).dot(np.transpose(self._C_alpha))

            # for beta:
            l_S_ii_beta_cond = []
            l_Sigma_ii_beta_cond = []
            for cond_i in self._unique_conditions:
                S_tilde_ii = self._build_S_ii(bin_i, self._K_beta, self._kernel_param_beta, self._inv_kernel_mat_induc_beta, self._S_beta_covars[cond_i,:,:,:])
                l_S_ii_beta_cond.append(S_tilde_ii)
                l_Sigma_ii_beta_cond.append(self._C_beta.dot(l_S_ii_beta_cond[cond_i]).dot(np.transpose(self._C_beta)))

            # for gamma:
            S_ii_gamma = self._build_S_ii(bin_i, self._K_gamma, self._kernel_param_gamma, self._inv_kernel_mat_induc_gamma, self._S_gamma_covars[0,:,:,:])
                
            Sigma_ii_gamma = self._C_gamma.dot(S_ii_gamma).dot(np.transpose(self._C_gamma))

            
            m_tilde_i_r_alpha = self._K_i_list_alpha[bin_i].dot(self._temp_param_m_alpha[:])
            mu_i_r_alpha = self._C_alpha.dot(m_tilde_i_r_alpha)

            # now the trial sum:
            for trial_i in range(0, self._numb_trials):

                this_cond = self._trial_conditions[trial_i]

                # get m_tilde i r
                m_tilde_i_r_beta = self._K_i_list_beta[bin_i].dot(self._temp_param_m_beta[this_cond,:])
                m_tilde_i_r_gamma = self._K_i_list_gamma[bin_i].dot(self._temp_param_m_gamma[trial_i,:])

                # get mu_i_r
                mu_i_r_beta = self._C_beta.dot(m_tilde_i_r_beta)
                mu_i_r_gamma = self._C_gamma.dot(m_tilde_i_r_gamma)

                diff = (self._dataset[trial_i,:,bin_i] - self._d_bias)

                # update sum terms:
                # C:
                C1_sum_term_alpha += np.outer( diff - mu_i_r_gamma - mu_i_r_beta, m_tilde_i_r_alpha )
                C2_sum_term_alpha += S_ii_alpha + np.outer(m_tilde_i_r_alpha, m_tilde_i_r_alpha)

                # C:
                C1_sum_term_beta += np.outer( diff - mu_i_r_alpha - mu_i_r_gamma, m_tilde_i_r_beta )
                C2_sum_term_beta += l_S_ii_beta_cond[this_cond] + np.outer(m_tilde_i_r_beta, m_tilde_i_r_beta)

                # C:
                C1_sum_term_gamma += np.outer( diff - mu_i_r_alpha - mu_i_r_beta, m_tilde_i_r_gamma )
                C2_sum_term_gamma += S_ii_gamma + np.outer(m_tilde_i_r_gamma, m_tilde_i_r_gamma)

                # d:
                d_sum_term += self._dataset[trial_i,:,bin_i] - mu_i_r_beta - mu_i_r_gamma - mu_i_r_alpha

                # R:
                temp = diff - mu_i_r_beta - mu_i_r_gamma - mu_i_r_alpha
                R_sum_term += np.outer(temp, temp) + l_Sigma_ii_beta_cond[this_cond] + Sigma_ii_gamma + Sigma_ii_alpha


        self._C_alpha = C1_sum_term_alpha.dot(np.linalg.inv(C2_sum_term_alpha))

        self._C_beta = C1_sum_term_beta.dot(np.linalg.inv(C2_sum_term_beta))

        self._C_gamma = C1_sum_term_gamma.dot(np.linalg.inv(C2_sum_term_gamma))

        self._C_matrix = np.concatenate([self._C_alpha, self._C_beta, self._C_gamma], axis=1)
        self._C_matrix_big = np.concatenate([self._C_alpha, self._C_beta, self._C_gamma], axis=1)

        self._d_bias = (1 / ( self._numb_trials * self._numb_bins )) * d_sum_term

        self._R_noise = (1 / ( self._numb_trials * self._numb_bins )) * R_sum_term
        if self._force_R_diag:
            self._R_noise = np.diag(np.diag(self._R_noise))



        #####################################################
        # if self._learn_kernel_params:
        #     self._temp_param_m = self._temp_param_m.astype(float)
        #     self._update_kernel_tau()

        # if self._learn_GP_noise or self._learn_kernel_params:
        #     # we update all the variables that only depend on kernel parameters and z_loc:
        #     self._pre_compute_kernel_matrices()



    def _build_S_ii(self, bin_i, K_numb_latents, kernel_param, inv_kernel_mat_induc, S_covars):
        """
        ll
        """

        S_tilde_ii = np.zeros((0,0))
        for lat_k in range(0,K_numb_latents):
            k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, kernel_param, lat_k)
            k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
            k_Zi = np.transpose(k_iZ)
            matr_term = inv_kernel_mat_induc[lat_k,:,:].dot(S_covars[lat_k,:,:]).dot(inv_kernel_mat_induc[lat_k,:,:])
            matr_term -= inv_kernel_mat_induc[lat_k,:,:]

            temp = k_ii + k_iZ.dot(matr_term).dot(k_Zi)

            if lat_k == 0:
                S_tilde_ii = k_ii + k_iZ.dot(matr_term).dot(k_Zi)
            else:
                S_tilde_ii = block_diag(S_tilde_ii , temp)

        return S_tilde_ii



    def _recover_path_for_model(self):
        """
        Used to obtain the latent path after the fitting of the model.

        This is a method that is called after the EM iterations.
        
        """
        
        latents_alpha = np.zeros((self._K_alpha, self._numb_bins))
        count = 0
        for lat_k in range(0,self._K_alpha):
            self._m_alpha_induc_mean[lat_k,:] = self._temp_param_m_alpha[count:count+self._numb_inducing]
            count += self._numb_inducing

            kappa_mat = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param_alpha, lat_k)
            kappa_invK = kappa_mat.dot(self._inv_kernel_mat_induc_alpha[lat_k,:,:])

            latents_alpha[lat_k,:] = kappa_invK.dot(self._m_alpha_induc_mean[lat_k,:])

        latents_beta = np.zeros((self._numb_conditions,self._K_beta,self._numb_bins))
        for cond_i in range(0,self._numb_conditions):
            count = 0
            for lat_k in range(0,self._K_beta):
                self._m_beta_induc_mean[cond_i,lat_k,:] = self._temp_param_m_beta[cond_i,count:count+self._numb_inducing]
                count += self._numb_inducing

                kappa_mat = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param_beta, lat_k)
                kappa_invK = kappa_mat.dot(self._inv_kernel_mat_induc_beta[lat_k,:,:])

                latents_beta[cond_i,lat_k,:] = kappa_invK.dot(self._m_beta_induc_mean[cond_i,lat_k,:])


        latents_gamma = np.zeros((self._numb_trials,self._K_gamma,self._numb_bins))
        for trial_i in range(0,self._numb_trials):
            count = 0
            for lat_k in range(0,self._K_gamma):
                self._m_gamma_induc_mean[trial_i,lat_k,:] = self._temp_param_m_gamma[trial_i,count:count+self._numb_inducing]
                count += self._numb_inducing

                kappa_mat = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param_gamma, lat_k)
                kappa_invK = kappa_mat.dot(self._inv_kernel_mat_induc_gamma[lat_k,:,:])

                latents_gamma[trial_i,lat_k,:] = kappa_invK.dot(self._m_gamma_induc_mean[trial_i,lat_k,:])


        return latents_alpha, latents_beta, latents_gamma














    def _compute_free_energy_para_taus(self, param):
        """
        Compute and return the free energy from the kernel parameters.

        This function is to be used with auto grad to update the kernel parameters.
        
        """

        # some precomputations:
        if self._force_R_diag:
            Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise))

        CalRinv = np.matmul( np.transpose(self._C_alpha), Rinv )
        CalRinvCal = np.matmul( CalRinv, self._C_alpha )

        CbeRinv = np.matmul( np.transpose(self._C_beta), Rinv )
        CbeRinvCbe = np.matmul( CbeRinv, self._C_beta )

        CgaRinv = np.matmul( np.transpose(self._C_gamma), Rinv )
        CgaRinvCga = np.matmul( CgaRinv, self._C_gamma )


        first_term = 0
        second_term_alpha = 0
        second_term_beta = 0
        second_term_gamma = 0


        for lat_k in range(0,self._K_alpha):

            expr = np.trace( np.matmul(self._inv_kernel_mat_induc_alpha[lat_k,:,:] , self._S_alpha_covars[lat_k,:,:]) )

            expr += np.matmul( np.matmul( np.transpose( self._m_alpha_induc_mean[lat_k,:] ), self._inv_kernel_mat_induc_alpha[lat_k,:,:] ), self._m_alpha_induc_mean[lat_k,:])

            expr += np.log( np.linalg.det(self._kernel_mat_induc_alpha[lat_k,:,:]) ) - self._numb_inducing - np.log( np.linalg.det(self._S_alpha_covars[lat_k,:,:]) )

            second_term_alpha += 0.5 * expr * self._numb_trials

        for lat_k in range(0,self._K_beta):
            for cond_i in range(0, self._numb_conditions):
                expr = np.trace( np.matmul(self._inv_kernel_mat_induc_beta[lat_k,:,:] , self._S_beta_covars[cond_i,lat_k,:,:]) )

                expr += np.matmul( np.matmul( np.transpose( self._m_beta_induc_mean[cond_i,lat_k,:] ), self._inv_kernel_mat_induc_beta[lat_k,:,:] ), self._m_beta_induc_mean[cond_i,lat_k,:])

                expr += np.log( np.linalg.det(self._kernel_mat_induc_beta[lat_k,:,:]) ) - self._numb_inducing - np.log( np.linalg.det(self._S_beta_covars[cond_i,lat_k,:,:]) )

                second_term_beta += 0.5 * expr * self._numb_trial_per_cond[cond_i]

        for lat_k in range(0,self._K_gamma):
            for trial_i in range(0, self._numb_trials):
                expr = np.trace( np.matmul(self._inv_kernel_mat_induc_gamma[lat_k,:,:] , self._S_gamma_covars[trial_i,lat_k,:,:]) )

                expr += np.matmul( np.matmul( np.transpose( self._m_gamma_induc_mean[trial_i,lat_k,:] ), self._inv_kernel_mat_induc_gamma[lat_k,:,:] ), self._m_gamma_induc_mean[trial_i,lat_k,:])

                expr += np.log( np.linalg.det(self._kernel_mat_induc_gamma[lat_k,:,:]) ) - self._numb_inducing - np.log( np.linalg.det(self._S_gamma_covars[trial_i,lat_k,:,:]) )

                second_term_gamma += 0.5 * expr

        # the first term : 

        first_term += (logdet_R + self._numb_neurons * np.log(2 * np.pi))
        first_term += 0
        first_term *= - 0.5 * self._numb_bins * self._numb_trials * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            # last terms
            sum_over_k_first = 0

            S_ii_alpha = self._build_S_ii(bin_i, self._K_alpha, self._kernel_param_alpha, self._inv_kernel_mat_induc_alpha, self._S_alpha_covars[:,:,:])
            sum_over_k_first += np.trace( np.matmul(CalRinvCal, S_ii_alpha) )

            S_tilde_ii = self._build_S_ii(bin_i, self._K_beta, self._kernel_param_beta, self._inv_kernel_mat_induc_beta, self._S_beta_covars[0,:,:,:])
            sum_over_k_first += np.trace( np.matmul(CbeRinvCbe, S_tilde_ii) )

            S_ii_gamma = self._build_S_ii(bin_i, self._K_gamma, self._kernel_param_gamma, self._inv_kernel_mat_induc_gamma, self._S_gamma_covars[0,:,:,:])
            sum_over_k_first += np.trace( np.matmul(CgaRinvCga, S_ii_gamma) )


            m_tilde_i_r_alpha = self._K_i_list_alpha[bin_i].dot(self._temp_param_m_alpha[:])
            mu_i_r_alpha = self._C_alpha.dot(m_tilde_i_r_alpha)

            for trial_i in range(0,self._numb_trials):

                this_cond = self._trial_conditions[trial_i]

                # forming m tilde:
                m_tilde_i_r_beta = self._K_i_list_beta[bin_i].dot(self._temp_param_m_beta[this_cond,:])
                m_tilde_i_r_gamma = self._K_i_list_gamma[bin_i].dot(self._temp_param_m_gamma[trial_i,:])

                # get mu_i_r
                mu_i_r_beta = self._C_beta.dot(m_tilde_i_r_beta)
                mu_i_r_gamma = self._C_gamma.dot(m_tilde_i_r_gamma)

                # putting it together:
                m_tilde_C = mu_i_r_alpha + mu_i_r_beta + mu_i_r_gamma

                # computing all the terms:
                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[trial_i,:,bin_i]) , Rinv), self._dataset[trial_i,:,bin_i])

                # combining the two m K_i C terms:
                first_term += np.matmul( np.matmul(np.transpose(m_tilde_C) , Rinv) , (self._dataset[trial_i,:,bin_i] - self._d_bias   )  ) 

                # combining the two d terms:
                first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[trial_i,:,bin_i] - 0.5 * self._d_bias ) )

                # the big term:
                first_term += - 0.5 * np.matmul(  np.matmul( np.transpose(m_tilde_C) , Rinv) , m_tilde_C )

                first_term += - 0.5 * sum_over_k_first

        
        F = first_term - (second_term_alpha + second_term_beta + second_term_gamma)

        # self._free_energy_1_list.append(first_term[0,0])
        # self._free_energy_2_list.append(big_term)
        return F


        




    def _get_tau_param(self):
        """
        Returns the kernel time scale parameters in order to input them in the free energy function.
        """

        tau = np.zeros(self._K_alpha+self._K_beta+self._K_gamma)

        tau[0:self._K_alpha] = self._kernel_param_alpha["tau"]
        tau[self._K_alpha:self._K_alpha+self._K_beta] = self._kernel_param_beta["tau"]
        tau[self._K_alpha+self._K_beta:] = self._kernel_param_gamma["tau"]

        return tau


    def _compute_free_energy_para_taus_for_derive(self, param):
        """
        Compute and return the free energy from the kernel parameters.

        This function is to be used with auto grad to update the kernel parameters.
        
        """

        kernel_param_alpha = {"tau" : param[0:self._K_alpha], "sigma_n" : self._kernel_param_alpha["sigma_n"]}
        kernel_param_beta = {"tau" : param[self._K_alpha:self._K_alpha+self._K_beta], "sigma_n" : self._kernel_param_beta["sigma_n"]}
        kernel_param_gamma = {"tau" : param[self._K_alpha+self._K_beta:], "sigma_n" : self._kernel_param["sigma_n"]}


        # some precomputations:
        if self._force_R_diag:
            Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise))

        CalRinv = np.matmul( np.transpose(self._C_alpha), Rinv )
        CalRinvCal = np.matmul( CalRinv, self._C_alpha )

        CbeRinv = np.matmul( np.transpose(self._C_beta), Rinv )
        CbeRinvCbe = np.matmul( CbeRinv, self._C_beta )

        CgaRinv = np.matmul( np.transpose(self._C_gamma), Rinv )
        CgaRinvCga = np.matmul( CgaRinv, self._C_gamma )



        first_term = 0
        second_term_alpha = 0
        second_term_beta = 0
        second_term_gamma = 0


        inv_Kzz_list_alpha = []
        for lat_k in range(0,self._K_alpha):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param_alpha, lat_k)
            inv_Kzz = np.linalg.inv(K_zz)
            inv_Kzz_list_alpha.append(inv_Kzz)

            expr = np.trace( np.matmul(inv_Kzz , self._S_alpha_covars[lat_k,:,:]) )

            expr += np.matmul( np.matmul( np.transpose( self._m_alpha_induc_mean[lat_k,:] ), inv_Kzz ), self._m_alpha_induc_mean[lat_k,:])

            expr += np.log( np.linalg.det(K_zz) ) - self._numb_inducing - np.log( np.linalg.det(self._S_alpha_covars[lat_k,:,:]) )

            second_term_alpha += 0.5 * expr * self._numb_trials

        inv_Kzz_list_beta = []
        for lat_k in range(0,self._K_beta):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param_beta, lat_k)
            inv_Kzz = np.linalg.inv(K_zz)
            inv_Kzz_list_beta.append(inv_Kzz)

            for cond_i in range(0, self._numb_conditions):
                expr = np.trace( np.matmul(inv_Kzz , self._S_beta_covars[cond_i,lat_k,:,:]) )

                expr += np.matmul( np.matmul( np.transpose( self._m_beta_induc_mean[cond_i,lat_k,:] ), inv_Kzz ), self._m_beta_induc_mean[cond_i,lat_k,:])

                expr += np.log( np.linalg.det(K_zz) ) - self._numb_inducing - np.log( np.linalg.det(self._S_beta_covars[cond_i,lat_k,:,:]) )

                second_term_beta += 0.5 * expr * self._numb_trial_per_cond[cond_i]

        inv_Kzz_list_gamma = []
        for lat_k in range(0,self._K_gamma):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param_gamma, lat_k)
            inv_Kzz = np.linalg.inv(K_zz)
            inv_Kzz_list_gamma.append(inv_Kzz)

            for trial_i in range(0, self._numb_trials):
                expr = np.trace( np.matmul(inv_Kzz , self._S_gamma_covars[trial_i,lat_k,:,:]) )

                expr += np.matmul( np.matmul( np.transpose( self._m_gamma_induc_mean[trial_i,lat_k,:] ), inv_Kzz ), self._m_gamma_induc_mean[trial_i,lat_k,:])

                expr += np.log( np.linalg.det(K_zz) ) - self._numb_inducing - np.log( np.linalg.det(self._S_gamma_covars[trial_i,lat_k,:,:]) )

                second_term_gamma += 0.5 * expr

        # the first term : 

        first_term += - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            # last terms
            sum_over_k_first = 0
            for lat_k in range(0,self._K_alpha):
                inv_Kzz = inv_Kzz_list_alpha[lat_k]

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, kernel_param_alpha, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param_alpha, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz ,self._S_alpha_covars[lat_k,:,:]), inv_Kzz )
                matr_term -= inv_Kzz

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CalRinvCal[lat_k,lat_k] * temp

            for lat_k in range(0,self._K_beta):
                inv_Kzz = inv_Kzz_list_beta[lat_k]

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, kernel_param_beta, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param_beta, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz ,self._S_beta_covars[0,lat_k,:,:]), inv_Kzz )
                matr_term -= inv_Kzz

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CbeRinvCbe[lat_k,lat_k] * temp

            for lat_k in range(0,self._K_gamma):
                inv_Kzz = inv_Kzz_list_gamma[lat_k]

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, kernel_param_gamma, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param_gamma, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz ,self._S_gamma_covars[0,lat_k,:,:]), inv_Kzz )
                matr_term -= inv_Kzz

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CgaRinvCga[lat_k,lat_k] * temp


            m_tilde_i_r_alpha = self._K_i_list_alpha[bin_i].dot(self._temp_param_m_alpha[:])

            for trial_i in range(0,self._numb_trials):

                this_cond = self._trial_conditions[trial_i]

                # forming m tilde:
                m_tilde_beta = np.zeros((1, self._K_beta))
                e_unit_vect = np.eye(self._K_beta)
                for lat_k in range(0,self._K_beta):
                    inv_Kzz = inv_Kzz_list_beta[lat_k]
                    K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param_beta, lat_k)
                    m_tilde_beta += e_unit_vect[:,lat_k] * np.dot(np.matmul(K_new,inv_Kzz),self._m_beta_induc_mean[this_cond,lat_k,:])

                m_tilde_gamma = np.zeros((1, self._K_gamma))
                e_unit_vect = np.eye(self._K_gamma)
                for lat_k in range(0,self._K_gamma):
                    inv_Kzz = inv_Kzz_list_gamma[lat_k]
                    K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param_gamma, lat_k)
                    m_tilde_gamma += e_unit_vect[:,lat_k] * np.dot(np.matmul(K_new,inv_Kzz),self._m_gamma_induc_mean[trial_i,lat_k,:])

                # putting it together:
                m_tilde_C = np.matmul(self._C_alpha, np.transpose(m_tilde_alpha) )
                m_tilde_C += np.matmul(self._C_beta, np.transpose(m_tilde_beta) ) + np.matmul(self._C_gamma, np.transpose(m_tilde_gamma) )

                # computing all the terms:
                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[trial_i,:,bin_i]) , Rinv), self._dataset[trial_i,:,bin_i])

                # combining the two m K_i C terms:
                first_term += np.matmul( np.matmul(np.transpose(m_tilde_C) , Rinv) , (self._dataset[trial_i,:,bin_i] - self._d_bias   )  ) 

                # combining the two d terms:
                first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[trial_i,:,bin_i] - 0.5 * self._d_bias ) )

                # the big term:
                first_term += - 0.5 * np.matmul(  np.matmul( np.transpose(m_tilde_C) , Rinv) , m_tilde_C )[0,0]

                first_term += - 0.5 * sum_over_k_first[0,0]

        
        F = first_term - (second_term_alpha + second_term_beta + second_term_gamma)

        # self._free_energy_1_list.append(first_term[0,0])
        # self._free_energy_2_list.append(big_term)
        return F[0]

