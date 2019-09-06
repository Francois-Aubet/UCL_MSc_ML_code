"""

The sparce variational approximation of the Gaussian Process Factor Analysis (GPFA) model.



"""

from algorithms.Algorithm import Algorithm
from algorithms.IGPFA import IGPFA
import numpy as np
import time
from tqdm import tqdm 
from utilis import *
from scipy.linalg import block_diag
from scipy.optimize import minimize
import matplotlib.pyplot as plt



# the auto grad things:
import autograd.numpy as np
#import autograd.numpy.random as npr
from autograd import value_and_grad, grad

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

SMALL = 1e-18 #1e-12


class GPFA_sv(IGPFA):
    """ sparce variational GPFA 
    
    """

    algName = "sv_GPFA"


    def __init__(self, dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step):
        """ Preparation of all the needed variables. """

        super().__init__(dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step)


        self._convergence_criteria = 10**(-15)

        self._temp_param_m = np.zeros((self._numb_trials,self._K_numb_latents*self._numb_inducing))

        self._use_lr_E_steps = False

        self._use_gradient_E_steps = False

        self._number_gradient_iterations_l = []



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

        # print(self._z_induc_loc)

        # self._z_induc_loc[:] = self._bin_times[0:self._numb_bins-1]

        if self._numb_inducing > self._numb_bins:
            self._z_induc_loc[0:self._numb_bins] = self._bin_times
            for i in range(0, self._numb_inducing - self._numb_bins):
                ind = self._numb_bins + i
                self._z_induc_loc[ind] = self._bin_times[-1] + i +2

        self._pre_compute_kernel_matrices()

        self._perturbate_S(120)

        #self._C_matrix = self._meta_data["C_gamma"]


    def _set_Sk_from_S_full(self):
        """
        """
        M = self._numb_inducing
        for lat_k in range(0, self._K_numb_latents):
            for trial_i in range(0, self._numb_trials):
                self._S_covars[trial_i,lat_k,:,:] = self._S_full_matr[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]

    def _set_S_full_from_Sk(self):
        """
        """
        # self._S_full_matr = self._S_covars[0,0,:,:]
        # for lat_k in range(1, self._K_numb_latents):
        #     self._S_full_matr = block_diag(self._S_full_matr, self._S_covars[0,lat_k,:,:])

        M = self._numb_inducing
        for lat_k in range(0, self._K_numb_latents):
            self._S_full_matr[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M] = self._S_covars[0,lat_k,:,:]



    def _pre_compute_kernel_matrices(self):
        """
        This method is used to compute and save all the matrices that only 
            depend on the kernel parameters and the inducing points.

        If these are not updated, this method is only called once and then the values are saved.

        """
        
        # the kernel matrices and get their inverses:
        self._kernel_mat_induc = np.zeros((self._K_numb_latents, self._numb_inducing, self._numb_inducing))
        self._inv_kernel_mat_induc = np.zeros((self._K_numb_latents, self._numb_inducing, self._numb_inducing))

        for lat_k in range(0,self._K_numb_latents):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            self._kernel_mat_induc[lat_k,:,:] = K_zz
            self._inv_kernel_mat_induc[lat_k,:,:] = np.linalg.inv(K_zz)

        # the K tilde as defined in the report
        K_tilde = self._kernel_mat_induc[0,:,:]
        for lat_k in range(1,self._K_numb_latents):
            K_tilde = block_diag(K_tilde, self._kernel_mat_induc[lat_k,:,:])
        self._temp_param_K_tilde = K_tilde
        self._temp_param_inv_K_tilde = np.linalg.inv(K_tilde)

        # the list of the K_i as defined in the report
        K_i_list = []
        for bin_i in range(0,self._numb_bins):
            # create K_i
            K_i = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, 0).dot(self._inv_kernel_mat_induc[0,:,:])
            for lat_k in range(1,self._K_numb_latents):
                K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                K_i = block_diag(K_i, K_new.dot(self._inv_kernel_mat_induc[lat_k,:,:]))
            K_i_list.append(K_i)
        self._temp_param_K_i_list = K_i_list


    def _E_step(self):
        """
        """

        if self._use_lr_E_steps:
            self._E_step_lr()

        elif self.use_MF_approximation:

            if self._use_gradient_E_steps:
                self._E_step_grad_mf()
            else:
                self._E_step_mf()

        else:
            if self._use_gradient_E_steps:
                self._E_step_grad_full()
            else:
                self._E_step_full()



    def _M_step(self):
        """
        """
        self._M_step_full()

        # if self.use_MF_approximation:
        #     self._M_step_mf()
        # else:
        #     self._M_step_full()

        if self._learn_kernel_params:
            self._temp_param_m = self._temp_param_m.astype(float)
            self._update_kernel_tau()

        if self._learn_GP_noise or self._learn_kernel_params:
            # we update all the variables that only depend on kernel parameters and z_loc:
            self._pre_compute_kernel_matrices()





    def _E_step_mf(self, update_m = True):
        """
        The E step of the EM algorithm to fit the model.
        Retrieves the extimate of the latent values.
        
        This is used to compute the updates of:
        - S_k the covariance matrices of the inducing points of the latent GPs
        - m_k the mean vector inducing points of the latent GPs
        """
        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()


        ## Compute the update of S_k

        for lat_k in range(0,self._K_numb_latents):
            sum_of_N = np.zeros((self._numb_inducing, self._numb_inducing))

            for bin_i in range(0,self._numb_bins):
                kappa1 = make_K(self._z_induc_loc, self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                kappa2 = np.transpose(kappa1)

                sum_of_N += np.outer(kappa1, kappa2)

            S_k_main = self._kernel_mat_induc[lat_k,:,:] + CRinvC[lat_k,lat_k] * sum_of_N

            S_k = self._kernel_mat_induc[lat_k,:,:].dot(np.linalg.inv(S_k_main)).dot(self._kernel_mat_induc[lat_k,:,:])

            for trial_i in range(0,self._numb_trials):
                self._S_covars[trial_i,lat_k,:,:] = S_k

        ## Compute the update of m_k

        # first compute the inversed term, which is not trial dependent
        first_term_Ti = 0

        for bin_i in range(0,self._numb_bins):

            first_term_Ti += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinvC).dot(self._temp_param_K_i_list[bin_i])

        first_term_Ti = np.linalg.inv( first_term_Ti + np.linalg.inv(self._temp_param_K_tilde) )

        self._temp_param_m = np.zeros((self._numb_trials,self._K_numb_latents*self._numb_inducing))

        for trial_i in range(0,self._numb_trials):
            # compute the second term for each trial:
            second_term = 0

            for bin_i in range(0,self._numb_bins):
                second_term += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinv).dot( self._dataset[trial_i,:,bin_i] - self._d_bias )

            m = first_term_Ti.dot(second_term)
            self._temp_param_m[trial_i,:] = m

            count = 0
            for lat_k in range(0,self._K_numb_latents):
                self._m_induc_mean[trial_i,lat_k,:] = m[count:count+self._numb_inducing]
                count += self._numb_inducing
        
        # update other things:
        self._set_S_full_from_Sk()
        self._C_matrix_big = self._C_matrix


    def _M_step_mf(self):
        """
        The M step of the EM algorithm to fit the model.


        """

        d_sum_term = 0
        C1_sum_term = 0
        C2_sum_term = 0
        R_sum_term = 0

        for bin_i in range(0,self._numb_bins):

            # get S_ii:
            # k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, 0)
            # k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, 0)
            # k_Zi = np.transpose(k_iZ)
            # matr_term = self._inv_kernel_mat_induc[0,:,:].dot(self._S_covars[0,0,:,:]).dot(self._inv_kernel_mat_induc[0,:,:])
            # matr_term -= self._inv_kernel_mat_induc[0,:,:]

            # S_tilde_ii = k_ii + k_iZ.dot(matr_term).dot(k_Zi)

            for lat_k in range(0,self._K_numb_latents):
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = self._inv_kernel_mat_induc[lat_k,:,:].dot(self._S_covars[0,lat_k,:,:]).dot(self._inv_kernel_mat_induc[lat_k,:,:])
                matr_term -= self._inv_kernel_mat_induc[lat_k,:,:]

                temp = k_ii + k_iZ.dot(matr_term).dot(k_Zi)

                if lat_k == 0:
                    S_tilde_ii = temp.copy()
                else:
                    S_tilde_ii = block_diag(S_tilde_ii , temp)


            # get Sigma_ii:
            Sigma_ii = self._C_matrix.dot(S_tilde_ii).dot(np.transpose(self._C_matrix))

            # now the trial sum:
            for trial_i in range(0, self._numb_trials):

                # get m_tilde i r
                m_tilde_i_r = self._temp_param_K_i_list[bin_i].dot(self._temp_param_m[trial_i,:])

                # get mu_i_r
                mu_i_r = self._C_matrix.dot(m_tilde_i_r) + self._d_bias

                # update sum terms:
                # C:
                C1_sum_term += np.outer( (self._dataset[trial_i,:,bin_i] - self._d_bias) , m_tilde_i_r )
                C2_sum_term += S_tilde_ii + np.outer(m_tilde_i_r, m_tilde_i_r)

                # d:
                d_sum_term += self._dataset[trial_i,:,bin_i] - self._C_matrix.dot(m_tilde_i_r)

                # R:
                temp = self._dataset[trial_i,:,bin_i] - mu_i_r
                R_sum_term += np.outer(temp, temp) + Sigma_ii

        self._C_matrix = C1_sum_term.dot(np.linalg.inv(C2_sum_term))
        self._C_matrix_big = self._C_matrix

        # print("orth numerator C: ",get_orthogonality_score(C1_sum_term, False)[0] )
        # print( get_orthogonality_score(C2_sum_term, False)[0], get_orthogonality_score(np.linalg.inv(C2_sum_term), False)[0] , get_orthogonality_score(self._C_matrix, False)[0] )
        # print(C2_sum_term, np.linalg.inv(C2_sum_term) * 10000)

        self._d_bias = (1 / ( self._numb_trials * self._numb_bins )) * d_sum_term

        self._R_noise = (1 / ( self._numb_trials * self._numb_bins )) * R_sum_term
        if self._force_R_diag:
            self._R_noise = np.diag(np.diag(self._R_noise))


        #####################################################
        if self._learn_kernel_params:
            self._temp_param_m = self._temp_param_m.astype(float)
            self._update_kernel_tau()

        if self._learn_GP_noise or self._learn_kernel_params:
            # we update all the variables that only depend on kernel parameters and z_loc:
            self._pre_compute_kernel_matrices()




    def _recover_path_for_model(self):
        """
        Used to obtain the latent path after the fitting of the model.

        This is a method that is called after the EM iterations.
        
        """
        latents = np.zeros((self._meta_data["numb_trials"],self._K_numb_latents,len(self._meta_data["bin_times"])))

        ## compute the latent GP trajectories from the variational means
        for lat_k in range(0,self._K_numb_latents):
            kappa_mat = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            kappa_invK = kappa_mat.dot(self._inv_kernel_mat_induc[lat_k,:,:])

            for trial_i in range(0,self._numb_trials):
                x_k_r = kappa_invK.dot(self._m_induc_mean[trial_i,lat_k,:])
                latents[trial_i,lat_k,:] = x_k_r

        return latents



    ############ alternative E-step s #################

    def _E_step_full(self):
        """
        E step with no mean field assumption: getting the full covariance matrix.

        """
        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        ## Compute the update of S
        sum_T = 0

        for bin_i in range(0,self._numb_bins):
            sum_T += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinvC).dot(self._temp_param_K_i_list[bin_i])

        # print( (( self._temp_param_K_i_list[bin_i]) != 0).astype(int))
        # print( (( CRinvC) != 0).astype(int))
        # print( (( sum_T) != 0).astype(int))

        S = np.linalg.inv( sum_T + self._temp_param_inv_K_tilde ) 

        # print( (( S ) != 0).astype(int))
        self._S_full_matr = S.copy()
            
        # free_current = self._compute_free_energy_full()
        # print("done updating S...", free_current)
        ## Compute the update of m_k

        self._E_update_m()
        
        # update other things:
        # self._set_S_full_from_Sk()
        self._set_Sk_from_S_full()
        self._C_matrix_big = self._C_matrix



    def _E_update_m(self):
        """
        The E-step for m.

        Put on the side because it is the same for both mf and full.
        """
        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        ## Compute the update of m_k

        # first compute the inversed term, which is not trial dependent
        first_term_Ti = 0

        for bin_i in range(0,self._numb_bins):

            first_term_Ti += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinvC).dot(self._temp_param_K_i_list[bin_i])

        first_term_Ti = np.linalg.inv( first_term_Ti + np.linalg.inv(self._temp_param_K_tilde) )

        self._temp_param_m = np.zeros((self._numb_trials,self._K_numb_latents*self._numb_inducing))

        for trial_i in range(0,self._numb_trials):
            # compute the second term for each trial:
            second_term = 0

            for bin_i in range(0,self._numb_bins):
                second_term += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinv).dot( self._dataset[trial_i,:,bin_i] - self._d_bias )

            m = first_term_Ti.dot(second_term)
            self._temp_param_m[trial_i,:] = m

            count = 0
            for lat_k in range(0,self._K_numb_latents):
                self._m_induc_mean[trial_i,lat_k,:] = m[count:count+self._numb_inducing]
                count += self._numb_inducing


    def _perturbate_S(self, denominator = 500):
        """
        A random preturbation of S.
        """

        # print(is_pos_def(self._S_full_matr))

        if self.use_MF_approximation:

            for lat_k in range(0, self._K_numb_latents):
                rand = np.random.randn(self._S_covars[0,lat_k,:,:].shape[0], self._S_covars[0,lat_k,:,:].shape[1]) / denominator

                L = rand#np.tril(rand)

                for trial_i in range(0, self._numb_trials):
                    self._S_covars[trial_i,lat_k,:,:] += L.transpose().dot(L)

            self._set_S_full_from_Sk()

        else:
            rand = np.random.randn(self._S_full_matr.shape[0], self._S_full_matr.shape[1]) / denominator

            L = rand#np.tril(rand)

            self._S_full_matr += L.transpose().dot(L)
            self._set_Sk_from_S_full()

        # print(is_pos_def(self._S_full_matr))



    def __minus_grad_L_k(self,theta,lat_k):
        """
        The gradient of the free energy with respect to m.
        """
        M = self._numb_inducing 
        L_k = np.zeros((M, M))

        for i in range(0,M):
            for j in range(i,M):
                place = i * M + j - ((i * (i + 1)) // 2)
                L_k[i,j] = theta[place]

        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        sum_T = 0
        for bin_i in range(0,self._numb_bins):
            kappa1 = make_K(self._z_induc_loc, self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
            kappa2 = np.transpose(kappa1)

            matr_kappa = np.outer(kappa1, kappa2)

            kappa_i = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            sum_T += CRinvC[lat_k,lat_k] * self._inv_kernel_mat_induc[lat_k,:,:].dot(matr_kappa).dot(self._inv_kernel_mat_induc[lat_k,:,:]).dot(L_k)

        # with gradient: (need to constraint the matrix)
        L_k_grad =  -  sum_T -  self._inv_kernel_mat_induc[lat_k,:,:].dot(L_k) + np.linalg.inv(L_k).transpose()


        numb_param = ((M * (M + 1)) // 2)

        theta = np.zeros(numb_param)

        for i in range(0,M):
            for j in range(i,M):
                place = i * M + j - ((i * (i + 1)) // 2)
                theta[place] = - L_k_grad[i,j]

        return theta



    def _E_step_grad_mf(self):
        """
        E step with mean field assumption using gradient method.

        """
        # self._set_Sk_from_S_full()
        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        # print(self._compute_likelihood_term_full() - self._compute_KL_term_full())
                
        ## update of S using grad method:
        for lat_k in range(0, self._K_numb_latents):
            M = self._numb_inducing # just making things clearer to read here
            numb_param = ((M * (M + 1)) // 2)

            # try:
            L = np.linalg.cholesky(self._S_covars[0,lat_k,:,:])
            # except:
            #     L = np.linalg.cholesky(np.eye(M))
            
            L = L.transpose()

            theta = np.zeros(numb_param)

            for i in range(0,M):
                for j in range(i,M):
                    place = i * M + j - ((i * (i + 1)) // 2)
                    theta[place] = L[i,j]

            # optimise:
            results = minimize(self._free_energy_opt_embelisher_for_S_mf, theta, args=(lat_k), method='L-BFGS-B', \
                    jac=self.__minus_grad_L_k, options={'ftol': self._convergence_criteria, 'disp': False})

            self._number_gradient_iterations_l.append(results.nit)

            theta = results.x

            # reform the theta into the new S:
            L_k = np.zeros((M, M))
            for i in range(0,M):
                for j in range(i,M):
                    place = i * M + j - ((i * (i + 1)) // 2)
                    L_k[i,j] = theta[place]

            S = L_k.transpose().dot(L_k)
            for trial_i in range(0,self._numb_trials):
                self._S_covars[trial_i,lat_k,:,:] = S.copy()

            self._set_S_full_from_Sk()

            # print(self._compute_likelihood_term_full() - self._compute_KL_term_full())


        # print("done updating S")


        ## update of m using grad method:
        self._E_grad_update_m()



    def __minus_grad_L(self,theta):
        """
        The gradient of the free energy with respect to m.
        """
        M = self._numb_inducing # just making things clearer to read here
        Mk = M * self._K_numb_latents
        L = np.zeros((Mk, Mk))

        for i in range(0,Mk):
            for j in range(i,Mk):
                place = i * Mk + j - ((i * (i + 1)) // 2)
                L[i,j] = theta[place]

        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()
        sum_T = 0
        for bin_i in range(0,self._numb_bins):
            sum_T += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinvC).dot(self._temp_param_K_i_list[bin_i]).dot(L)

        # with gradient: (need to constraint the matrix)
        L_grad =  -  sum_T -  self._temp_param_inv_K_tilde.dot(L) + np.linalg.inv(L).transpose()

        numb_param = ((Mk * (Mk + 1)) // 2)


        theta = np.zeros(numb_param)

        for i in range(0,Mk):
            for j in range(i,Mk):
                place = i * Mk + j - ((i * (i + 1)) // 2)
                theta[place] = - L_grad[i,j]

        return theta

    def _E_step_grad_full(self):
        """
        E step with no mean field assumption: getting the full covariance matrix.

        """
        
        ###############     The method with the optimise function      #####################

        # print(self._compute_likelihood_term_full() - self._compute_KL_term_full())
        # create the theta vector to have all of S:
        
        Mk =self._numb_inducing * self._K_numb_latents
        numb_param = ((Mk * (Mk + 1)) // 2)

        # L = np.linalg.cholesky(self._S_full_matr)
        # try:
        L = np.linalg.cholesky(self._S_full_matr)
        # except:
        #     L = np.linalg.cholesky(np.eye(Mk))
        L = L.transpose()

        theta = np.zeros(numb_param)

        for i in range(0,Mk):
            for j in range(i,Mk):
                place = i * Mk + j - ((i * (i + 1)) // 2)
                theta[place] = L[i,j]
                
        # print(np.sum(np.abs(theta)), np.sum(np.abs(L)), np.mean(np.abs(theta)))

        # # optimise:
        use_minimise_function = True
        if use_minimise_function:
            results = minimize(self._free_energy_opt_embelisher_for_S, theta, method='L-BFGS-B', \
                    jac=self.__minus_grad_L, options={'ftol': self._convergence_criteria, 'disp': False})
                    
            self._number_gradient_iterations_l.append(results.nit)
            theta = results.x
        else:
            old_funct_val = 10
            diff = 1
            count = 0
            while np.abs(diff) > self._convergence_criteria:
                count += 1

                free_current = self._free_energy_opt_embelisher_for_S(theta)
                print(free_current)

                theta -= self.__minus_grad_L(theta) * self._lr_grad_update_learning_rate

                if count == 10:
                    count = 0
                    old_funct_val = free_current
                    free_current = self._free_energy_opt_embelisher_for_S(theta)
                    print(free_current)
                    diff = (old_funct_val - free_current) / max(np.abs(free_current), 1, np.abs(old_funct_val))
                    print(diff)

        # free_current = self._free_energy_opt_embelisher_for_S(theta)
        # print("done updating S...", free_current)


        L = np.zeros((Mk, Mk))

        for i in range(0,Mk):
            for j in range(i,Mk):
                place = i * Mk + j - ((i * (i + 1)) // 2)
                L[i,j] = theta[place]

        S = L.transpose().dot(L)

        L = np.linalg.cholesky(S)

        self._S_full_matr = S.copy()
        self._set_Sk_from_S_full()

        # print(self._compute_likelihood_term_full() - self._compute_KL_term_full())
            
        # print("done updating S")

        self._E_grad_update_m()

        # update other things:
        self._C_matrix_big = self._C_matrix


    def __minus_grad_m(self,m,trial_i):
        """
        The gradient of the free energy with respect to m.
        """
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        first_term_Ti = 0
        for bin_i in range(0,self._numb_bins):
            m_tilde_i_r = self._temp_param_K_i_list[bin_i].dot(m)
            mu_r_i = self._dataset[trial_i,:,bin_i] - self._d_bias - self._C_matrix.dot(m_tilde_i_r)

            first_term_Ti += np.transpose(self._temp_param_K_i_list[bin_i]).dot(CRinv).dot( mu_r_i )


        second_term = self._temp_param_inv_K_tilde.dot(m)

        grad_m = first_term_Ti - second_term

        return - grad_m



    def _E_grad_update_m(self):
        """
        Using a gradient step for m.

        Put on the side because it is the same for both mf and full.
        """

        ##### method with "minimize"

        for trial_i in range(0,self._numb_trials):

            x_0 = self._temp_param_m[trial_i,:]
            # result = minimize(self._compute_free_energy_para_taus, x0, method='L-BFGS-B', jac=gradient_func, \
            #     options={'disp': None, 'maxcor': 6, 'maxfun': 150, 'maxiter': 15, 'iprint': -1, 'maxls': 6})

            results = minimize(self._free_energy_opt_embelisher_for_m, x_0, args=(trial_i), method='L-BFGS-B', \
                 jac=self.__minus_grad_m, options={'ftol': self._convergence_criteria, 'disp': False}) # jac=self.__grad_m,

            self._number_gradient_iterations_l.append(results.nit)
                #print(results)

            self._temp_param_m[trial_i,:] = results.x

            # print(self._compute_likelihood_term_full() - self._compute_KL_term_full())



    def _M_step_full(self):
        """
        Using the matrix to update things:
        """

        d_sum_term = 0
        C1_sum_term = 0
        C2_sum_term = 0
        R_sum_term = 0
        
        C_covar_loging_term = 0

        for bin_i in range(0,self._numb_bins):

            # get K_ii:
            for lat_k in range(0,self._K_numb_latents):
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii - k_iZ.dot(self._inv_kernel_mat_induc[lat_k,:,:]).dot(k_Zi)

                if lat_k == 0:
                    K_tilde_ii = temp
                else:
                    K_tilde_ii = block_diag(K_tilde_ii , temp)

            correcting_term = np.matmul( np.matmul(self._temp_param_K_i_list[bin_i], self._S_full_matr), np.transpose(self._temp_param_K_i_list[bin_i]) )

            # put together to have S_ii:
            S_tilde_ii = K_tilde_ii + correcting_term

            # get Sigma_ii:
            Sigma_ii = self._C_matrix.dot(S_tilde_ii).dot(np.transpose(self._C_matrix))

            # print( (correcting_term != 0).astype(int) )
            # print( correcting_term )
            #print( (S_hat != 0).astype(int) )
            
            # now the trial sum:
            for trial_i in range(0, self._numb_trials):

                # get m_tilde i r
                m_tilde_i_r = self._temp_param_K_i_list[bin_i].dot(self._temp_param_m[trial_i,:])

                # get mu_i_r
                mu_i_r = self._C_matrix.dot(m_tilde_i_r) + self._d_bias

                # update sum terms:
                # C:
                C1_sum_term += np.outer( (self._dataset[trial_i,:,bin_i] - self._d_bias) , m_tilde_i_r )
                C2_sum_term += np.outer(m_tilde_i_r, m_tilde_i_r) + S_tilde_ii
                C_covar_loging_term += S_tilde_ii

                # d:
                d_sum_term += self._dataset[trial_i,:,bin_i] - self._C_matrix.dot(m_tilde_i_r)

                # R:
                temp = self._dataset[trial_i,:,bin_i] - mu_i_r
                R_sum_term += np.outer(temp, temp) + Sigma_ii

        denominator = np.linalg.inv(C2_sum_term)
        self._C_matrix = C1_sum_term.dot(denominator)
        # print(denominator.dot(denominator.transpose()))
        self._gather_data_Cupdate(C1_sum_term, C2_sum_term, C_covar_loging_term)
        
        # print("orth numerator C: ",get_orthogonality_score(C1_sum_term, False)[0] )
        # print( get_orthogonality_score(C2_sum_term, False)[0], get_orthogonality_score(np.linalg.inv(C2_sum_term), False)[0] , get_orthogonality_score(self._C_matrix, False)[0] )
        # print(C2_sum_term, np.linalg.inv(C2_sum_term) * 10000)

        # if not self._learn_only_C:

        self._d_bias = (1 / ( self._numb_trials * self._numb_bins )) * d_sum_term

        self._R_noise = (1 / ( self._numb_trials * self._numb_bins )) * R_sum_term
        if self._force_R_diag:
            self._R_noise = np.diag(np.diag(self._R_noise))



    def _do_Rinv_precompute(self):
        # some precomputations:
        if self._force_R_diag:
            Rinv = np.diag(1 / (np.diag(self._R_noise) ))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise))

        CRinv = np.matmul( np.transpose(self._C_matrix), Rinv )
        CRinvC = np.matmul( CRinv, self._C_matrix )

        return Rinv, CRinv, CRinvC, logdet_R



    #########################      getting the free energy      ###################


    def _compute_free_energy_para_taus(self, param):
        """
        Compute and return the free energy from the kernel parameters.

        This function is to be used with auto grad to update the kernel parameters.
        
        """

        # TODO: in the computation of all the kernel things, make it use the input params

        kernel_param = {"tau" : param, "sigma_n" : self._kernel_param["sigma_n"]}

        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        first_term = 0
        second_term = 0
        big_term = 0

        inv_Kzz_list = []
        sum_over_k_second = 0
        for lat_k in range(0,self._K_numb_latents):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
            inv_Kzz = np.linalg.inv(K_zz)
            inv_Kzz_list.append(inv_Kzz)

            for trial_i in range(0, self._numb_trials):
                expr = np.trace( np.matmul(inv_Kzz , self._S_covars[trial_i,lat_k,:,:]) )

                expr += np.matmul( np.matmul( np.transpose( self._m_induc_mean[trial_i,lat_k,:] ), inv_Kzz ), self._m_induc_mean[trial_i,lat_k,:])

                expr += np.log( np.linalg.det(K_zz) ) - self._numb_inducing - np.log( np.linalg.det(self._S_covars[trial_i,lat_k,:,:]) )

                sum_over_k_second += 0.5 * expr


        first_term += - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):
                # K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
                # inv_Kzz = np.linalg.inv(K_zz)
                inv_Kzz = inv_Kzz_list[lat_k]

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz ,self._S_covars[0,lat_k,:,:]), inv_Kzz )
                matr_term -= inv_Kzz

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp



            for trial_i in range(0,self._numb_trials):

                # forming m tilde:
                m_tilde = np.zeros((self._K_numb_latents))

                e_unit_vect = np.eye(self._K_numb_latents)

                for lat_k in range(0,self._K_numb_latents):
                    # K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
                    # inv_Kzz = np.linalg.inv(K_zz)
                    inv_Kzz = inv_Kzz_list[lat_k]
                    K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
                    m_tilde += e_unit_vect[lat_k,:] * np.dot(np.matmul(K_new,inv_Kzz),self._m_induc_mean[trial_i,lat_k,:])


                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[trial_i,:,bin_i]) , Rinv), self._dataset[trial_i,:,bin_i])

                # combining the two m K_i C terms:
                temp = np.transpose(m_tilde)
                first_term += np.matmul( temp, np.matmul( CRinv, (self._dataset[trial_i,:,bin_i] - self._d_bias   )  ) )

                # combining the two d terms:
                first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[trial_i,:,bin_i] - 0.5 * self._d_bias ) )

                # the big term:
                temp = m_tilde
                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

                first_term += - 0.5 * sum_over_k_first

        
        F = first_term - sum_over_k_second

        return F[0,0]




    def _compute_KL_term(self):
        """
        Compute the KL term.
        
        """

        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        kl_term = 0
        for lat_k in range(0,self._K_numb_latents):

            for trial_i in range(0, self._numb_trials):
                expr = np.trace( np.matmul(self._inv_kernel_mat_induc[lat_k,:,:] , self._S_covars[trial_i,lat_k,:,:]) )

                expr += np.matmul( np.matmul( np.transpose( self._m_induc_mean[trial_i,lat_k,:] ), self._inv_kernel_mat_induc[lat_k,:,:] ), self._m_induc_mean[trial_i,lat_k,:])

                expr += np.linalg.slogdet(self._kernel_mat_induc[lat_k,:,:])[1] - self._numb_inducing - np.linalg.slogdet(self._S_covars[trial_i,lat_k,:,:])[1]

                kl_term += 0.5 * expr

        # print("kl_term",kl_term)

        return kl_term
        

    def _compute_KL_term_full(self, S_full_matr = None, param_m = None):
        """
        Compute the KL term.
        
        """
        if S_full_matr is None:
            S_full_matr = self._S_full_matr.copy()
        if param_m is None:
            param_m = self._temp_param_m.copy()

        # kl_term = 0
        # # kl_term += 0.5 * self._K_numb_latents * self._numb_inducing * (np.log(2 * np.pi * np.e ) - np.log(2 * np.pi ))
        
        # for trial_i in range(0, self._numb_trials):
        #     expr = np.trace( np.matmul(self._temp_param_inv_K_tilde , S_full_matr) )

        #     expr += np.matmul( np.matmul( np.transpose( param_m[trial_i,:] ), self._temp_param_inv_K_tilde ), param_m[trial_i,:])

        #     # if self._numb_inducing != len(self._bin_times):
        #     expr += np.linalg.slogdet(self._temp_param_K_tilde)[1] - np.linalg.slogdet(S_full_matr)[1] - self._numb_inducing * self._K_numb_latents
        #     # else: 
        #         # expr += np.log( np.linalg.det(self._temp_param_K_tilde) + SMALL ) - np.log( np.linalg.det( self._S_full_matr) + SMALL ) - self._numb_inducing * self._K_numb_latents

        #     kl_term += 0.5 * expr

        # # print("kl_term",kl_term)
        param_m_induc_mean = np.zeros((self._numb_trials, self._K_numb_latents, self._numb_inducing))


        for trial_i in range(0, self._numb_trials):
            count = 0
            for lat_k in range(0,self._K_numb_latents):
                param_m_induc_mean[trial_i,lat_k,:] = param_m[trial_i,count:count+self._numb_inducing]
                count += self._numb_inducing

        kl_to_prior = 0

        # KL term:
        kl_to_prior -= 0.5 * np.linalg.slogdet(S_full_matr)[1]
        kl_to_prior -= 0.5 * self._K_numb_latents * self._numb_inducing #* (np.log(2 * np.pi * np.e ) - np.log(2 * np.pi ))

        kl_to_prior += 0.5 * np.linalg.slogdet(self._temp_param_K_tilde )[1]
        kl_to_prior += 0.5 * np.trace(self._temp_param_inv_K_tilde.dot(S_full_matr))

        kl_to_prior *= self._numb_trials

        for trial_i in range(0, self._numb_trials):

            # KL term:
            for lat_k in range(0, self._K_numb_latents):
                temp = param_m_induc_mean[trial_i,lat_k,:]#self._m_induc_mean[trial_i,lat_k,:]

                kl_to_prior += 0.5 * np.trace(self._inv_kernel_mat_induc[lat_k,:,:].dot(np.outer(temp, temp)))

        # print(np.linalg.slogdet(self._S_full_matr)[1], np.linalg.slogdet(self._temp_param_K_tilde )[1], np.trace(self._temp_param_inv_K_tilde.dot(self._S_full_matr)))

        return kl_to_prior


    def _compute_likelihood_term(self):
        """
        Compute the likelihood term.
        
        """
        
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()


        first_term = - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):
                
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( self._inv_kernel_mat_induc[lat_k,:,:] ,self._S_covars[0,lat_k,:,:]), self._inv_kernel_mat_induc[lat_k,:,:] )
                matr_term -= self._inv_kernel_mat_induc[lat_k,:,:]

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp

            for trial_i in range(0,self._numb_trials):

                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[trial_i,:,bin_i]) , Rinv), self._dataset[trial_i,:,bin_i])
                
                m_tilde = np.dot(self._temp_param_m[trial_i,:], self._temp_param_K_i_list[bin_i].transpose())

                # combining the two m K_i C terms:
                temp = np.transpose(m_tilde)
                first_term += np.matmul( temp, np.matmul( CRinv, (self._dataset[trial_i,:,bin_i] - self._d_bias   )  ) )

                # combining the two d terms:
                first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[trial_i,:,bin_i] - 0.5 * self._d_bias ) )

                # the big term:
                temp = m_tilde
                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

                first_term += - 0.5 * sum_over_k_first

        return first_term[0,0]


    def _compute_free_energy_full(self):

        like = self._compute_likelihood_term_full()
        kl = self._compute_KL_term_full()

        #true_like = self.compute_likelihood()

        return like - kl, like, kl#, true_like, true_like - (like - kl)


    def _free_energy_opt_embelisher_for_S_mf(self, theta, latent_to_derive):
        """
        This is method takes some parameters as input and uses them to compute the free energy
            it is used for the gradient methods updates.
        """
        theta_learnt = theta

        M = self._numb_inducing # just making things clearer to read here
        Mk = M * self._K_numb_latents
        numb_param = ((M * (M + 1)) // 2)

        # rearange the matrix to have S_full_matr
        L = np.zeros((M, M))

        for i in range(0,M):
            for j in range(i,M):
                place = i * M + j - ((i * (i + 1)) // 2)
                L[i,j] = theta[place]

        S_k = L.transpose().dot(L)


        S = np.zeros((Mk, Mk))

        for lat_k in range(0, self._K_numb_latents):
            if lat_k == latent_to_derive:
                S[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M] = S_k
                
            else:
                S[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M] = self._S_covars[0,lat_k,:,:]

        return -1 * self._compute_free_energy_full_for_opt(S, self._temp_param_m)




    def _free_energy_opt_embelisher_for_S(self, theta):
        """
        This is method takes some parameters as input and uses them to compute the free energy
            it is used for the gradient methods updates.
        """
        theta_learnt = theta

        M = self._numb_inducing # just making things clearer to read here
        Mk = M * self._K_numb_latents
        numb_param = ((Mk * (Mk + 1)) // 2)

        # rearange the matrix to have S_full_matr
        L = np.zeros((Mk, Mk))

        for i in range(0,Mk):
            for j in range(i,Mk):
                place = i * Mk + j - ((i * (i + 1)) // 2)
                L[i,j] = theta[place]

        S = L.transpose().dot(L)

        return -1 * self._compute_free_energy_full_for_opt(S, self._temp_param_m)


    

    def _free_energy_opt_embelisher_for_m(self, m, trial_i):

        param_m = self._temp_param_m.copy()

        param_m[trial_i,:] = m

        return -1 * self._compute_free_energy_full_for_opt(self._S_full_matr, param_m)


    def _compute_free_energy_full_for_opt(self, S_full_matr, param_m):
        """
        This is method takes some parameters as input and uses them to compute the free energy
            it is used for the gradient methods updates.
        """

        like = self._compute_likelihood_term_full(S_full_matr, param_m)
        kl = self._compute_KL_term_full(S_full_matr, param_m)

        return (like - kl)



    def _compute_likelihood_term_full(self, S_full_matr = None, param_m = None):
        """
        Compute the likelihood term.
        
        """
        if S_full_matr is None:
            S_full_matr = self._S_full_matr.copy()
        if param_m is None:
            param_m = self._temp_param_m.copy()
        
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()


        first_term = - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii - np.matmul( np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:]), k_Zi)

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp

            matr_term = np.matmul( np.matmul( self._temp_param_K_i_list[bin_i], S_full_matr ), self._temp_param_K_i_list[bin_i].transpose() )
            matr_term = np.matmul( np.matmul( self._C_matrix, matr_term), self._C_matrix.transpose() )

            second_term = np.trace( np.matmul(Rinv , (matr_term ) ) ) + sum_over_k_first


            for trial_i in range(0,self._numb_trials):

                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[trial_i,:,bin_i]) , Rinv), self._dataset[trial_i,:,bin_i])
                
                m_tilde = np.dot(param_m[trial_i,:], self._temp_param_K_i_list[bin_i].transpose())
                mu_t = np.matmul( self._C_matrix, m_tilde) + self._d_bias

                # combining the two m K_i C terms:
                first_term += np.matmul( np.transpose(mu_t), np.matmul( Rinv, (self._dataset[trial_i,:,bin_i]  )  ) )

                # combining the two d terms:
                # first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[trial_i,:,bin_i] - 0.5 * self._d_bias ) )

                # the big term:
                first_term += - 0.5 * np.matmul( np.matmul( np.transpose(mu_t), Rinv ), mu_t )

                first_term += - 0.5 * second_term

        return first_term[0,0]




    #########################      computing other metrics      ###################

    def compute_likelihood(self):
        """
        Compute and return the log likelihood (LL).
        """

        # create C tilde:
        C_tilde = np.zeros((self._numb_bins*self._numb_neurons,self._K_numb_latents*self._numb_bins))
        for lat_k in range(0,self._K_numb_latents):
            C_k = np.kron(np.eye(self._numb_bins), self._C_matrix[:,lat_k])
            # print( (C_k != 0).astype(int) )
            C_tilde[:,self._numb_bins*lat_k:self._numb_bins*(lat_k+1)] = C_k.transpose()

        R_big = np.kron(np.eye(self._numb_bins), self._R_noise)

        likelihood = 0

        d_big = np.kron(np.ones(self._numb_bins), self._d_bias)

        kernel_xx_mat_induc = np.zeros((self._K_numb_latents, self._numb_bins, self._numb_bins))
        for lat_k in range(0,self._K_numb_latents):
            K_xx = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
            kernel_xx_mat_induc[lat_k,:,:] = K_xx

        K_tilde_XX = kernel_xx_mat_induc[0,:,:]
        for lat_k in range(1,self._K_numb_latents):
            K_tilde_XX = block_diag(K_tilde_XX, kernel_xx_mat_induc[lat_k,:,:])
             
        sigma_like = R_big + C_tilde.dot(K_tilde_XX).dot(np.transpose(C_tilde)) 
        sigma_like_inv = np.linalg.inv( sigma_like )

        log_det = np.linalg.slogdet(sigma_like)[1]

        for trial_i in range(0,self._numb_trials):
            y_big = np.transpose(self._dataset[trial_i,:,:]).reshape(self._numb_bins*self._numb_neurons)

            big_diff = y_big - d_big

            likelihood += - 0.5 * self._numb_bins * self._numb_neurons * np.log(2 * np.pi)
            likelihood += - 0.5 * big_diff.dot(sigma_like_inv).dot(big_diff)
            likelihood += - 0.5 * log_det


        return likelihood

    def compute_likelihood_old(self):
        """
        Compute the likelihood.

        """
        
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        C_tilde = np.zeros((self._numb_bins*self._numb_neurons,self._K_numb_latents*self._numb_bins))
        for lat_k in range(0,self._K_numb_latents):
            C_k = np.kron(np.eye(self._numb_bins), self._C_matrix[:,lat_k])
            # print( (C_k != 0).astype(int) )
            C_tilde[:,self._numb_bins*lat_k:self._numb_bins*(lat_k+1)] = C_k.transpose()

        Rinv_big = np.kron(np.eye(self._numb_bins), Rinv)
        R_big = np.kron(np.eye(self._numb_bins), self._R_noise)

        K_tau = 0
        kappa_tau_tau = 0

        for lat_k in range(0,self._K_numb_latents):
                k_ii = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii - np.matmul( np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:]), k_Zi)

                temp_for_K = np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:])

                if lat_k == 0:
                    kappa_tau_tau = temp
                    K_tau = temp_for_K
                else:
                    kappa_tau_tau = block_diag(kappa_tau_tau, temp)
                    K_tau = block_diag(K_tau, temp_for_K)

        temp_C_K_tau = np.matmul( C_tilde, K_tau)
        matr_term = np.matmul( np.matmul( temp_C_K_tau, self._temp_param_K_tilde), temp_C_K_tau.transpose())

        kappa_tau_tau = np.matmul( np.matmul( C_tilde, kappa_tau_tau), C_tilde.transpose())

        sigma = R_big + kappa_tau_tau + matr_term

        likelihood = - (0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        # get the inverse
        inv_sigma = np.linalg.inv(sigma)

        log_det = np.linalg.slogdet(sigma)[1] #np.log( np.linalg.det(sigma) )

        # print(np.linalg.det(sigma + SMALL), np.linalg.det(sigma + SMALL) == 0 )
        # print(np.linalg.slogdet(sigma), np.linalg.slogdet(sigma) == 0 )


        d_big = np.kron(np.ones(self._numb_bins), self._d_bias)

        ## compute the core:
        for trial_i in range(0,self._numb_trials):

            y_big = np.transpose(self._dataset[trial_i,:,:]).reshape(self._numb_bins*self._numb_neurons)

            side = y_big - d_big

            likelihood += - 0.5 * np.matmul( np.matmul( side.transpose()  , inv_sigma), side )

            likelihood += - 0.5 * log_det

        return likelihood





    def compute_the_expected_log_marginal(self):
        """
        Compute the expected log marginal.

        Carefull, this might be wrong and need the same change as the likelihood

        """

        log_marginal = - (0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        ## get the sum over T
        for bin_i in range(0,self._numb_bins):

            ## create Sigma (the covariance matrix)

            # create kappa
            kappa = 0
            for lat_k in range(0,self._K_numb_latents):

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii - np.matmul( np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:]), k_Zi)

                # kappa += np.matmul( np.matmul( self._C_matrix[:,lat_k], temp), self._C_matrix[:,lat_k].transpose() )
                kappa += np.outer( np.outer( self._C_matrix[:,lat_k], temp ), np.transpose(self._C_matrix[:,lat_k] ) )
                
            matr_term = np.matmul( np.matmul( self._temp_param_K_i_list[bin_i], self._S_full_matr ), self._temp_param_K_i_list[bin_i].transpose() )
            matr_term = np.matmul( np.matmul( self._C_matrix, matr_term), self._C_matrix.transpose() )

            sigma = kappa + matr_term + self._R_noise

            # temp = block_diag(kappa, self._S_full_matr)
            # print(np.mean(temp),np.mean(np.abs(temp)),np.linalg.det(temp))
            
            # get the inverse
            inv_sigma = np.linalg.inv(sigma)

            c_k = np.matmul( self._C_matrix, self._temp_param_K_i_list[bin_i])

            ## compute the core:
            for trial_i in range(0,self._numb_trials):

                c_k_m = np.matmul( c_k, self._temp_param_m[trial_i,:])
                side = self._dataset[trial_i,:,bin_i] - c_k_m - self._d_bias

                log_marginal += - 0.5 * np.matmul( np.matmul( side.transpose()  , inv_sigma), side )

                log_marginal += - 0.5 * np.log( np.linalg.det(sigma) )

        return log_marginal





    def compute_KL_to_true_posterior_full(self):
        """
        Computes the KL to the true posterior.

        """
        
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()
        
        kl_term = 0

        for bin_i in range(0,self._numb_bins):
            #print("-----------------")

            # create kappa
            kappa = 0
            for lat_k in range(0,self._K_numb_latents):

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii - np.matmul( np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:]), k_Zi)

                #print(temp)
                #print(np.mean(self._inv_kernel_mat_induc[lat_k,:,:]),np.mean(np.abs(self._inv_kernel_mat_induc[lat_k,:,:])),np.linalg.det(temp))

                # kappa += np.matmul( np.matmul( self._C_matrix[:,lat_k], temp), self._C_matrix[:,lat_k].transpose() )
                kappa += np.outer( np.outer( self._C_matrix[:,lat_k], temp ), np.transpose(self._C_matrix[:,lat_k] ) )
                
            #kappa += SMALL

            C_Kt = np.matmul( self._C_matrix, self._temp_param_K_i_list[bin_i])
                
            C_Kt_S_Kt_C = np.matmul( np.matmul( C_Kt, self._S_full_matr ), C_Kt.transpose() )

            # create sigmas:
            kappa_inv = np.linalg.inv(kappa)
            #kappa_inv = (kappa)

            kappa_prime = kappa + C_Kt_S_Kt_C
            kappa_prime_inv = np.linalg.inv(kappa_prime)

            sigma_h_inv = Rinv + kappa_inv
            sigma_h = np.linalg.inv(sigma_h_inv)

            sigma_u_inv = self._temp_param_inv_K_tilde + np.matmul( np.matmul( C_Kt.transpose(), kappa_inv ), C_Kt )
            sigma_u = np.linalg.inv(sigma_u_inv)

            first_term_hy = 0
            second_term_uh = 0
            third_term_entrop_h = 0
            fourth_term_entropy_u = 0

            C_Kt_Sigu_Kt_C = np.matmul( np.matmul( C_Kt, sigma_u.transpose() ), C_Kt.transpose() )
            
            # print(np.mean(kappa),np.mean(np.abs(kappa)),np.linalg.det(kappa),np.mean(kappa_inv),np.mean(np.abs(kappa_inv)),np.linalg.det(kappa_inv))
            

            for trial_i in range(0, self._numb_trials):
                # create mus:
                m = self._temp_param_m[trial_i,:]
                y = self._dataset[trial_i,:,bin_i]
                d = self._d_bias.copy()

                C_Kt_m = np.matmul( C_Kt, m )

                # first, the first term:
                A = sigma_h_inv - 2 * kappa_inv + np.matmul( np.matmul( kappa_inv, sigma_h ), kappa_inv )

                first_term_hy = np.linalg.slogdet(sigma_h)[1] + self._numb_neurons * np.log(2 * np.pi)

                Rinv_Sigma_h =  np.matmul( Rinv , sigma_h )
                first_term_hy += np.matmul( np.matmul( y.transpose(), Rinv_Sigma_h ), np.matmul( Rinv, y ) )

                first_term_hy += np.trace( np.matmul( sigma_h_inv, kappa ) )

                first_term_hy += np.trace( np.matmul( A, C_Kt_S_Kt_C ) )

                first_term_hy += np.matmul( np.matmul( C_Kt_m.transpose() , A ), C_Kt_m )

                first_term_hy += 2 * np.matmul( np.matmul( self._d_bias.transpose() , A ), C_Kt_m )

                first_term_hy += np.matmul( np.matmul( self._d_bias.transpose() , A ), self._d_bias )

                temp = np.matmul( Rinv_Sigma_h, kappa_inv ) - Rinv
                first_term_hy += 2 * np.matmul( np.matmul( y.transpose() , temp ), (C_Kt_m + self._d_bias) )

                first_term_hy *= - 0.5

                # then, the second term:
                second_term_uh = np.linalg.slogdet(sigma_u)[1] + self._K_numb_latents * self._numb_inducing * np.log(2 * np.pi)
                
                second_term_uh += 4 * d.transpose().dot(kappa_inv).dot(C_Kt_Sigu_Kt_C).dot(kappa_inv).dot(d)

                second_term_uh += np.trace(sigma_u_inv.dot(self._S_full_matr))
                
                second_term_uh += np.matmul( np.matmul( m.transpose() , sigma_u_inv ), m )

                second_term_uh += - 2 * np.trace(kappa_inv.dot(C_Kt_S_Kt_C))

                second_term_uh += - 2 * np.matmul( np.matmul( m.transpose() , C_Kt.transpose().dot(kappa_inv).dot(C_Kt) ), m )

                second_term_uh += np.trace(kappa_inv.dot(C_Kt_Sigu_Kt_C).dot(kappa_inv).dot(C_Kt_S_Kt_C))

                second_term_uh += np.trace(kappa_inv.dot(C_Kt_Sigu_Kt_C))

                second_term_uh += np.matmul( np.matmul( C_Kt_m.transpose() , kappa_inv.dot(C_Kt_Sigu_Kt_C).dot(kappa_inv) ), C_Kt_m )

                second_term_uh *= - 0.5


                # then the entropy of h:
                third_term_entrop_h = np.linalg.slogdet(kappa)[1] + self._numb_neurons * (np.log(2 * np.pi) + 1)
                third_term_entrop_h *= 0.5 


                # then the entropy of u:
                fourth_term_entropy_u = np.linalg.slogdet(self._S_full_matr)[1] + self._numb_inducing * self._K_numb_latents * (np.log(2 * np.pi) + 1)
                fourth_term_entropy_u *= 0.5 


                kl_term += first_term_hy + second_term_uh + third_term_entrop_h + fourth_term_entropy_u


        return kl_term


    def _compute_free_energy_para_taus_fake(self, param):
        """
        Compute and return the free energy from the kernel parameters.

        This function is to be used with auto grad to update the kernel parameters.
        
        """

        like = self.compute_likelihood()

        kl_term = like - self._compute_free_energy_full()[0] #self.compute_KL_to_true_posterior_full()

        F = like - kl_term

        # print(F,first_term,sum_over_k_second)

        self._free_energy_1_list.append(like)
        self._free_energy_2_list.append(kl_term)
        return F


    # def _compute_free_energy_para_taus_old(self, param):
    #     """
    #     Compute and return the free energy from the kernel parameters.

    #     This function is to be used with auto grad to update the kernel parameters.
        
    #     """

    #     # TODO: in the computation of all the kernel things, make it use the input params

    #     kernel_param = {"tau" : param, "sigma_n" : self._kernel_param["sigma_n"]}

    #     # some precomputations:
    #     if self._force_R_diag:
    #         Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
    #         logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
    #     else:
    #         Rinv = np.linalg.inv(self._R_noise)
    #         Rinv = ( Rinv + np.transpose(Rinv) ) / 2
    #         logdet_R = np.log(np.linalg.det(self._R_noise))

    #     CRinv = np.matmul( np.transpose(self._C_matrix), Rinv )
    #     CRinvC = np.matmul( CRinv, self._C_matrix )

    #     # TODO: rearange loops so that it is efficient even without saving the arrays
    #     # some precomputations involving the params:
    #     # kernel_mat_induc = np.zeros((self._K_numb_latents, self._numb_inducing, self._numb_inducing))
    #     # inv_kernel_mat_induc = np.zeros((self._K_numb_latents, self._numb_inducing, self._numb_inducing))

    #     # for lat_k in range(0,self._K_numb_latents):
    #     #     K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #     #     kernel_mat_induc[lat_k,:,:] = K_zz
    #     #     inv_kernel_mat_induc[lat_k,:,:] = np.linalg.inv(K_zz)

    #     # the list of the K_i as defined in the report
    #     # K_i_list = []
    #     # for bin_i in range(0,self._numb_bins):
    #     #     # create K_i
    #     #     K_i = np.matmul( make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, 0),(self._kernel_mat_induc[0,:,:]) )
    #     #     for lat_k in range(1,self._K_numb_latents):
    #     #         K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #     #         K_i = block_diag(K_i, np.matmul( K_new, self._kernel_mat_induc[lat_k,:,:]) )
    #     #     K_i_list.append(K_i)


    #     first_term = 0
    #     second_term = 0
    #     big_term = 0

    #     first_term += - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

    #     for bin_i in range(0,self._numb_bins):

    #         sum_over_k_first = 0
    #         for lat_k in range(0,self._K_numb_latents):
    #             K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #             inv_Kzz = np.linalg.inv(K_zz)

    #             k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, kernel_param, lat_k)
    #             k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #             k_Zi = np.transpose(k_iZ)
    #             matr_term = np.matmul( np.matmul( inv_Kzz ,self._S_covars[0,lat_k,:,:]), inv_Kzz )
    #             matr_term -= inv_Kzz

    #             temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

    #             sum_over_k_first += CRinvC[lat_k,lat_k] * temp


    #         for trial_i in range(0,self._numb_trials):
    #             K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, 0)
    #             inv_Kzz = np.linalg.inv(K_zz)
    #             K_i = np.matmul( make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, 0),(inv_Kzz) )
    #             for lat_k in range(1,self._K_numb_latents):
    #                 K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #                 inv_Kzz = np.linalg.inv(K_zz)
    #                 K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #                 K_i = block_diag(K_i, np.matmul( K_new, inv_Kzz ) )
                    

    #             # this should be correct
    #             # first_term += - 0.5 * logdet_R - 0.5 * self._numb_neurons * np.log(2 * np.pi)

    #             first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[trial_i,:,bin_i]) , Rinv), self._dataset[trial_i,:,bin_i])

    #             # combining the two m K_i C terms:
    #             temp = np.matmul( np.transpose(self._temp_param_m[trial_i,:]) , np.transpose(K_i))
    #             first_term += np.matmul( temp, np.matmul( CRinv, (self._dataset[trial_i,:,bin_i] - self._d_bias   )  ) )

    #             # combining the two d terms:
    #             first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[trial_i,:,bin_i] - 0.5 * self._d_bias ) )

    #             # the big term:
    #             temp = np.matmul(  K_i , self._temp_param_m[trial_i,:])
    #             first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )
    #             big_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

    #             first_term += - 0.5 * sum_over_k_first

        
    #     sum_over_k_second = 0
    #     for lat_k in range(0,self._K_numb_latents):
    #         K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, kernel_param, lat_k)
    #         inv_Kzz = np.linalg.inv(K_zz)

    #         expr = np.trace( np.matmul(inv_Kzz , self._S_covars[0,lat_k,:,:]) )

    #         expr += np.matmul( np.matmul( np.transpose( self._m_induc_mean[0,lat_k,:] ), inv_Kzz ), self._m_induc_mean[0,lat_k,:])

    #         expr += np.log( np.linalg.det(K_zz) ) - self._numb_inducing - np.log( np.linalg.det(self._S_covars[0,lat_k,:,:]) )

    #         sum_over_k_second += 0.5 * expr

    #     F = first_term - self._numb_trials *sum_over_k_second

    #     self._free_energy_1_list.append(first_term[0,0])
    #     self._free_energy_2_list.append(big_term)
    #     return F[0,0]


    # def compute_likelihood_withoutT(self):
    #     """
    #     Compute the likelihood.

    #     """

    #     likelihood = - (0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

    #     ## get the sum over T
    #     for bin_i in range(0,self._numb_bins):

    #         ## create sigma (the covariance matrix)

    #         # create kappa
    #         kappa = 0
    #         for lat_k in range(0,self._K_numb_latents):

    #             k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
    #             k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
    #             k_Zi = np.transpose(k_iZ)

    #             temp = k_ii - np.matmul( np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:]), k_Zi)

    #             # kappa += np.matmul( np.matmul( self._C_matrix[:,lat_k], temp), self._C_matrix[:,lat_k].transpose() )
    #             kappa += np.outer( np.outer( self._C_matrix[:,lat_k], temp ), np.transpose(self._C_matrix[:,lat_k] ) )
                
    #         matr_term = np.matmul( np.matmul( self._temp_param_K_i_list[bin_i], self._temp_param_K_tilde ), self._temp_param_K_i_list[bin_i].transpose() )
    #         matr_term = np.matmul( np.matmul( self._C_matrix, matr_term), self._C_matrix.transpose() )

    #         sigma = kappa + matr_term + self._R_noise

    #         # get the inverse
    #         inv_sigma = np.linalg.inv(sigma)

    #         log_det = np.log( np.linalg.det(sigma) )


    #         ## compute the core:
    #         for trial_i in range(0,self._numb_trials):

    #             side = self._dataset[trial_i,:,bin_i] - self._d_bias

    #             likelihood += - 0.5 * np.matmul( np.matmul( side.transpose()  , inv_sigma), side )

    #             likelihood += - 0.5 * log_det

    #     return likelihood




