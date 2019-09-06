"""

The linear response improvement of the multi class sparce variational approximation of the Gaussian Process Factor Analysis (GPFA) model.

This is just an extension of the GPFA_sv_mc class where add the method to compute the linear response correction of
    the covariance function.



"""

from algorithms.GPFA_sv_mc import GPFA_sv_mc
import time
from tqdm import tqdm 
from utilis import *
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# import numpy as np
# the auto grad things:
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad, grad
from autograd import hessian


SMALL = 1e-3


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)




class GPFA_sv_mc_lr(GPFA_sv_mc):
    """ sparce variational GPFA 
    
    """

    algName = "lr_mc_sv_GPFA"


    def __init__(self, dataset, meta_data, bin_times, numb_shared_lat, numb_grouped_lat, numb_latents, \
                 max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy):

        super().__init__(dataset, meta_data, bin_times, numb_shared_lat, numb_grouped_lat, numb_latents, \
                 max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy)



    def do_the_linear_response_correction(self):

        ## Linear Response:
        #for trial_i in range(0, self._numb_trials):

        trial_i = 0
        S_hat, m = self._linear_response_correction(trial_i)

        stats_on_Sm = self._compare_with_previous(S_hat, m)

        # self._E_step()

        # self._S_full_matr = S_hat

        return stats_on_Sm


    # TODO:
    def _compare_with_previous(self, S_hat, m):
        """
        Compares the newly found parameters with the ones of MF.

        """

        percentage_of_higer_S = 0
        abs_diff_S = 0
        abs_diff_m = 0

        M = self._numb_inducing
        for lat_k in range(0, self._K_numb_latents):
            diff = np.abs(S_hat[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]) - np.abs(self._S_covars_all[lat_k,:,:])
            abs_diff_S += np.mean(diff)
            percentage_of_higer_S += np.mean((diff > 0).astype(int))
            
        abs_diff_S *= (1 / self._K_numb_latents)
        percentage_of_higer_S *= (1 / self._K_numb_latents)

        # diff = np.abs(m) - np.abs(self._temp_param_m[self._temp_trial_i,:])
        # abs_diff_m += np.mean(diff)
        abs_diff_m = 0

        return percentage_of_higer_S, abs_diff_S, abs_diff_m


    def _linear_response_correction(self,trial_i):
        """
        Correcting the covariance between the q(u) using 

        """

        self._temp_trial_i = trial_i
        self._temp_cond = self._trial_conditions[trial_i]


        ################
        # get V and H

        # put together the m of this trial:
        self._m_induc_mean_all = np.concatenate([self._m_alpha_induc_mean,self._m_beta_induc_mean[self._temp_cond,:,:], self._m_gamma_induc_mean[trial_i,:,:]])
        self._S_covars_all = np.concatenate([self._S_alpha_covars,self._S_beta_covars[self._temp_cond,:,:,:], self._S_gamma_covars[trial_i,:,:,:]])

        self._K_numb_latents = self._K_alpha + self._K_beta + self._K_gamma
        
        #print(self._m_induc_mean_all.shape, self._S_covars_all.shape, self._K_numb_latents)

        # create theta:
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)

        theta = np.zeros(self._K_numb_latents * numb_param_per_latent)

        for lat_k in range(0,self._K_numb_latents):
            start_index = lat_k * numb_param_per_latent

            # the means:
            theta[start_index:start_index+M] = self._m_induc_mean_all[lat_k,:]

            S_temp = self._S_covars_all[lat_k,:,:].copy()
            temp1 = np.repeat(self._m_induc_mean_all[lat_k,:].reshape(M,1), M, 1)
            # S_temp += np.multiply(temp1, np.transpose(temp1))
            # print(S_temp)
            S_temp += np.outer(self._m_induc_mean_all[lat_k,:], self._m_induc_mean_all[lat_k,:])

            # print(S_temp)

            for i in range(0,M):
                for j in range(i,M):
                    place = i * M + j - ((i * (i + 1)) // 2)
                    #print(place==place2, i,j,place)
                    theta[start_index+M+place] = S_temp[i,j]


        ###
        # code to check that autograd does its work as it should:
        # self._Lm(theta)
        # gradient_func = grad(self._Lm)

        # gradients_tau = gradient_func(theta)

        # print(gradients_tau)

        # print("finite diff: ")
        # print(self._finite_diff_of_Lm(theta))

        ###
        # get the Hessian
        hessian_func = hessian(self._Lm)

        H = hessian_func(theta)

        H = np.squeeze(H) 

        # get V:
        V = self._get_V(theta)

        # print(H.shape)
        # print(V.shape)

        # print(H)
        # print(self._S_covars[self._temp_trial_i,:,:,:])
        # print(V)
        

        # code to check that autograd does its work as it should:
        # print("finite diff: ")
        # H_fine = self._finite_diff_H_of_Lm(theta)
        # # print(H_fine)
        # # print((H_fine != 0).astype(int))

        # H = H_fine

        ################
        # get Sigma
        
        temp1 = np.eye(self._K_numb_latents * numb_param_per_latent) - np.dot(V,H) 
        temp = np.linalg.inv(temp1) 
        Sigma_new = np.dot( temp , V )

        # print("V: ")
        # print((V != 0).astype(int))
        # print("H: ")
        # print((H != 0).astype(int))
        # print("I - VH: ")
        # print((temp1 != 0).astype(int))
        # print("inv(I - VH): ")
        # print((temp != 0).astype(int))
        # print("Sigma_new: ")
        # print((Sigma_new != 0).astype(int))

        # print("Sigma_new - V: ")
        # print((Sigma_new - V != 0).astype(int))

        ################
        # reshapes things in order to get Sigma

        S_hat, m = self._get_S_m_from_Sigma_new(Sigma_new, M)

        # print(S_hat)
        
        self._update_C_matrix(S_hat)

        return S_hat, m



    def _update_C_matrix(self, S_hat):
        """
        Using the matrix to update things:
        """

        C1_sum_term = 0
        C2_sum_term = 0

        for bin_i in range(0,self._numb_bins):

            # get K_ii:
            K_tilde_ii = None
            for lat_k in range(0,self._K_alpha):
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param_alpha, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param_alpha, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii + k_iZ.dot(self._inv_kernel_mat_induc_alpha[lat_k,:,:]).dot(k_Zi)

                if K_tilde_ii is None:
                    K_tilde_ii = temp
                else:
                    K_tilde_ii = block_diag(K_tilde_ii , temp)

            for lat_k in range(0,self._K_beta):
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param_beta, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param_beta, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii + k_iZ.dot(self._inv_kernel_mat_induc_beta[lat_k,:,:]).dot(k_Zi)

                if K_tilde_ii is None:
                    K_tilde_ii = temp
                else:
                    K_tilde_ii = block_diag(K_tilde_ii , temp)

            for lat_k in range(0,self._K_gamma):
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param_gamma, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param_gamma, lat_k)
                k_Zi = np.transpose(k_iZ)

                temp = k_ii + k_iZ.dot(self._inv_kernel_mat_induc_gamma[lat_k,:,:]).dot(k_Zi)

                if K_tilde_ii is None:
                    K_tilde_ii = temp
                else:
                    K_tilde_ii = block_diag(K_tilde_ii , temp)


            correcting_term = np.matmul( np.matmul(self._temp_param_K_i_list[bin_i], S_hat), np.transpose(self._temp_param_K_i_list[bin_i]) )

            # print( (correcting_term != 0).astype(int) )
            # print( correcting_term )
            #print( (S_hat != 0).astype(int) )
            
            # now the trial sum:
            for trial_i in range(0, self._numb_trials):

                this_cond = self._trial_conditions[trial_i]

                # print(self._m_alpha_induc_mean.shape)
                # print(self._m_beta_induc_mean[this_cond,:,:].shape)
                # print(self._m_gamma_induc_mean[trial_i,:,:].shape)
                m_induc_mean_all_loc = np.concatenate([self._m_alpha_induc_mean,self._m_beta_induc_mean[this_cond,:,:], self._m_gamma_induc_mean[trial_i,:,:]])
                m_induc_mean_all_loc = m_induc_mean_all_loc.flatten()

                # get m_tilde i r
                m_tilde_i_r = self._temp_param_K_i_list[bin_i].dot(m_induc_mean_all_loc)

                # update sum terms:
                # C:
                C1_sum_term += np.outer( (self._dataset[trial_i,:,bin_i] - self._d_bias) , m_tilde_i_r )
                C2_sum_term += K_tilde_ii + np.outer(m_tilde_i_r, m_tilde_i_r) + correcting_term


        self._C_matrix_big = C1_sum_term.dot(np.linalg.inv(C2_sum_term))


    def _get_S_m_from_Sigma_new(self, Sigma_new, M):

        S = np.zeros((self._K_numb_latents * M, self._K_numb_latents * M))
        m = np.zeros((self._K_numb_latents * M))
        
        len_block = M + ((M * (M + 1)) // 2)

        for lat_k_i in range(0, self._K_numb_latents):
            for lat_k_j in range(0, self._K_numb_latents):
                S[lat_k_i*M:lat_k_i*M+M, lat_k_j*M:lat_k_j*M+M] = Sigma_new[lat_k_i*len_block:lat_k_i*len_block+M, lat_k_j*len_block:lat_k_j*len_block+M]
                
        # print(self._S_covars[self._temp_trial_i,:,:,:])
        # print("")
        # print(S)


        for lat_k in range(0, self._K_numb_latents):
            for i in range(0, M):
                
                place_i = self._get_place_from_i_j(i,i) + M

                e_ii = Sigma_new[len_block*lat_k + i, len_block*lat_k + i]
                e_iii = Sigma_new[len_block*lat_k + i, len_block*lat_k + place_i]
                
                m[lat_k*M + i] = (e_iii / (2 * e_ii))


        ##### just seeing if S_hat give a higher uncertainty ####
        # for lat_k in range(0, self._K_numb_latents):
        #     diff = np.abs(S[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]) - np.abs(self._S_covars[self._temp_trial_i,lat_k,:,:])
        #     print( (diff > 0).astype(int) )
        #     print(S[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M])
        #     print(self._S_covars[self._temp_trial_i,lat_k,:,:])
        # ###########

        # print("")
        # print(self._m_induc_mean[self._temp_trial_i,:,:])
        # print(m)
        # print("")

        return S, m


    def _get_place_from_i_j(self,i,j):

        return i * self._numb_inducing + j - ((i * (i + 1)) // 2)

    def _get_i_j_from_place(self,place):

        M = self._numb_inducing
        det = (M - 1) * (M - 1) - 8 * (0)



    def _get_V(self,theta):
        """
        Returns the V matrix.
        """

        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)


        # V is block diagonal, so we build it block after block
        for lat_k in range(0,self._K_numb_latents):
            E_the_the_k = np.zeros((numb_param_per_latent,numb_param_per_latent))

            # improving readability:
            S = self._S_covars_all[lat_k,:,:]
            m_k = self._m_induc_mean_all[lat_k,:]

            # above square:
            for i in range(0,M):
                for j in range(i,M):
                    val = S[i,j] + m_k[i] * m_k[j]
                    E_the_the_k[i,j] = val
                    E_the_the_k[j,i] = val

            # two side blocks that are not on the diagonal:
            for l in range(0,M):
                for i in range(0,M):
                    for j in range(i,M):
                        place = self._get_place_from_i_j(i,j) + M

                        val = m_k[i] * m_k[j] * m_k[l]
                        val += S[i,j] * m_k[l]
                        val += S[i,l] * m_k[j]
                        val += S[l,j] * m_k[i]

                        E_the_the_k[l,place] = val
                        E_the_the_k[place,l] = val


            # big square block  in the bottom right corner:
            for l in range(0,M):
                for o in range(l,M):
                    for i in range(0,M):
                        for j in range(i,M):
                            place_1 = self._get_place_from_i_j(i,j) + M
                            place_2 = self._get_place_from_i_j(l,o) + M

                            val = m_k[i] * m_k[j] * m_k[l] * m_k[o]
                            val += S[i,j] * m_k[l] * m_k[o] + S[i,l] * m_k[j] * m_k[o] + S[l,j] * m_k[i] * m_k[o]
                            val += S[i,o] * m_k[l] * m_k[j] + S[o,l] * m_k[j] * m_k[i] + S[o,j] * m_k[i] * m_k[l]
                            val += S[i,o] * S[j,l] + S[i,l] * S[j,o] + S[i,j] * S[o,l]

                            E_the_the_k[place_1,place_2] = val
                            E_the_the_k[place_2,place_1] = val

            # print(E_the_the_k)

            E_the_k = theta[lat_k * numb_param_per_latent:(lat_k+1) * numb_param_per_latent]
            V_k = E_the_the_k - np.outer(E_the_k, E_the_k)

            # print(E_the_k)

            # print(V_k)

            if lat_k == 0:
                V = V_k
            else:
                V = block_diag(V, V_k)

        return V



    def _Lm(self,theta):
        """
        This is the short implementation of it where all the elements that are not product of variables are deleted.

        """
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)

        ################# 
        # the first step is to get the latent parameters m and S from the vector of mean parameters
        m_induc_mean = np.zeros((self._K_numb_latents, self._numb_inducing))
        S_covars = np.zeros((self._K_numb_latents, self._numb_inducing, self._numb_inducing))

        e_unit_vec_tilde = np.eye(numb_param_per_latent)
        e_unit_vec = np.eye(M)

        m_induc_mean_list = []
        S_covar_list = []

        for lat_k in range(0,self._K_numb_latents):
            start_index = lat_k * numb_param_per_latent

            # the means:
            # m_induc_mean[lat_k,:] = theta[start_index:start_index+M] 
            m_induc_mean_k = theta[start_index:start_index+M]
            m_induc_mean_list.append(m_induc_mean_k)

            S = np.zeros((self._numb_inducing, self._numb_inducing))

            for i in range(0,M):
                for j in range(i,M):
                    place = i * M + j - ((i * (i + 1)) // 2)
                    #print(place==place2, i,j,place)
                    S += np.outer(e_unit_vec[i,:], e_unit_vec[j,:]) * theta[start_index+M+place]

                    if i != j:
                        S += np.outer(e_unit_vec[j,:], e_unit_vec[i,:]) * theta[start_index+M+place]

            #temp1 = np.repeat(m_induc_mean_k.reshape(M,1), M, 1)
            # S -= np.multiply(temp1, np.transpose(temp1))
            S -= np.outer(m_induc_mean_k, m_induc_mean_k)

            S_covar_list.append(S)

        ###########
        # then compute Lm using these

        # some precomputations:
        if self._force_R_diag:
            Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise) + SMALL)

        CRinv = np.matmul( np.transpose(self._C_matrix_big), Rinv )
        CRinvC = np.matmul( CRinv, self._C_matrix_big )

        Lm = 0

        first_term = 0
        second_term = 0

        for bin_i in range(0, self._numb_bins):

            for lat_k in range(0, self._K_numb_latents):

                temp_1 = np.matmul( self._temp_param_K_i_list[bin_i][:,lat_k*self._numb_inducing:(lat_k+1)*self._numb_inducing], m_induc_mean_list[lat_k] )

                for lat_k_2 in range(0, self._K_numb_latents):

                    temp_2 = np.matmul( self._temp_param_K_i_list[bin_i][:,lat_k_2*self._numb_inducing:(lat_k_2+1)*self._numb_inducing], m_induc_mean_list[lat_k_2] )

                    first_term += - 0.5 * np.matmul( np.matmul(temp_1.transpose() , CRinvC ) , temp_2 )


        for lat_k in range(0, self._K_alpha):
            second_term += 0.5 * np.matmul( np.matmul(m_induc_mean_list[lat_k].transpose() , self._inv_kernel_mat_induc_alpha[lat_k] ) , m_induc_mean_list[lat_k] )

        for lat_k in range(0, self._K_beta):
            lat_2 = lat_k + self._K_alpha
            second_term += 0.5 * np.matmul( np.matmul(m_induc_mean_list[lat_2].transpose() , self._inv_kernel_mat_induc_beta[lat_k] ) , m_induc_mean_list[lat_2] )

        for lat_k in range(0, self._K_gamma):
            lat_2 = lat_k + self._K_alpha + self._K_beta
            second_term += 0.5 * np.matmul( np.matmul(m_induc_mean_list[lat_2].transpose() , self._inv_kernel_mat_induc_gamma[lat_k] ) , m_induc_mean_list[lat_2] )


        Lm = first_term + second_term

        return Lm



    def _Lm_sv_long(self,theta):
        """
        This is the correct implementation of it.

        So bits could be deleted as we are actually only interested in the derivative of this.

        """
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)

        ################# 
        # the first step is to get the latent parameters m and S from the vector of mean parameters
        m_induc_mean = np.zeros((self._K_numb_latents, self._numb_inducing))
        S_covars = np.zeros((self._K_numb_latents, self._numb_inducing, self._numb_inducing))

        e_unit_vec_tilde = np.eye(numb_param_per_latent)
        e_unit_vec = np.eye(M)

        m_induc_mean_list = []
        S_covar_list = []

        for lat_k in range(0,self._K_numb_latents):
            start_index = lat_k * numb_param_per_latent

            # the means:
            # m_induc_mean[lat_k,:] = theta[start_index:start_index+M] 
            m_induc_mean_k = theta[start_index:start_index+M]
            m_induc_mean_list.append(m_induc_mean_k)

            S = np.zeros((self._numb_inducing, self._numb_inducing))

            for i in range(0,M):
                for j in range(i,M):
                    place = i * M + j - ((i * (i + 1)) // 2)
                    #print(place==place2, i,j,place)
                    S += np.outer(e_unit_vec[i,:], e_unit_vec[j,:]) * theta[start_index+M+place]

                    if i != j:
                        S += np.outer(e_unit_vec[j,:], e_unit_vec[i,:]) * theta[start_index+M+place]

            #temp1 = np.repeat(m_induc_mean_k.reshape(M,1), M, 1)
            # S -= np.multiply(temp1, np.transpose(temp1))
            S -= np.outer(m_induc_mean_k, m_induc_mean_k)

            S_covar_list.append(S)

        ###########
        # then compute Lm using these

        Lm = 0

        first_term = 0
        second_term = 0
        third_term = 0

        # some precomputations:
        if self._force_R_diag:
            Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise) + SMALL)

        CRinv = np.matmul( np.transpose(self._C_matrix), Rinv )
        CRinvC = np.matmul( CRinv, self._C_matrix )

        # second term :
        inv_Kzz_list = []
        sum_over_k_second = 0
        for lat_k in range(0,self._K_numb_latents):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            inv_Kzz = np.linalg.inv(K_zz)
            inv_Kzz_list.append(inv_Kzz)

            expr = np.trace( np.matmul(inv_Kzz , S_covar_list[lat_k]) )

            expr += np.matmul( np.matmul( np.transpose( m_induc_mean_list[lat_k]  ), inv_Kzz ), m_induc_mean_list[lat_k] )

            expr += np.log( np.linalg.det(K_zz) + SMALL )

            sum_over_k_second += 0.5 * expr

        # first term:
        first_term += - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            third_term_temp_kappa = 0
            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):
                inv_Kzz = inv_Kzz_list[lat_k]

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz , S_covar_list[lat_k]), inv_Kzz )
                matr_term -= inv_Kzz

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp

                temp = k_ii + np.matmul( np.matmul( k_iZ, inv_Kzz), k_Zi)
                third_term_temp_kappa += np.outer( np.outer( np.transpose(self._C_matrix[:,lat_k]), temp), self._C_matrix[:,lat_k] )
            #print(third_term_temp_kappa.shape)

            third_term += 0.5 *  np.log(np.linalg.det(third_term_temp_kappa) + SMALL)


            # for trial_i in range(0,self._numb_trials):

            # forming m tilde:
            m_tilde = np.zeros((self._K_numb_latents))

            e_unit_vect = np.eye(self._K_numb_latents)

            for lat_k in range(0,self._K_numb_latents):
                inv_Kzz = inv_Kzz_list[lat_k]
                K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                m_tilde += e_unit_vect[lat_k,:] * np.dot(np.matmul(K_new,inv_Kzz), m_induc_mean_list[lat_k] )

            first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[self._temp_trial_i,:,bin_i]) , Rinv), self._dataset[self._temp_trial_i,:,bin_i])

            # combining the two m K_i C terms:
            temp = np.transpose(m_tilde)
            first_term += np.matmul( temp, np.matmul( CRinv, (self._dataset[self._temp_trial_i,:,bin_i] - self._d_bias   )  ) )

            # combining the two d terms:
            first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[self._temp_trial_i,:,bin_i] - 0.5 * self._d_bias ) )

            # the big term:
            temp = m_tilde
            first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

            first_term += - 0.5 * sum_over_k_first

        # third term (end):
        third_term += (self._numb_neurons / 2) * (np.log(2 * np.pi) + 1) * self._numb_bins
            
        Lm = first_term + second_term + third_term
        return Lm




    def _finite_diff_of_Lm(self,theta):

        derivate = np.zeros_like(theta)

        Lm_theta = self._Lm(theta)

        for var_i, var in enumerate(theta):
            theta_plus = theta.copy()
            theta_plus[var_i] += SMALL
            # theta_minus[var_i] -= SMALL

            derivate[var_i] = (self._Lm(theta_plus) - Lm_theta) / SMALL

        return derivate




    def _finite_diff_H_of_Lm(self,theta):


        hessian_H = np.zeros((len(theta),len(theta)))

        e_uni = np.eye(len(theta))

        Lm_theta = self._Lm(theta)
        Lm_theta_list = []

        for i in range(0,len(theta)):
            Lm_theta_list.append(self._Lm(theta + SMALL * e_uni[i,:]))       

        for i in range(0,len(theta)):
            for j in range(0,len(theta)):

                eval_1 = self._Lm(theta + SMALL * e_uni[i,:] + SMALL * e_uni[j,:])
                hessian_H[i,j] = (eval_1 - Lm_theta_list[i] - Lm_theta_list[j] + Lm_theta) / (SMALL * SMALL)

        return hessian_H
