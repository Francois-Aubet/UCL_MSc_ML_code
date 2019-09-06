"""

The linear response improvement of the sparce variational approximation of the Gaussian Process Factor Analysis (GPFA) model.

This is just an extension of the GPFA_sv class where add the method to compute the linear response correction of
    the covariance function.




TODO here:




"""

from algorithms.GPFA_sv import GPFA_sv

# from algorithms.GPFA import GPFA
# from AlgoHandler import AlgoHandler

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

from scipy.linalg import polar 

SMALL = 1e-13


# np.set_printoptions(edgeitems=30, linewidth=100000, 
#     formatter=dict(float=lambda x: "%.5g" % x))

# np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)




class GPFA_sv_lr(GPFA_sv):
    """ sparce variational GPFA 

    Organisation:
    - Linear response correction
    - Update of the other parameters, change in E and M steps
    - Lm computation
    - Measurements methods, assessing the estimate
    - Finite difference code


    
    """

    algName = "lr_sv_GPFA"


    def __init__(self, dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step ):
        """ Preparation of all the needed variables. """

        super().__init__(dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step)

        self._use_rearanged_H_and_V = True

        self._schur_zerlegung = True



    #############################   Linear response correction  ################################

    def linear_response_correction(self):
        """
        """

        trial_i = 0
        S_hat, m = self._linear_response_correction(trial_i)
        
        stats_on_Sm = self._compare_with_previous(S_hat, m)

        # self._compare_S_with_generating()
        # self._compute_mean_off_diag_S()

        # self._E_step()
        

        #### find the the next closest one:
        if self._use_gradient_E_steps:

            A = S_hat.copy()

            B = (A + A.transpose()) / 2

            # version 1:
            # U, H = polar(A)
            # X_f = (B + H) / 2

            # print("distance between the two: ", fnorm(X_f - self._S_full_matr), fnorm(X_f) - fnorm(self._S_full_matr))
            # print("after correction, is psd? : ",is_pos_def(X_f))

            # version 2:
            D, U = np.linalg.eig(A)
            D[D < 0] = 0

            S_hat = U.dot(np.diag(D)).dot(U.transpose())

            # print("distance between the two: ", fnorm(X_f - self._S_full_matr), fnorm(X_f) - fnorm(self._S_full_matr))
            # print("after correction, is psd? : ",is_pos_def(X_f))
            


        # print(is_pos_def(S_hat))

        self._S_full_matr = S_hat.copy()

        M = self._numb_inducing
        for lat_k in range(0, self._K_numb_latents):
            for trial_i in range(0, self._K_numb_latents):
                self._S_covars[trial_i,lat_k,:,:] = S_hat[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]

        # self._compare_S_with_generating()
        # self._compute_mean_off_diag_S()

        return stats_on_Sm



    def _linear_response_correction(self,trial_i):
        """
        Correcting the covariance between the q(u) using 

        """
        self._temp_trial_i = trial_i

        ################
        # get V and H

        # create theta:
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)

        theta = self._create_theta_for_Lm(self._temp_trial_i)

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
        H = self._get_H(theta) 

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
        
        # temp1 = np.eye(self._K_numb_latents * numb_param_per_latent) - np.dot(V,H) 
        temp1 = np.eye(H.shape[0]) - np.dot(V,H) 
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

        # print((Sigma_new - V).astype(int))

        ################
        # reshapes things in order to get Sigma

        S_hat, m = self._get_S_m_from_Sigma_new(Sigma_new, M)

        # print(S_hat)

        return S_hat, m


    def _get_H(self, theta):
        """
        Get the Hessian, different methods possible for that.
        """

        method = 3
        # 0 = Vanilla, 1 = short, 2 = short + optimised, 3 = very higly optimized

        start = time.time()
        # Vanilla method:
        if method == 0:
            hessian_func = hessian(self._Lm)

            H = hessian_func(theta)
            H = np.squeeze(H)

        elif method == 1:
            hessian_func = hessian(self._Lm_short_non_opt)

            H = hessian_func(theta)
            H = np.squeeze(H)

        elif method == 2:
            theta = self._create_theta_short_for_Lm_opt(self._temp_trial_i)

            hessian_func = hessian(self._Lm_short_opt)

            H = hessian_func(theta)
            H = np.squeeze(H)

            # unpacking the Hessian
            M = self._numb_inducing # just making things clearer to read here
            numb_param_per_latent = M + ((M * (M + 1)) // 2)

            H_true = np.zeros((self._K_numb_latents * numb_param_per_latent, self._K_numb_latents * numb_param_per_latent))

            for lat_k1 in range(0,self._K_numb_latents):
                for lat_k2 in range(0,self._K_numb_latents):
                    start1 = numb_param_per_latent * lat_k1
                    start2 = numb_param_per_latent * lat_k2

                    H_true[start1:start1+M, start2:start2+M] = H[M*lat_k1:M*lat_k1+M, M*lat_k2:M*lat_k2+M]

            # print(H)

            H = H_true


        elif method == 3:
            
            common_Lm_term = 0

            test_term = 0

            Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

            for bin_i in range(0, self._numb_bins):

                common_Lm_term += - 0.5 * self._temp_param_K_i_list[bin_i].transpose().dot(CRinvC).dot(self._temp_param_K_i_list[bin_i])
            #     common_Lm_term += - 0.5 * self._temp_param_K_i_list[bin_i].transpose().dot(np.diag(np.diag(CRinvC))).dot(self._temp_param_K_i_list[bin_i])
            # common_Lm_term += 0.5 * self._temp_param_inv_K_tilde

            for lat_k in range(0, self._K_numb_latents):
                process_term = 0
                for bin_i in range(0, self._numb_bins):
                    kappa = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                    process_term += 0.5 * CRinvC[lat_k,lat_k] * self._inv_kernel_mat_induc[lat_k,:,:].dot(kappa.transpose()).dot(kappa).dot(self._inv_kernel_mat_induc[lat_k,:,:])
                # process_term += 0.5 * self._inv_kernel_mat_induc[lat_k,:,:]
                common_Lm_term[self._numb_inducing*lat_k:self._numb_inducing*(lat_k+1), self._numb_inducing*lat_k:self._numb_inducing*(lat_k+1)] += process_term


                #process_Lm_term_list.append(process_term)

            # print((common_Lm_term == 0).astype(int))
            # print((test_term == 0).astype(int))
            # print((test_term - common_Lm_term == 0).astype(int))
            # print(np.mean(np.abs(test_term - common_Lm_term)))
            # print(np.mean(np.abs(test_term)), np.mean(np.abs(common_Lm_term)))

            side_len = self._K_numb_latents * self._numb_inducing
            H = common_Lm_term * 2#np.zeros((side_len, side_len))
            # for i in range(0, side_len):
            #     for j in range(0, side_len):

            #         H[i,j] = common_Lm_term[i,j] * 2#+ common_Lm_term[j,i]

            #H_true = np.zeros((self._K_numb_latents * numb_param_per_latent, self._K_numb_latents * numb_param_per_latent))

            M = self._numb_inducing # just making things clearer to read here
            numb_param_per_latent = M + ((M * (M + 1)) // 2)
            H_true = np.zeros((self._K_numb_latents * numb_param_per_latent, self._K_numb_latents * numb_param_per_latent))

            for lat_k1 in range(0,self._K_numb_latents):
                for lat_k2 in range(0,self._K_numb_latents):
                    start1 = numb_param_per_latent * lat_k1
                    start2 = numb_param_per_latent * lat_k2

                    H_true[start1:start1+M, start2:start2+M] = H[M*lat_k1:M*lat_k1+M, M*lat_k2:M*lat_k2+M]

            # print(H)

            H = H_true
        end = time.time()

        if self._use_rearanged_H_and_V:
            H = self._get_rearanged(H)
            #print("changing H!!!")

        # print("Time needed for the Hessian: ",end - start)

        return H


    def _get_rearanged(self, matrix):
        """
        """

        rea_matrix = np.zeros_like(matrix)


        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)
        numb_V = ((M * (M + 1)) // 2)

        res_mat_uu = np.zeros((M*self._K_numb_latents, M*self._K_numb_latents))
        res_mat_uV = np.zeros((M*self._K_numb_latents, numb_V*self._K_numb_latents))
        res_mat_VV = np.zeros((numb_V*self._K_numb_latents, numb_V*self._K_numb_latents))

        for lat_k1 in range(0,self._K_numb_latents):
            for lat_k2 in range(0,self._K_numb_latents):
                start1 = numb_param_per_latent * lat_k1
                start2 = numb_param_per_latent * lat_k2

                res_mat_uu[M*lat_k1:M*(lat_k1+1), M*lat_k2:M*(lat_k2+1)] = matrix[start1:start1+M, start2:start2+M]

                start1 = numb_param_per_latent * lat_k1
                start2 = numb_param_per_latent * lat_k2 + M

                res_mat_uV[M*lat_k1:M*(lat_k1+1), numb_V*lat_k2:numb_V*(lat_k2+1)] = matrix[start1:start1+M, start2:start2+numb_V]

                start1 = numb_param_per_latent * lat_k1 + M
                start2 = numb_param_per_latent * lat_k2 + M

                res_mat_VV[numb_V*lat_k1:numb_V*(lat_k1+1), numb_V*lat_k2:numb_V*(lat_k2+1)] = matrix[start1:start1+numb_V, start2:start2+numb_V]

        rea_matrix[0:self._K_numb_latents*M, 0:self._K_numb_latents*M] = res_mat_uu
        rea_matrix[self._K_numb_latents*M:, 0:self._K_numb_latents*M] = res_mat_uV.transpose()
        rea_matrix[0:self._K_numb_latents*M, self._K_numb_latents*M:] = res_mat_uV
        rea_matrix[self._K_numb_latents*M:, self._K_numb_latents*M:] = res_mat_VV


        if self._schur_zerlegung:
            return res_mat_uu

        return rea_matrix
        




    def _get_S_m_from_Sigma_new(self, Sigma_new, M):

        if self._schur_zerlegung:
            S = Sigma_new
            m = np.zeros((self._K_numb_latents * M))
            return S, m

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


        if self._use_rearanged_H_and_V:
            S =Sigma_new[0:self._K_numb_latents * M, 0:self._K_numb_latents * M]


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
            S = self._S_covars[self._temp_trial_i,lat_k,:,:]
            m_k = self._m_induc_mean[self._temp_trial_i,lat_k,:]

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

        if self._use_rearanged_H_and_V:
            V = self._get_rearanged(V)

        return V


    def _get_V_full(self,theta):
        """
        Returns the V matrix.
        """

        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)

        V = np.zeros((self._K_numb_latents * numb_param_per_latent, self._K_numb_latents * numb_param_per_latent))


        # V is block diagonal, so we build it block after block
        for lat_k1 in range(0,self._K_numb_latents):
            for lat_k2 in range(0,self._K_numb_latents):
                E_the_the_k = np.zeros((numb_param_per_latent,numb_param_per_latent))

                # improving readability:
                S = self._S_full_matr[M*lat_k1:M*(lat_k1+1),M*lat_k2:M*(lat_k2+1)] #[self._temp_trial_i,lat_k,:,:]
                m_k1 = self._m_induc_mean[self._temp_trial_i,lat_k1,:]
                m_k2 = self._m_induc_mean[self._temp_trial_i,lat_k2,:]

                # above square:
                for i in range(0,M):
                    for j in range(i,M):
                        val = S[i,j] + m_k1[i] * m_k2[j]
                        E_the_the_k[i,j] = val
                        E_the_the_k[j,i] = val

                # two side blocks that are not on the diagonal:
                for l in range(0,M):
                    for i in range(0,M):
                        for j in range(i,M):
                            place = self._get_place_from_i_j(i,j) + M

                            val = m_k2[i] * m_k2[j] * m_k1[l]
                            val += S[i,j] * m_k1[l]
                            val += S[i,l] * m_k2[j]
                            val += S[l,j] * m_k2[i]

                            E_the_the_k[l,place] = val

                for l in range(0,M):
                    for i in range(0,M):
                        for j in range(i,M):
                            place = self._get_place_from_i_j(i,j) + M

                            val = m_k1[i] * m_k1[j] * m_k2[l]
                            val += S[i,j] * m_k2[l]
                            val += S[i,l] * m_k1[j]
                            val += S[l,j] * m_k1[i]
                        
                            E_the_the_k[place,l] = val


                # big square block  in the bottom right corner:
                for l in range(0,M):
                    for o in range(l,M):
                        for i in range(0,M):
                            for j in range(i,M):
                                place_1 = self._get_place_from_i_j(i,j) + M
                                place_2 = self._get_place_from_i_j(l,o) + M

                                val = m_k1[i] * m_k1[j] * m_k2[l] * m_k2[o]
                                val += S[i,j] * m_k2[l] * m_k2[o] + S[i,l] * m_k1[j] * m_k2[o] + S[l,j] * m_k1[i] * m_k2[o]
                                val += S[i,o] * m_k2[l] * m_k1[j] + S[o,l] * m_k1[j] * m_k1[i] + S[o,j] * m_k1[i] * m_k2[l]
                                val += S[i,o] * S[j,l] + S[i,l] * S[j,o] + S[i,j] * S[o,l]

                                E_the_the_k[place_1,place_2] = val

                for l in range(0,M):
                    for o in range(l,M):
                        for i in range(0,M):
                            for j in range(i,M):
                                place_1 = self._get_place_from_i_j(i,j) + M
                                place_2 = self._get_place_from_i_j(l,o) + M

                                val = m_k2[i] * m_k2[j] * m_k1[l] * m_k1[o]
                                val += S[i,j] * m_k1[l] * m_k1[o] + S[i,l] * m_k2[j] * m_k1[o] + S[l,j] * m_k2[i] * m_k1[o]
                                val += S[i,o] * m_k1[l] * m_k2[j] + S[o,l] * m_k2[j] * m_k2[i] + S[o,j] * m_k2[i] * m_k1[l]
                                val += S[i,o] * S[j,l] + S[i,l] * S[j,o] + S[i,j] * S[o,l]

                                E_the_the_k[place_2,place_1] = val

                # print(E_the_the_k)

                E_the_k1 = theta[lat_k1 * numb_param_per_latent:(lat_k1+1) * numb_param_per_latent]
                E_the_k2 = theta[lat_k2 * numb_param_per_latent:(lat_k2+1) * numb_param_per_latent]
                V_k = E_the_the_k - np.outer(E_the_k1, E_the_k2)

                V[lat_k1 * numb_param_per_latent:(lat_k1+1) * numb_param_per_latent, lat_k2 * numb_param_per_latent:(lat_k2+1) * numb_param_per_latent] = V_k



        if self._use_rearanged_H_and_V:
            V = self._get_rearanged(V)

        return V




    #############################   Update of the other parameters, change in E and M steps  ################################

    def _E_step_lr(self):
        """
        A modified E step, where we do the linear response correction:

        """

        if self._use_gradient_E_steps:
            # reset the off diagonal elements:
            self._S_full_matr = np.zeros((self._K_numb_latents*self._numb_inducing, self._K_numb_latents*self._numb_inducing))
            self._set_S_full_from_Sk()

            self._E_step_grad_mf()
        else:
            self._E_step_mf()
            
        # print("before LR, is psd? : ",is_pos_def(self._S_full_matr))

        self.linear_response_correction()

        # print("after LR, is psd? : ",is_pos_def(self._S_full_matr))





    def E_step_update_m_only(self):
        """
        """
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


    def _M_step_lr(self):
        """
        The M step of the EM algorithm to fit the model using the full covariance.

        """
        self.update_model_parameters_after_LR()


    def update_model_parameters_after_LR(self, method = 1):
        """
        Using the matrix to update things:
        """

        self._M_step_full()






    #############################   Lm computation  ################################

    def _Lm_short_opt(self,theta_short):
        """
        This is the short implementation of Lm, containing only the terms of which the second derivative is not zero.
        Plus this is optimized to use only a small theta, just the usefull, to have a way smaller matrix.

        """
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M #+ ((M * (M + 1)) // 2)

        ################# 
        # the first step is to get the latent parameters m and S from the vector of mean parameters

        m_induc_mean_list = []
        S_covar_list = []

        for lat_k in range(0,self._K_numb_latents):
            start_index = lat_k * numb_param_per_latent
            # the means:
            m_induc_mean_k = theta_short[start_index:start_index+M]
            m_induc_mean_list.append(m_induc_mean_k)

            S = np.zeros((self._numb_inducing, self._numb_inducing))

            S -= np.outer(m_induc_mean_k, m_induc_mean_k)

            S_covar_list.append(S)

        ###########
        # then compute Lm using these
        Lm = 0

        first_term = 0
        second_term = 0
        third_term = 0

        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        # second term :
        # inv_Kzz_list = []
        # sum_over_k_second = 0
        # for lat_k in range(0,self._K_numb_latents):
        #     K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
        #     inv_Kzz = np.linalg.inv(K_zz)
        #     inv_Kzz_list.append(inv_Kzz)

        #     expr = np.trace( np.matmul(inv_Kzz , S_covar_list[lat_k]) )

        #     expr += np.matmul( np.matmul( np.transpose( m_induc_mean_list[lat_k]  ), inv_Kzz ), m_induc_mean_list[lat_k] )

        #     sum_over_k_second += 0.5 * expr
        #second_term = sum_over_k_second #* 2

        # first term:
        first_term += 0

        for bin_i in range(0,self._numb_bins):

            third_term_temp_kappa = 0
            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):
                inv_Kzz = inv_Kzz_list[lat_k]

                # k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz , S_covar_list[lat_k]), inv_Kzz ) # self._S_covars[0,lat_k,:,:]
                #matr_term -= inv_Kzz

                temp = np.matmul( np.matmul( k_iZ,matr_term), k_Zi) 

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp

            # forming m tilde:
            m_tilde = np.zeros((self._K_numb_latents))

            e_unit_vect = np.eye(self._K_numb_latents)

            for lat_k in range(0,self._K_numb_latents):
                inv_Kzz = inv_Kzz_list[lat_k]
                K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                m_tilde += e_unit_vect[lat_k,:] * np.dot(np.matmul(K_new,inv_Kzz), m_induc_mean_list[lat_k] )

            # the big term:
            temp = m_tilde
            first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

            first_term += - 0.5 * sum_over_k_first

        Lm = first_term + second_term

        return Lm


    def _Lm_short_non_opt(self,theta):
        """
        This is the short implementation of Lm, containing only the terms of which the second derivative is not zero.

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

            S -= np.outer(m_induc_mean_k, m_induc_mean_k)

            S_covar_list.append(S)

        ###########
        # then compute Lm using these
        Lm = 0

        first_term = 0
        second_term = 0
        third_term = 0

        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        # second term :
        inv_Kzz_list = []
        sum_over_k_second = 0
        for lat_k in range(0,self._K_numb_latents):
            K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            inv_Kzz = np.linalg.inv(K_zz)
            inv_Kzz_list.append(inv_Kzz)

            expr = np.matmul( np.matmul( np.transpose( m_induc_mean_list[lat_k]  ), inv_Kzz ), m_induc_mean_list[lat_k] )

            sum_over_k_second += 0.5 * expr

        # first term:
        first_term += 0

        for bin_i in range(0,self._numb_bins):

            third_term_temp_kappa = 0
            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):
                inv_Kzz = inv_Kzz_list[lat_k]

                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                matr_term = np.matmul( np.matmul( inv_Kzz , S_covar_list[lat_k]), inv_Kzz ) # self._S_covars[0,lat_k,:,:]
                #matr_term -= inv_Kzz

                temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp

            # forming m tilde:
            m_tilde = np.zeros((self._K_numb_latents))

            e_unit_vect = np.eye(self._K_numb_latents)

            for lat_k in range(0,self._K_numb_latents):
                inv_Kzz = inv_Kzz_list[lat_k]
                K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                m_tilde += e_unit_vect[lat_k,:] * np.dot(np.matmul(K_new,inv_Kzz), m_induc_mean_list[lat_k] )

            # combining the two m K_i C terms:
            # temp = np.transpose(m_tilde)
            # first_term += np.matmul( temp, np.matmul( CRinv, (self._dataset[self._temp_trial_i,:,bin_i] - self._d_bias   )  ) )

            # combining the two d terms:
            # first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[self._temp_trial_i,:,bin_i] - 0.5 * self._d_bias ) )

            # the big term:
            temp = m_tilde
            first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

            first_term += - 0.5 * sum_over_k_first

        # third term (end):
        #third_term += (self._numb_neurons / 2) * (np.log(2 * np.pi) + 1) * self._numb_bins
            
        Lm = first_term + second_term #+ third_term
        return Lm


    def _Lm(self,theta):
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

            expr += np.linalg.slogdet(K_zz)[1]#np.log( np.linalg.det(K_zz) + SMALL )

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

                temp = k_ii - np.matmul( np.matmul( k_iZ, inv_Kzz), k_Zi)
                third_term_temp_kappa += np.outer( np.outer( np.transpose(self._C_matrix[:,lat_k]), temp), self._C_matrix[:,lat_k] )
            #print(third_term_temp_kappa.shape)

            third_term += 0.5 *  np.linalg.slogdet(third_term_temp_kappa)[1]#np.log(np.linalg.det(third_term_temp_kappa) + SMALL)

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



    def _create_theta_short_for_Lm_opt(self, trial_i):
        """
        """
        
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M

        theta = np.zeros(self._K_numb_latents * numb_param_per_latent)

        for lat_k in range(0,self._K_numb_latents):
            start_index = lat_k * numb_param_per_latent

            # the means:
            theta[start_index:start_index+M] = self._m_induc_mean[trial_i,lat_k,:]


        return theta


    def _create_theta_for_Lm(self, trial_i):
        """
        """
        
        M = self._numb_inducing # just making things clearer to read here
        numb_param_per_latent = M + ((M * (M + 1)) // 2)

        theta = np.zeros(self._K_numb_latents * numb_param_per_latent)

        for lat_k in range(0,self._K_numb_latents):
            start_index = lat_k * numb_param_per_latent

            # the means:
            theta[start_index:start_index+M] = self._m_induc_mean[trial_i,lat_k,:]

            S_temp = self._S_covars[trial_i,lat_k,:,:].copy()
            temp1 = np.repeat(self._m_induc_mean[trial_i,lat_k,:].reshape(M,1), M, 1)
            # S_temp += np.multiply(temp1, np.transpose(temp1))
            # print(S_temp)
            S_temp += np.outer(self._m_induc_mean[trial_i,lat_k,:], self._m_induc_mean[trial_i,lat_k,:])

            # print(S_temp)

            for i in range(0,M):
                for j in range(i,M):
                    place = i * M + j - ((i * (i + 1)) // 2)
                    #print(place==place2, i,j,place)
                    theta[start_index+M+place] = S_temp[i,j]

        return theta

    def get_Lm_for_comparison_porpuses(self, return_as_string = False):

        Lm = 0
        Lm_full = 0

        for trial_i in range(0, self._numb_trials):
            self._temp_trial_i = trial_i
            theta = self._create_theta_for_Lm(trial_i)

            Lm_full += self._Lm_full_non_differentiable()
            Lm += self._Lm(theta)

        # return Lm[0,0], Lm_full[0,0]
        if return_as_string:
            return str(Lm[0,0]) + "," + str(Lm_full)
        else:
            return Lm[0,0], Lm_full


    def _Lm_full_non_differentiable(self):

        Lm = 0

        first_term = 0
        second_term = 0
        third_term = 0

        # some precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()
        
        # second term :
        expr = np.trace( np.matmul(self._temp_param_inv_K_tilde , self._S_full_matr) )

        expr += np.matmul( np.matmul( np.transpose( self._temp_param_m[self._temp_trial_i,:] ), self._temp_param_inv_K_tilde ), self._temp_param_m[self._temp_trial_i,:])

        if self._numb_inducing != len(self._bin_times):
            expr += np.log( np.linalg.det(self._temp_param_K_tilde) )
        else:
            expr += np.log( np.linalg.det(self._temp_param_K_tilde) + SMALL ) 

        second_term += 0.5 * expr

        # first term:
        
        first_term += - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

        for bin_i in range(0,self._numb_bins):

            third_term_temp_kappa = 0
            sum_over_k_first = 0
            for lat_k in range(0,self._K_numb_latents):
                k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                k_Zi = np.transpose(k_iZ)
                
                temp = k_ii - np.matmul( np.matmul( k_iZ, self._inv_kernel_mat_induc[lat_k,:,:]), k_Zi)

                sum_over_k_first += CRinvC[lat_k,lat_k] * temp

                third_term_temp_kappa += np.outer( np.outer( np.transpose(self._C_matrix[:,lat_k]), temp), self._C_matrix[:,lat_k] )
            #print(third_term_temp_kappa.shape)

            third_term += 0.5 *  np.linalg.slogdet(third_term_temp_kappa)[1]#np.log(np.linalg.det(third_term_temp_kappa) + SMALL)

            matr_term = np.matmul( np.matmul( self._temp_param_K_i_list[bin_i], self._S_full_matr ), self._temp_param_K_i_list[bin_i].transpose() )
            matr_term = np.matmul( np.matmul( self._C_matrix, matr_term), self._C_matrix.transpose() )

            sum_over_k_first = np.sum(self._S_full_matr)#np.trace( np.matmul(Rinv , (matr_term ) ) ) + sum_over_k_first

            # for trial_i in range(0,self._numb_trials):

            # forming m tilde:
            m_tilde = np.dot(self._temp_param_m[self._temp_trial_i,:], self._temp_param_K_i_list[bin_i].transpose())

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







    #############################   Measurements methods, assessing the estimate.  ################################

    def get_full_XX_covariance(self):
        """
        """

        for lat_k in range(0,self._K_numb_latents):
            K_new = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            K_prior = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
            if lat_k == 0:
                K_i = K_new
                K_i_prior = K_prior
            else:
                K_i = block_diag(K_i, K_new)
                K_i_prior = block_diag(K_i_prior, K_prior)

        matr_from_S = np.matmul( np.matmul(self._temp_param_inv_K_tilde , self._S_full_matr), self._temp_param_inv_K_tilde)
        matr_from_S = np.matmul( np.matmul(K_i , matr_from_S - self._temp_param_inv_K_tilde), K_i.transpose())
        matr_from_S = K_i_prior + matr_from_S

        return matr_from_S




    def compare_S_with_GPFA(self, verbose = True):
        """
        Compares the newly found parameters with the generating one.

        """        

        # big comparion for off block diag elements not available yet

        mean_abs_true = np.mean(np.abs(self._gpfa_covariance))
        mean_abs_found = np.mean(np.abs(self._S_covars))

        # create the mapping of S into bin:
        matr_from_S = self.get_full_XX_covariance()

        sum_diff = np.sum(np.abs(self._gpfa_covariance - matr_from_S))


        # block by block comparison
        sum_gpfa_diag = 0
        sum_gpfa_total = np.sum(np.abs(self._gpfa_covariance))

        # get the diagonal terms
        M = self._numb_inducing
        T = self._numb_bins
        sum_diag = 0
        for lat_k in range(0, self._K_numb_latents):
            # kappa_mat = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
            # kappa_invK = kappa_mat.dot(self._inv_kernel_mat_induc[lat_k,:,:])
            # matr_from_S = np.matmul( np.matmul(kappa_invK , self._S_full_matr[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]), kappa_invK.transpose())

            # sum_diag += np.sum(np.abs(self._gpfa_covariance[:,:,lat_k]  - matr_from_S))
            sum_diag += np.sum(np.abs(self._gpfa_covariance[lat_k*T:lat_k*T+T, lat_k*T:lat_k*T+T]  - matr_from_S[lat_k*T:lat_k*T+T, lat_k*T:lat_k*T+T]))

            sum_gpfa_diag += np.sum(np.abs(self._gpfa_covariance[lat_k*T:lat_k*T+T, lat_k*T:lat_k*T+T]))

        numb_total_elements = ((self._K_numb_latents * self._numb_inducing)**2)
        numb_diag_elements = ((self._numb_inducing)**2 * self._K_numb_latents)

        mean_diff = sum_diff / numb_total_elements
        mean_diff_diag = sum_diag / numb_diag_elements
        mean_diff_off_diag = (sum_diff - sum_diag) / (numb_total_elements - numb_diag_elements)

        mean_gpfa = sum_gpfa_total / numb_total_elements
        mean_gpfa_diag = sum_gpfa_diag / numb_diag_elements
        mean_gpfa_off_diag = (sum_gpfa_total - sum_gpfa_diag) / (numb_total_elements - numb_diag_elements)
        

        # singular values comparison:

        U, S, V = np.linalg.svd(matr_from_S)
        S = np.diag(S)

        fnorm_s = np.sqrt(np.trace( np.square(S) ))

        
        U, S, V = np.linalg.svd(self._gpfa_covariance)
        S = np.diag(S)

        fnorm_gpfa = np.sqrt(np.trace( np.square(S) ))

        
        U, S, V = np.linalg.svd((self._gpfa_covariance - matr_from_S))
        S = np.diag(S)

        fnorm_diff = np.sqrt(np.trace( np.square(S) ))

        if verbose:
            print(mean_abs_true, mean_abs_found)
            print("Difference between S and the generating one: ", mean_diff)
            print("     in diagonal:",mean_diff_diag)
            print("     off diagonal:",mean_diff_off_diag)
            print("fnorm things: ",fnorm_gpfa,fnorm_s,np.abs(fnorm_gpfa-fnorm_s),fnorm_diff)
            # print("   gpfa:",mean_gpfa,mean_gpfa_diag,mean_gpfa_off_diag)

        return mean_abs_true, mean_abs_found, mean_diff, mean_diff_diag, mean_diff_off_diag



    def _compare_S_with_generating(self, verbose = True):
        """
        
        This is a slightly questionable method on a theoretical level, so just ignore this, this is wrong.

        I was confusing pior covariance and posterior covariance.

        """        

        if "S_full_covars" in self._meta_data:

            mean_abs_true = np.mean(np.abs(self._meta_data["S_full_covars"]))
            mean_abs_found = np.mean(np.abs(self._S_full_matr))
            
            sum_diff = np.sum(np.abs(self._meta_data["S_full_covars"] - self._S_full_matr))

            # get the diagonal terms
            M = self._numb_inducing
            sum_diag = 0
            for lat_k in range(0, self._K_numb_latents):
                    sum_diag += np.sum(np.abs(self._meta_data["S_full_covars"][lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M] - self._S_full_matr[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]))

            numb_total_elements = ((self._K_numb_latents * self._numb_inducing)**2)
            numb_diag_elements = ((self._numb_inducing)**2 * self._K_numb_latents)

            mean_diff = sum_diff / numb_total_elements
            mean_diff_diag = sum_diag / numb_diag_elements
            mean_diff_off_diag = (sum_diff - sum_diag) / (numb_total_elements - numb_diag_elements)

            if verbose:
                print(mean_abs_true, mean_abs_found)
                print("Difference between S and the generating one: ", mean_diff)
                print("     in diagonal:",mean_diff_diag)
                print("     off diagonal:",mean_diff_off_diag)

            return mean_abs_true, mean_abs_found, mean_diff, mean_diff_diag, mean_diff_off_diag

        else:

            print("\nReconstructing generating S")

            # get generating S:
            generating_S = np.zeros((self._K_numb_latents,self._numb_inducing,self._numb_inducing))
            for lat_k in range(0, self._K_numb_latents):
                # k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
                K_xx = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
                K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                K_xz = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                K_zx = K_xz.transpose()

                zXz_inv = np.linalg.inv( np.matmul(K_zx, K_xz) )

                mid = np.matmul(np.matmul(K_zx , K_xx), K_xz)

                S = np.matmul(np.matmul(zXz_inv , mid), zXz_inv)
                S = np.matmul(np.matmul(K_zz , mid), K_zz)

                generating_S[lat_k,:,:] = S

            # compare:
            for lat_k in range(0, self._K_numb_latents):
                diff = np.abs(generating_S[lat_k,:,:] - self._S_covars[0,lat_k,:,:])
                # diff_2 = np.abs(generating_S[lat_k,:,:] - self._meta_data["S_full_covars"][lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M])

                print(np.mean(diff))
                # print("Difference between recovered and true generating",np.mean(diff_2))
            
            print("\nMapping to the other thing")

            # get generating S:
            generating_S = np.zeros((self._K_numb_latents,self._numb_inducing,self._numb_inducing))
            for lat_k in range(0, self._K_numb_latents):
                # k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)

                zXz_inv = np.linalg.inv( np.matmul(K_zx, K_xz) )

                mid = np.matmul(np.matmul(K_zx , K_xx), K_xz)

                S = np.matmul(np.matmul(zXz_inv , mid), zXz_inv)
                S = np.matmul(np.matmul(K_zz , mid), K_zz)

                generating_S[lat_k,:,:] = S

            # compare:
            for lat_k in range(0, self._K_numb_latents):
                K_xx = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
                K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                K_xz = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
                K_zx = K_xz.transpose()

                K_zz_inv = np.linalg.inv(K_zz)

                mid = np.matmul(np.matmul(K_zz_inv, self._S_covars[0,lat_k,:,:]), K_zz_inv)
                hat_K_xx = np.matmul(np.matmul(K_xz, mid), K_zx)      

                diff = np.abs(K_xx - hat_K_xx)

                print(np.mean(diff))


    def _compute_mean_off_diag_S(self):
        """
        Computes the mean of the covariance between different latent processes.
        """

        sum_on_S = np.sum(np.abs(self._S_full_matr))

        # get the diagonal terms
        M = self._numb_inducing
        sum_diag = 0
        for lat_k in range(0, self._K_numb_latents):
                sum_diag += np.sum(np.abs(self._S_full_matr[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]))

        mean_on_S = (sum_on_S - sum_diag) / ((self._K_numb_latents * self._numb_inducing)**2)
        print("mean off diag eleme of S:",mean_on_S)


    def _compare_with_previous(self, S_hat, m):
        """
        Compares the newly found parameters with the ones of MF.

        """

        percentage_of_higer_S = 0
        abs_diff_S = 0
        abs_diff_m = 0

        M = self._numb_inducing
        for lat_k in range(0, self._K_numb_latents):
            diff = np.abs(S_hat[lat_k*M:lat_k*M+M, lat_k*M:lat_k*M+M]) - np.abs(self._S_covars[self._temp_trial_i,lat_k,:,:])
            abs_diff_S += np.mean(diff)
            percentage_of_higer_S += np.mean((diff > 0).astype(int))
            
        abs_diff_S *= (1 / self._K_numb_latents)
        percentage_of_higer_S *= (1 / self._K_numb_latents)

        diff = np.abs(m) - np.abs(self._temp_param_m[self._temp_trial_i,:])
        abs_diff_m += np.mean(diff)

        # print(diff)
        # print(m)

        # print(percentage_of_higer_S, abs_diff_S, abs_diff_m)

        return percentage_of_higer_S, abs_diff_S, abs_diff_m







    #############################   Finite difference code  ################################


    def _finite_diff_of_Lm(self,theta):
        """
        Taking the finite difference derivative of Lm to be sure that autograd does its work properly.
        """

        derivate = np.zeros_like(theta)

        Lm_theta = self._Lm(theta)

        for var_i, var in enumerate(theta):
            theta_plus = theta.copy()
            theta_plus[var_i] += SMALL
            # theta_minus[var_i] -= SMALL

            derivate[var_i] = (self._Lm(theta_plus) - Lm_theta) / SMALL

        return derivate



    def _finite_diff_H_of_Lm(self,theta):
        """
        Taking the finite difference Hessian of Lm to be sure that autograd does its work properly.

        Carefull, it seems like the formula used here does not really work...
        """


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



    
    # def _recover_path_for_model(self):
    #     """
    #     Used to obtain the latent path after the fitting of the model.

    #     This is a method that is called after the EM iterations.
        
    #     """

        # for lat_k in range(0, self._K_numb_latents):
        #     for trial_i in range(0, self._numb_trials-1):
        #         print(np.mean( (self._S_covars[trial_i,lat_k,:,:]==self._S_covars[trial_i+1,lat_k,:,:]).astype(int) ))

    #     self.do_the_linear_response_correction()

    #     latents = np.zeros((self._meta_data["numb_trials"],self._K_numb_latents,len(self._meta_data["bin_times"])))

    #     # compute the latent GP trajectories from the variational means
    #     for lat_k in range(0,self._K_numb_latents):
    #         kappa_mat = make_K(self._bin_times, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
    #         kappa_invK = kappa_mat.dot(self._inv_kernel_mat_induc[lat_k,:,:])

    #         for trial_i in range(0,self._numb_trials):
    #             x_k_r = kappa_invK.dot(self._m_induc_mean[trial_i,lat_k,:])
    #             latents[trial_i,lat_k,:] = x_k_r

    #     return latents



    # def _Lm(self,theta):
    #     """
    #     This is the correct implementation of it.

    #     So bits could be deleted as we are actually only interested in the derivative of this.

    #     """
    #     M = self._numb_inducing # just making things clearer to read here
    #     numb_param_per_latent = M + ((M * (M + 1)) // 2)

    #     ################# 
    #     # the first step is to get the latent parameters m and S from the vector of mean parameters

    #     e_unit_vec_tilde = np.eye(numb_param_per_latent)
    #     e_unit_vec = np.eye(M)

    #     m_induc_mean_list = []
    #     S_covar_list = []

    #     for lat_k in range(0,self._K_numb_latents):
    #         start_index = lat_k * numb_param_per_latent

    #         # the means:
    #         # m_induc_mean[lat_k,:] = theta[start_index:start_index+M] 
    #         m_induc_mean_k = theta[start_index:start_index+M]
    #         m_induc_mean_list.append(m_induc_mean_k)

    #         S = np.zeros((self._numb_inducing, self._numb_inducing))

    #         for i in range(0,M):
    #             for j in range(i,M):
    #                 place = i * M + j - ((i * (i + 1)) // 2)
    #                 #print(place==place2, i,j,place)
    #                 S += np.outer(e_unit_vec[i,:], e_unit_vec[j,:]) * theta[start_index+M+place]

    #                 if i != j:
    #                     S += np.outer(e_unit_vec[j,:], e_unit_vec[i,:]) * theta[start_index+M+place]

    #         #temp1 = np.repeat(m_induc_mean_k.reshape(M,1), M, 1)
    #         # S -= np.multiply(temp1, np.transpose(temp1))
    #         S -= np.outer(m_induc_mean_k, m_induc_mean_k)

    #         S_covar_list.append(S)

    #     # just checking:
    #     # print("\n")
    #     # for lat_k in range(0,self._K_numb_latents):
    #     #     print(np.mean(np.abs(S_covar_list[lat_k] - self._S_covars[self._temp_trial_i,lat_k,:,:])), np.mean(np.abs(m_induc_mean_list[lat_k] - self._m_induc_mean[self._temp_trial_i,lat_k,:])))


    #     ###########
    #     # then compute Lm using these

    #     Lm = 0

    #     first_term = 0
    #     second_term = 0
    #     third_term = 0

    #     # some precomputations:
    #     if self._force_R_diag:
    #         Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
    #         logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
    #     else:
    #         Rinv = np.linalg.inv(self._R_noise)
    #         Rinv = ( Rinv + np.transpose(Rinv) ) / 2
    #         logdet_R = np.log(np.linalg.det(self._R_noise) + SMALL)

    #     CRinv = np.matmul( np.transpose(self._C_matrix), Rinv )
    #     CRinvC = np.matmul( CRinv, self._C_matrix )

    #     # second term :
    #     inv_Kzz_list = []
    #     sum_over_k_second = 0
    #     for lat_k in range(0,self._K_numb_latents):
    #         K_zz = make_K(self._z_induc_loc, self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
    #         inv_Kzz = np.linalg.inv(K_zz)
    #         inv_Kzz_list.append(inv_Kzz)

    #         expr = np.trace( np.matmul(inv_Kzz , S_covar_list[lat_k]) )

    #         expr += np.matmul( np.matmul( np.transpose( m_induc_mean_list[lat_k]  ), inv_Kzz ), m_induc_mean_list[lat_k] )

    #         expr += np.log( np.linalg.det(K_zz) + SMALL )

    #         sum_over_k_second += 0.5 * expr

    #     # first term:
    #     first_term += - (0.5 * logdet_R + 0.5 * self._numb_neurons * np.log(2 * np.pi)) * (self._numb_bins * self._numb_trials)

    #     for bin_i in range(0,self._numb_bins):

    #         third_term_temp_kappa = 0
    #         sum_over_k_first = 0
    #         for lat_k in range(0,self._K_numb_latents):
    #             inv_Kzz = inv_Kzz_list[lat_k]

    #             k_ii = make_K(self._bin_times[bin_i], self._bin_times[bin_i], self._kernel_func, self._kernel_param, lat_k)
    #             k_iZ = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
    #             k_Zi = np.transpose(k_iZ)
    #             matr_term = np.matmul( np.matmul( inv_Kzz , S_covar_list[lat_k]), inv_Kzz )
    #             matr_term -= inv_Kzz

    #             temp = k_ii + np.matmul( np.matmul( k_iZ,matr_term), k_Zi)

    #             sum_over_k_first += np.sum(S_covar_list[lat_k]) #CRinvC[lat_k,lat_k] * temp

    #             temp = k_ii - np.matmul( np.matmul( k_iZ, inv_Kzz), k_Zi)
    #             third_term_temp_kappa += np.outer( np.outer( np.transpose(self._C_matrix[:,lat_k]), temp), self._C_matrix[:,lat_k] )
    #         #print(third_term_temp_kappa.shape)

    #         third_term += 0.5 *  np.log(np.linalg.det(third_term_temp_kappa) + SMALL)


    #         # for trial_i in range(0,self._numb_trials):

    #         # forming m tilde:  
    #         m_tilde = np.zeros((self._K_numb_latents))

    #         e_unit_vect = np.eye(self._K_numb_latents)

    #         for lat_k in range(0,self._K_numb_latents):
    #             inv_Kzz = inv_Kzz_list[lat_k]
    #             K_new = make_K(self._bin_times[bin_i], self._z_induc_loc, self._kernel_func, self._kernel_param, lat_k)
    #             m_tilde += e_unit_vect[lat_k,:] * np.dot(np.matmul(K_new,inv_Kzz), m_induc_mean_list[lat_k] )

    #         first_term += - 0.5 * np.matmul( np.matmul( np.transpose(self._dataset[self._temp_trial_i,:,bin_i]) , Rinv), self._dataset[self._temp_trial_i,:,bin_i])

    #         # combining the two m K_i C terms:
    #         temp = np.transpose(m_tilde)
    #         first_term += np.matmul( temp, np.matmul( CRinv, (self._dataset[self._temp_trial_i,:,bin_i] - self._d_bias   )  ) )

    #         # combining the two d terms:
    #         first_term += np.matmul( np.matmul( np.transpose(self._d_bias), Rinv ), ( self._dataset[self._temp_trial_i,:,bin_i] - 0.5 * self._d_bias ) )

    #         # the big term:
    #         temp = m_tilde
    #         first_term += - 0.5 * np.matmul( np.matmul( np.transpose(temp), CRinvC ), temp )

    #         first_term += - 0.5 * sum_over_k_first

    #     # third term (end):
    #     third_term += (self._numb_neurons / 2) * (np.log(2 * np.pi) + 1) * self._numb_bins
            
    #     Lm = first_term + second_term + third_term
    #     return Lm


