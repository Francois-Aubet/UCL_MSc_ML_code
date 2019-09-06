"""

The Gaussian Process Factor Analysis (GPFA) model.

Please excuse my tendency to abuse of the global scope in classes..



TODO here:
- change the inverse of the big_K matrix, to be much more efficient
- change the inverse of the block diagonal matrix in E-step to be more efficient
- change the computation of the matrix multiplications in the E-step to have it a bit efficient

- make that the latent GPs can have different kernel parameters
- make so that the kernel parameters of the GP can be learnt
- add the computation of the LL

- before returning the latent in the paper they have this reordering of the latents according to their importance
    -> have to add it here

"""

from algorithms.Algorithm import Algorithm
from algorithms.IGPFA import IGPFA
import numpy as np
import time
from tqdm import tqdm 
from utilis import *
from scipy.linalg import block_diag


SMALL = 1e-6


class GPFA(IGPFA):
    """ GPFA 
    
    """

    algName = "GPFA"

    def __init__(self, dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step):
        """ Preparation of all the needed variables. """

        super().__init__(dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy, save_ortho_step)


    def _specific_init(self):
        """
        This method is called as an initialisation of model specific variables.

        In the GPFA model everything is there.
        """
        self._pre_compute_kernel_matrices()


    def _pre_compute_kernel_matrices(self):
        """
        This method is used to compute and save all the matrices that only 
            depend on the kernel parameters and the inducing points.

        If these are not updated, this method is only called once and then the values are saved.

        """
        
        # the kernel matrices and get their inverses:
        self._kernel_mat_induc = np.zeros((self._K_numb_latents, self._numb_bins, self._numb_bins))
        self._inv_kernel_mat_induc = np.zeros((self._K_numb_latents, self._numb_bins, self._numb_bins))

        for lat_k in range(0,self._K_numb_latents):
            K_zz = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
            self._kernel_mat_induc[lat_k,:,:] = K_zz
            self._inv_kernel_mat_induc[lat_k,:,:] = np.linalg.inv(K_zz)

        # the K tilde as defined in the report
        K_tilde = self._kernel_mat_induc[0,:,:]
        for lat_k in range(1,self._K_numb_latents):
            K_tilde = block_diag(K_tilde, self._kernel_mat_induc[lat_k,:,:])
        self._temp_param_K_tilde = K_tilde
        self._temp_param_inv_K_tilde = np.linalg.inv(K_tilde)



    def _E_step(self):
        """
        The E step of the EM algorithm to fit the model.

        Derivation in my thesis.

        @return :
        
        """
        self._temp_param_x = np.zeros((self._numb_trials,self._K_numb_latents, self._numb_bins))

        Rinv = np.diag(1 / (np.diag(self._R_noise) + 0))#SMALL))
        logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))

        ## byron
        # CRinv = np.matmul( np.transpose(self._C_matrix), Rinv )
        # CRinvC = np.matmul( CRinv, self._C_matrix )
        # K_big, K_big_inv = self._make_big_K()
        # CRinvC_big = np.kron(np.eye(self._numb_bins), CRinvC)
        # invM = np.linalg.inv(K_big_inv + CRinvC_big)
        ## 

        # create C tilde:
        C_tilde = np.zeros((self._numb_bins*self._numb_neurons,self._K_numb_latents*self._numb_bins))
        for lat_k in range(0,self._K_numb_latents):
            C_k = np.kron(np.eye(self._numb_bins), self._C_matrix[:,lat_k])
            # print( (C_k != 0).astype(int) )
            C_tilde[:,self._numb_bins*lat_k:self._numb_bins*(lat_k+1)] = C_k.transpose()

        Rinv_big = np.kron(np.eye(self._numb_bins), Rinv)
        R_big = np.kron(np.eye(self._numb_bins), self._R_noise)

        CRinv = np.matmul( np.transpose(C_tilde), Rinv_big )

        Sigma_x_inv = self._temp_param_inv_K_tilde + np.transpose(C_tilde).dot(Rinv_big).dot(C_tilde)

        Sigma_x = np.linalg.inv( Sigma_x_inv ) 

        self._temp_Sigma_x = Sigma_x

        all_but_y = Sigma_x.dot(CRinv)
        d_big = np.kron(np.ones(self._numb_bins), self._d_bias)

        # sigma_like = R_big + C_tilde.dot(self._temp_param_K_tilde).dot(np.transpose(C_tilde)) 
        # sigma_like_inv = np.linalg.inv( sigma_like )

        get_likelihood = False
        likelihood = 0
        # log_det = np.linalg.slogdet(sigma_like)[1]
        # print( log_det, log_det==0 )

        ## byron
        # C_big = np.kron(np.eye(self._numb_bins), self._C_matrix)
        # RinvC_big = Rinv_big.dot((C_big))
        # CRinv_big = np.matmul( np.transpose(C_big), Rinv_big )
        # # RinvC_big = np.kron(np.eye(self._numb_bins), RinvC)
        # #all_but_y_by = np.matmul(np.matmul( K_big, np.transpose(C_big) ), (Rinv_big - np.matmul( np.matmul( RinvC_big , invM), np.transpose(RinvC_big) ) ) )
        # all_but_y_by = invM.dot(CRinv_big)
        # sigma_like_by = C_big.dot(K_big).dot( C_big.transpose() ) + R_big
        ## 


        for trial_i in range(0,self._numb_trials):
            y_big = np.transpose(self._dataset[trial_i,:,:]).reshape(self._numb_bins*self._numb_neurons)

            big_diff = y_big - d_big

            X_big = np.matmul(all_but_y, big_diff)

            # X_big_by = np.matmul(all_but_y_by, big_diff)

            self._temp_param_x[trial_i,:,:] = np.transpose(X_big.reshape(self._numb_bins, self._K_numb_latents))
            for lat_k in range(0,self._K_numb_latents):
                self._temp_param_x[trial_i,lat_k,:] = X_big[self._numb_bins*lat_k:self._numb_bins*(lat_k+1)]


            # X_by = np.transpose(X_big_by.reshape(self._numb_bins, self._K_numb_latents))
            X = self._temp_param_x[trial_i,:,:] 

            # self._temp_param_x[trial_i,:,:] = X_by

            if get_likelihood:

                likelihood += - 0.5 * self._numb_bins * self._numb_neurons * np.log(2 * np.pi)
                likelihood += - 0.5 * big_diff.dot(sigma_like_inv).dot(big_diff)
                likelihood += - 0.5 * log_det

        # self._free_energy_2_list.append(likelihood)
        # orth_res = get_orthogonality_score(self._C_matrix, False)
        # self._free_energy_1_list.append(orth_res[0])
        # print(likelihood)
        self._likelihood = likelihood

        self._xsm = self._temp_param_x.copy()
        Vsm = np.zeros((self._K_numb_latents, self._K_numb_latents, self._numb_bins))
        idx = range(0, self._K_numb_latents*self._numb_bins, self._numb_bins)
        idx = np.array(idx)
        
        for t in range(0, self._numb_bins):
            Vsm[:,:,t] = self._temp_Sigma_x[(idx+t)[:, np.newaxis],(idx+t)]#[start_t:end_t, start_t:end_t]

        if self.use_MF_approximation:
            for t in range(0, self._numb_bins):
                Vsm[:,:,t] = np.linalg.inv(np.diag(np.diag(np.linalg.inv(Vsm[:,:,t]))))


        # Vsm_by = np.zeros((self._K_numb_latents, self._K_numb_latents, self._numb_bins))
        # idx = range(0, self._K_numb_latents*self._numb_bins + 1, self._K_numb_latents)
        # for t in range(0, self._numb_bins):
        #     Vsm_by[:,:,t] = invM[idx[t]:idx[t+1],idx[t]:idx[t+1]]

        self._Vsm = Vsm


    def _M_step(self):
        """
        The M step of the EM algorithm to fit the model.

        Again, not following Byron anymore.

        """

        d_sum_term = 0
        C1_sum_term = 0
        C2_sum_term = 0
        R_sum_term = 0
        
        C_covar_loging_term = 0

        for bin_i in range(0,self._numb_bins):

            # get S_ii:
            start_t = self._K_numb_latents * bin_i
            end_t = self._K_numb_latents * (bin_i+1)
            S_tilde_ii = self._Vsm[:,:,bin_i]
            #self._temp_Sigma_x[start_t:end_t, start_t:end_t]

            # get Sigma_ii:
            Sigma_ii = self._C_matrix.dot(S_tilde_ii).dot(np.transpose(self._C_matrix))

            # now the trial sum:
            for trial_i in range(0, self._numb_trials):

                # get m_tilde i r
                m_tilde_i_r = self._temp_param_x[trial_i,:,bin_i]

                # get mu_i_r
                mu_i_r = self._C_matrix.dot(m_tilde_i_r) + self._d_bias

                # update sum terms:
                # C:
                C1_sum_term += np.outer( (self._dataset[trial_i,:,bin_i] - self._d_bias) , m_tilde_i_r )
                C2_sum_term += S_tilde_ii + np.outer(m_tilde_i_r, m_tilde_i_r)
                C_covar_loging_term += S_tilde_ii

                # d:
                d_sum_term += self._dataset[trial_i,:,bin_i] - self._C_matrix.dot(m_tilde_i_r)

                # R:
                temp = self._dataset[trial_i,:,bin_i] - mu_i_r
                R_sum_term += np.outer(temp, temp) + Sigma_ii

        self._C_matrix = C1_sum_term.dot(np.linalg.inv(C2_sum_term))
        self._C_matrix_big = self._C_matrix

        self._gather_data_Cupdate(C1_sum_term, C2_sum_term, C_covar_loging_term)

        # if not self._learn_only_C:

        self._d_bias = (1 / ( self._numb_trials * self._numb_bins )) * d_sum_term

        self._R_noise = (1 / ( self._numb_trials * self._numb_bins )) * R_sum_term
        if self._force_R_diag:
            self._R_noise = np.diag(np.diag(self._R_noise))

            



    def _E_step_by(self):
        """
        The E step of the EM algorithm to fit the model.

        Retrieves the extimate of the latent values.
        
        Since this is not the core of my project, and it has been done before,
            I just addapt here the matlab code of Byron Yu.
            (also most variables names and so on)

        @return :
            - xsm (xDim x T)        -- posterior mean at each timepoint
            - Vsm (xDim x xDim x T) -- posterior covariance at each timepoint
            - VsmGP (T x T x xDim)  -- posterior covariance of each GP
        """

        # Precomputations:
        if self._force_R_diag:            
            Rinv = np.diag(1 / (np.diag(self._R_noise) + SMALL))
            logdet_R = np.sum(np.log( np.diag(self._R_noise) + SMALL ))
        else:
            Rinv = np.linalg.inv(self._R_noise)
            Rinv = ( Rinv + np.transpose(Rinv) ) / 2
            logdet_R = np.log(np.linalg.det(self._R_noise))

        CRinv = np.matmul( np.transpose(self._C_matrix), Rinv )
        CRinvC = np.matmul( CRinv, self._C_matrix )

        K_big, K_big_inv = self._make_big_K()

        CRinvC_big = np.kron(np.eye(self._numb_bins), CRinvC)
        

        invM = np.linalg.inv(K_big_inv + CRinvC_big)

        # print((invM != 0).astype(int))
        # print(self._K_numb_latents, self._numb_bins)
        # print(invM.shape)

        # create Vsm, xDim x xDim posterior covariance for each timepoint
        Vsm = np.zeros((self._K_numb_latents, self._K_numb_latents, self._numb_bins))
        idx = range(0, self._K_numb_latents*self._numb_bins + 1, self._K_numb_latents)
        for t in range(0, self._numb_bins):
            Vsm[:,:,t] = invM[idx[t]:idx[t+1],idx[t]:idx[t+1]]
            # both have shap numb_latents x numb_latents

        # create T x T posterior covariance for each GP
        VsmGP = np.zeros((self._numb_bins, self._numb_bins, self._K_numb_latents))
        idx = range(0, self._K_numb_latents*self._numb_bins, self._K_numb_latents)
        idx = np.array(idx)
        for lat_i in range(0,self._K_numb_latents):
            VsmGP[:,:,lat_i] = invM[(idx+lat_i)[:, np.newaxis],(idx+lat_i)]

        # create xsm (xDim x T) posterior mean at each timepoint
        xsm = np.zeros((self._numb_trials,self._K_numb_latents, self._numb_bins))

        Rinv_big = np.kron(np.eye(self._numb_bins), Rinv)
        C_big = np.kron(np.eye(self._numb_bins), self._C_matrix)

        RinvC = np.transpose(CRinv)
        RinvC_big = np.kron(np.eye(self._numb_bins), RinvC)

        all_but_y = np.matmul(np.matmul( K_big, np.transpose(C_big) ), (Rinv_big - np.matmul( np.matmul( RinvC_big , invM), np.transpose(RinvC_big) ) ) )


        d_big = np.kron(np.ones(self._numb_bins), self._d_bias)

        for trial_i in range(0,self._numb_trials):
            y_big = np.transpose(self._dataset[trial_i,:,:]).reshape(self._numb_bins*self._numb_neurons)

            big_diff = y_big - d_big

            X_big = np.matmul(all_but_y, big_diff)

            xsm[trial_i,:,:] = np.transpose(X_big.reshape(self._numb_bins, self._K_numb_latents))


        self._xsm = xsm
        self._Vsm = Vsm
        self._VsmGP = VsmGP

        likelihood = 0
        if True:
            
            R_big = np.kron(np.eye(self._numb_bins), self._R_noise)

            sigma = C_big.dot(K_big).dot( C_big.transpose() ) + R_big

            # print((sigma != 0).astype(int))

            inv_sigma = np.linalg.inv(sigma)

            log_det = np.log( np.linalg.det(sigma) + SMALL )


            ## compute the core:
            for trial_i in range(0,self._numb_trials):

                y_big = np.transpose(self._dataset[trial_i,:,:]).reshape(self._numb_bins*self._numb_neurons)

                side = y_big - d_big

                likelihood += - 0.5 * np.matmul( np.matmul( side.transpose()  , inv_sigma), side )

                likelihood += - 0.5 * log_det

        self._free_energy_2_list.append(likelihood)
        orth_res = get_orthogonality_score(self._C_matrix, False)
        self._free_energy_1_list.append(orth_res[0])
        print(likelihood)
        self._likelihood = likelihood
            # Getting the LL.
            # val = - self._numb_bins * logdet_R


        # get full covars:

        self._full_covars = np.zeros((self._numb_bins*self._K_numb_latents, self._numb_bins*self._K_numb_latents))
        idx = range(0, self._K_numb_latents*self._numb_bins, self._K_numb_latents)
        idx = np.array(idx)
        T = self._numb_bins
        for lat_i in range(0,self._K_numb_latents):
            for lat_j in range(0,self._K_numb_latents):
                self._full_covars[lat_i*T:lat_i*T+T, lat_j*T:lat_j*T+T] = invM[(idx+lat_i)[:, np.newaxis],(idx+lat_j)]



    def _make_big_K(self):
        """
        Creates the big strange diagonal kernel matrix as described in the original GPFA paper.
        
        """

        kernel_func = se_kernel

        K_big = np.zeros((self._numb_bins*self._K_numb_latents, self._numb_bins*self._K_numb_latents))

        for lat_k in range(0,self._K_numb_latents):
            K = make_K(self._bin_times, self._bin_times, self._kernel_func, self._kernel_param, lat_k)
            #make_cov_matrix(self._bin_times, kernel_func)

            # there might be a more efficient method to do this: (a quick search did not allow me to find it)
            for index_i in range(lat_k,self._numb_bins*self._K_numb_latents,self._K_numb_latents):
                for index_j in range(lat_k,self._numb_bins*self._K_numb_latents,self._K_numb_latents):
                    K_big[index_i,index_j] = K[index_i // self._K_numb_latents, index_j // self._K_numb_latents]
                    
        #print(K_big)

        # TODO: warrning, this inverse here is super dupper inefficient, can easily be done better
        return K_big, np.linalg.inv(K_big)



    def _M_step_by(self):
        """
        The M step of the EM algorithm to fit the model.

        Again, following Byron Yu's code in order to update the C, d, and R. And potentially kernel params.
        """

        xsm = self._xsm
        Vsm = self._Vsm

        # update C and d
        sum_Pauto = np.zeros((self._K_numb_latents, self._K_numb_latents))
        sum_Vsm = np.sum(Vsm,2)
        for trial_i in range(0, self._numb_trials):
            sum_Pauto += sum_Vsm + np.matmul(xsm[trial_i,:,:], np.transpose(xsm[trial_i,:,:]))

        Y = self._dataset.transpose([0,2,1]).reshape((self._numb_trials*self._numb_bins,self._numb_neurons))
        Xsm = xsm.transpose([0,2,1]).reshape((self._numb_trials*self._numb_bins,self._K_numb_latents))

        sum_yxtrans = np.matmul(np.transpose(Y), Xsm)
        sum_yall = np.sum(Y,0)
        sum_xall = np.sum(Xsm,0)

        temp = np.column_stack((np.reshape(sum_xall, (1,self._K_numb_latents)) , np.array([self._numb_bins * self._numb_trials]) ))
        term = np.row_stack(( np.column_stack((sum_Pauto, sum_xall)) ,  temp ))

        numerator = np.column_stack((sum_yxtrans,  sum_yall ))

        Cd = np.matmul(numerator , np.linalg.inv(term) )

        self._C_matrix = Cd[:,0:self._K_numb_latents]
        self._C_matrix_big = self._C_matrix
        self._d_bias = Cd[:,-1]

        # update of R:
        if self._force_R_diag:
            sum_yytrans = np.sum(np.square(Y) ,0)
            yd = sum_yall * self._d_bias

            temp = sum_yxtrans - np.outer(self._d_bias, sum_xall)
            term = np.sum(temp * self._C_matrix, 1 )

            r = np.square(self._d_bias) + (sum_yytrans - 2 * yd - term) / (self._numb_bins * self._numb_trials)

            #r = np.max
            self._R_noise = np.diag(r)

        else:
            sum_yytrans = np.matmul(np.transpose(Y), Y)
            yd = np.outer(sum_yall, self._d_bias)

            term = np.matmul( (sum_yxtrans - np.outer(self._d_bias, sum_xall)), np.transpose(self._C_matrix) )

            R = np.outer(self._d_bias, self._d_bias) + (sum_yytrans - yd - np.transpose(yd) - term) / (self._numb_bins * self._numb_trials)

            self._R_noise = (R + np.transpose(R)) / 2
            



    def _recover_path_for_model(self):
        """
        Used to obtain the latent path after the fitting of the model.

        This is a method that is called after the EM iterations.
        
        """
        
        return self._xsm
        # return self._temp_param_x



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


    def _compute_free_energy_para_taus(self, param):

        return self._compute_free_energy_full()[0]


    def _compute_free_energy_full(self):
        """


        """
        # Precomputations:
        Rinv, CRinv, CRinvC, logdet_R = self._do_Rinv_precompute()

        first_term = 0
        kl_to_prior = 0

        # first term:
        first_term += logdet_R + self._numb_neurons * np.log(2 * np.pi )
        first_term *= self._numb_trials * self._numb_bins

        # KL term:
        kl_to_prior += 0.5 * np.linalg.slogdet(self._temp_Sigma_x)[1]#np.log(np.linalg.det(self._temp_Sigma_x))
        kl_to_prior += 0.5 * self._K_numb_latents * self._numb_bins * (np.log(2 * np.pi * np.e ) - np.log(2 * np.pi ))

        for lat_k in range(0, self._K_numb_latents):
            start_k = self._numb_bins * lat_k
            end_k = self._numb_bins * (lat_k+1)

            kl_to_prior += - 0.5 * np.linalg.slogdet(self._kernel_mat_induc[lat_k,:,:])[1]# * np.log(np.linalg.det(self._kernel_mat_induc[lat_k,:,:]))

            kl_to_prior += - 0.5 * np.trace(self._inv_kernel_mat_induc[lat_k,:,:].dot(self._temp_Sigma_x[start_k:end_k, start_k:end_k]))

        kl_to_prior *= self._numb_trials


        # now the trial sum:
        for trial_i in range(0, self._numb_trials):

            # KL term:
            for lat_k in range(0, self._K_numb_latents):
                temp = self._temp_param_x[trial_i,lat_k,:]

                kl_to_prior += - 0.5 * np.trace(self._inv_kernel_mat_induc[lat_k,:,:].dot(np.outer(temp, temp)))


            # now the trial sum:
            for bin_i in range(0, self._numb_bins):
                # get S_ii:
                S_tilde_ii = self._Vsm[:,:,bin_i]

                # get m_tilde i r
                m_tilde_i_r = self._temp_param_x[trial_i,:,bin_i]

                y = self._dataset[trial_i,:,bin_i]

                first_term += y.transpose().dot(Rinv).dot(y)

                first_term += np.trace(CRinvC.dot(S_tilde_ii + np.outer(m_tilde_i_r, m_tilde_i_r)))

                first_term += - 2 * y.transpose().dot(Rinv).dot(self._d_bias)

                first_term += self._d_bias.transpose().dot(Rinv).dot(self._d_bias)

                temp = self._C_matrix.dot(m_tilde_i_r)
                first_term += - 2 * temp.transpose().dot(Rinv).dot(y - self._d_bias)


        first_term *= - 0.5
        kl_to_prior *= -1

        free_energy = first_term - kl_to_prior

        #print(free_energy, first_term, kl_to_prior)
        return free_energy, first_term, kl_to_prior


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

        sigma_like = R_big + C_tilde.dot(self._temp_param_K_tilde).dot(np.transpose(C_tilde)) 
        sigma_like_inv = np.linalg.inv( sigma_like )

        log_det = np.linalg.slogdet(sigma_like)[1]

        for trial_i in range(0,self._numb_trials):
            y_big = np.transpose(self._dataset[trial_i,:,:]).reshape(self._numb_bins*self._numb_neurons)

            big_diff = y_big - d_big

            likelihood += - 0.5 * self._numb_bins * self._numb_neurons * np.log(2 * np.pi)
            likelihood += - 0.5 * big_diff.dot(sigma_like_inv).dot(big_diff)
            likelihood += - 0.5 * log_det


        return likelihood






















