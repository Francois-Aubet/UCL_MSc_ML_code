"""


This is the GPFA interface class with all setting for the GPFA model.


"""

from abc import ABCMeta, abstractmethod
from algorithms.Algorithm import Algorithm

import sklearn.decomposition
from scipy.linalg import block_diag

from tqdm import tqdm 
from utilis import *
import matplotlib.pyplot as plt

import numpy as np

#from algorithms.GPFA_sv_lr import GPFA_sv_lr
import copy

# the auto grad things:
#import autograd.numpy as np
#import autograd.numpy.random as npr
from autograd import value_and_grad, grad
from scipy.optimize import minimize

# np.set_printoptions(suppress=True)

SMALL = 1e-7


class IGPFA(Algorithm):
    """ 
    This is the interface common to all GPFA.

    I contains all common methods.
    
    """

    algName = ""

    # we define the class as an abstract class:
    __metaclass__ = ABCMeta



    def __init__(self, dataset, meta_data, bin_times, numb_latents, max_EM_iterations, numb_inducing, learn_kernel_params, plot_free_energy = 0, save_ortho_step = -1):
        """ Preparation of all the needed variables. """

        super().__init__(dataset, meta_data)

        self._K_numb_latents = numb_latents

        self._bin_times = bin_times

        self._numb_trials, self._numb_neurons, self._numb_bins = dataset.shape

        self._numb_inducing = numb_inducing
        
        self._total_duration = self._meta_data["total_duration"]
        #print(self._numb_bins == len(bin_times))

        # initialise model parameters:
        #self._d_bias = np.ones((self._numb_neurons))
        self._d_bias = np.mean(dataset, axis=(0,2))
        self._C_matrix = np.zeros((self._numb_neurons, self._K_numb_latents))
        self._C_matrix =  np.random.rand(self._numb_neurons, self._K_numb_latents) / 2
        self._R_noise = np.zeros((self._numb_neurons, self._numb_neurons))

        # the variational parameters:
        #self._z_induc_loc = np.zeros((self._numb_trials, self._numb_latents, self._numb_inducing))
        self._z_induc_loc = np.zeros((self._numb_inducing))
        self._m_induc_mean = np.zeros((self._numb_trials, self._K_numb_latents, self._numb_inducing))
        self._S_covars = np.zeros((self._numb_trials, self._K_numb_latents, self._numb_inducing, self._numb_inducing))
        self._S_full_matr = np.zeros((self._K_numb_latents*self._numb_inducing, self._K_numb_latents*self._numb_inducing))
        # self._S_full_matr = np.eye((self._K_numb_latents*self._numb_inducing))
        

        # initialise GPs parameters:
        tau_scales = np.array([20.0,170.0])#np.abs(100 * np.zeros((self._K_numb_latents)) + 20 * np.random.randn(self._K_numb_latents))
        tau_scales = np.linspace(50, 500, self._K_numb_latents)
        tau_scales = np.array([98.048, 130.231,  76.05,   72.676,  93.422, 377.282, 278.314, 424.391, 213.08, 511.815 ])

        sigma_ns = 0.001 * np.ones((self._K_numb_latents))
        self._kernel_param = {"tau" : tau_scales, "sigma_n" : sigma_ns}
        self._kernel_func = se_kernel

        # can set the original tau scales:
        if "tau_s_gamma" in self._meta_data and self._K_numb_latents > 0:
            for lat_k in range(0, min(self._K_numb_latents, self._meta_data["K_gamma"])):
                self._kernel_param["tau"][lat_k] = self._meta_data["tau_s_gamma"][lat_k]

        # print(self._kernel_param["tau"])

        # some training settings:
        self._max_EM_iterations = max_EM_iterations
        self._stopping_thershold = 10**(-8)
        self._logging_freq = 10

        self._learn_kernel_params = learn_kernel_params
        self._learn_GP_noise = False
        self._force_R_diag = True

        self._learn_only_C = False
        #self._learn_inducing_locations = False
        # for now we do everything with preset z locations

        self._plot_free_energy = plot_free_energy
        self._free_energy_list = []
        self._free_energy_1_list = []
        self._free_energy_2_list = []

        self._free_tupple_list = []
        self._angle_list = []

        self._save_ortho_step = save_ortho_step
        self._orth_c_list = []

        self._intialise_to_gen_param = False

        self._gpfa_covariance = None

        self.use_MF_approximation = False

        self._lr_grad_update_learning_rate = 0.00001


    def extract_neural_path(self, verbose = True):
        """ 
        
        @:return: the neural path latents
        """

        ## first a quick initialization of the model parameters using FA:
        self._fa_initialisation()
        
        # we initialize a couple of variables that only depend on kernel parameters and z_loc:
        self._specific_init()
        angle = get_orthogonality_score(self._C_matrix, False)

        # Fit model using EM:
        for epoch in tqdm(range(0,self._max_EM_iterations)):
            

            # run E step:
            self._E_step()
            
            # get the free energy and print it:
            # if self._plot_free_energy != 0:
            #     # F = self._compute_free_energy()
            #     F = self._compute_free_energy_full()[0]
            #     #F = self.compute_likelihood()
            #     self._free_energy_list.append(F)

            # run M step:
            self._M_step()


            # get the free energy and print it:
            if self._plot_free_energy != 0:
                # F = self._compute_free_energy()
                F = self._compute_free_energy_full()[0]
                #F = self.compute_likelihood()
                self._free_energy_list.append(F)

            # creating an array of measurements of the orthogonality of C:
            if self._save_ortho_step != -1 and np.mod(epoch, self._save_ortho_step) == 0:
                # if self.algName == "sv_GPFA" or self.algName == "lr_sv_GPFA":
                #     free = self._compute_free_energy()
                #     like = self._compute_likelihood_term()
                #     kl = self._compute_KL_term()

                #     # free_full, like_full, kl_full = self._algo._compute_free_energy_full()

                #     like_true = self.compute_likelihood()

                #     # print([free, like, kl])
                #     self._free_tupple_list.append([free, like, kl, like_true, like_true - free]) 

                # angle = get_angle_subspaces(self._meta_data["C_gamma"],self._C_matrix,False)
                # angle = get_angle_subspaces(self._C_gamma, self._C_alpha)
                angle = get_angle_subspaces(self._C_gamma[:,:2], self._C_gamma[:,2:])
                angle = get_orthogonality_score( np.concatenate([self._C_alpha , self._C_gamma ], axis=1) )
                # print(angle)
                # print(angle)
                self._angle_list.append(angle) 

                # slightly different for lr_sv_GPFA:
                if self.algName == "pppp_lr_sv_GPFA":
                    # this is ugly but fine for now:
                    #temp_algo = algorithms.GPFA_sv_lr(self._dataset, self._meta_data, self._bin_times, self._K_numb_latents, self._max_EM_iterations, self._numb_inducing, self._learn_kernel_params)
                    temp_algo = copy.deepcopy(self)
                    #temp_algo._C_matrix = self._C_matrix

                    temp_algo.linear_response_correction()
                    temp_algo.update_model_parameters_after_LR()

                    orth_res = get_orthogonality_score(temp_algo._C_matrix, False)
                    self._orth_c_list.append(orth_res[0])
                    #print(get_orthogonality_score(self._C_matrix, False))


                    free, like, kl = temp_algo._compute_free_energy_full()
                    like_true = temp_algo.compute_likelihood()
                    self._free_tupple_list.append([free, like, kl, like_true, like_true - free]) 

                else:
                    orth_res = get_orthogonality_score(self._C_matrix, False)
                    self._orth_c_list.append(orth_res[0])

        # run a last E step:
        self._E_step()
        angle = get_orthogonality_score(self._C_matrix, False)

        # if verbose:
        #     get_orthogonality_score(self._C_matrix)
        ## compute the latent GP trajectories from the variational means
        latents = self._recover_path_for_model()

        # monitoring the free energy
        if self._plot_free_energy != 0:
            print(np.around(self._free_energy_list,2))
            plt.plot(self._free_energy_list)#[1:])
            if self._plot_free_energy > 1:
                plt.figure()
                plt.plot(self._free_energy_1_list)
                plt.figure()
                plt.plot(self._free_energy_2_list)
        
        if self._learn_kernel_params:
            print("Taus: ",self._kernel_param["tau"])

        # plt.show()
            
        return latents



    def _fa_initialisation(self):
        """
        Initializes the model parameters (C and R) using factor analysis.

        """
        if self._intialise_to_gen_param:
            self._C_matrix = self._meta_data["C_gamma"].copy()
            self._d_bias = self._meta_data["d_bias"].copy()

            if "gen_R" in self._meta_data:
                self._R_noise = self._meta_data["gen_R"].copy()

            return


        # create a big matrix with all the inputs:
        temp_data = self._dataset.transpose([0,2,1]).reshape((self._numb_trials*self._numb_bins,self._numb_neurons))

        # fit fa:
        fa = sklearn.decomposition.FactorAnalysis(n_components=self._K_numb_latents)
        fa.fit(temp_data)

        # save the params
        C_cand = np.transpose(fa.components_)
        for lat_k in range(0, self._K_numb_latents):
            if np.mean(C_cand[:,lat_k]) > 0.001:
                self._C_matrix[:,lat_k] = C_cand[:,lat_k]
        self._R_noise = np.diag(fa.noise_variance_)

        self._meta_data["C_gamma"] = C_cand.copy()

        return

        # to initialise it less orthogonal:
        U, S, V = np.linalg.svd(self._C_matrix)
        m, n = self._C_matrix.shape

        # print(U.shape, S.shape, V.shape)
        Smat = np.zeros_like(self._C_matrix)
        Smat[:n,:n] = np.diag(S)
        USmat = U.dot(Smat)

        # now we want to perturbate it:
        # if "C_gamma" in self._meta_data:
        #     U_o, S_o, V_o = np.linalg.svd(self._meta_data["C_gamma"])
        #     self._C_matrix = USmat.dot(V_o)


        # else:
        a = 90

        a = get_orthogonality_score(USmat, False)[0]
        # print("self._C_matrix: ", a)
        while a > 50 or a < 30:
            perturbation_matr = np.random.rand(self._K_numb_latents,self._K_numb_latents)
            # print("perturbation_matr: ", get_orthogonality_score(perturbation_matr, False)[0])
            self._C_matrix = USmat.dot(perturbation_matr)
            a = get_orthogonality_score(self._C_matrix, False)[0]
            # print("self._C_matrix: ", a)

        return

        perturbation_matr = np.random.rand(self._K_numb_latents,self._K_numb_latents)
        angle = np.deg2rad(20)
        perturbation_matr = np.array([ [np.cos(angle), np.sin(angle)] , [-np.sin(angle), np.cos(angle)] ])

        goal_angle = 40
        theta = np.cos(np.deg2rad(goal_angle))

        # b_3 = None

        # while b_3 is None:
        #     b_1 = np.random.randn(1)
        #     b_2 = np.random.randn(1)

        #     b_2_2 = np.square(b_2)
        #     b_1_2 = np.square(b_1)
        #     theta = np.square(theta)
        #     b_comp_a = b_2 - b_2_2 * theta - b_1_2 * theta
        #     b_comp_d = 2 * b_2_2 * b_1 - b_1_2 * theta
        #     b_comp_c = b_1_2 * b_2_2 - b_1_2 * b_2_2 * theta - np.square(b_2_2) * theta

        #     det = np.square(b_comp_d) - 4 * b_comp_a * b_comp_c
        #     if det > 0:
        #         b_3 = (- b_comp_d + np.sqrt(det) ) / (2 * b_comp_a)
        #         b_3_1 = (- b_comp_d + np.sqrt(det) ) / (2 * b_comp_a)
        #         b_3_2 = (- b_comp_d - np.sqrt(det) ) / (2 * b_comp_a)
        #     else:
        #         print("problem")


        # perturbation_matr_1 = np.squeeze(np.array([ [b_1, b_2] , [b_2, b_3_1] ]))
        # perturbation_matr_2 = np.squeeze(np.array([ [b_1, b_2] , [b_2, b_3_2] ]))
        # # print(perturbation_matr, perturbation_matr.shape)
        # print("perturbation_matr: ", get_orthogonality_score(perturbation_matr_1)[0])
        # print("perturbation_matr: ", get_orthogonality_score(perturbation_matr_2)[0])


        # print("USmat: ", get_orthogonality_score(USmat)[0])


        # self._C_matrix = USmat.dot(perturbation_matr)
        # a = get_orthogonality_score(self._C_matrix, False)[0]
        # print("self._C_matrix: ", a)

        a = get_orthogonality_score(USmat[:,0:2])[0]
        a = get_orthogonality_score(USmat[:,0:2].dot(perturbation_matr))[0]

        if self._K_numb_latents > 2:
            self._C_matrix = USmat
            
            for lat_k1 in range(0,self._K_numb_latents):
                for lat_k2 in range(lat_k1,self._K_numb_latents):
                    print(lat_k1,lat_k2)

                    to_rotate = np.array([self._C_matrix[:,lat_k1], self._C_matrix[:,lat_k2]]).transpose()

                    rotated = to_rotate.dot(perturbation_matr)

                    self._C_matrix[:,lat_k1] = rotated[:,0]
                    self._C_matrix[:,lat_k2] = rotated[:,1]

                    a = get_orthogonality_score(self._C_matrix)[0]
                    # print("self._C_matrix: ", a)
            
            # for lat_k in range(0,self._K_numb_latents-1):

            #     perturbation_matr_k = np.eye(self._K_numb_latents)
            #     perturbation_matr_k[lat_k:lat_k+2,lat_k:lat_k+2] = perturbation_matr

            #     print("perturbation_matr: ", get_orthogonality_score(perturbation_matr_k, False)[0])
            #     self._C_matrix = self._C_matrix.dot(perturbation_matr_k)
            #     a = get_orthogonality_score(self._C_matrix, False)[0]
            #     print("self._C_matrix: ", a)


            # print(a)

        # perturbation_matr = np.array([[0.2,0.4,0.2],[0.4,-0.4,0.2],[0.2,0.2,0.7]])
        # self._C_matrix = USmat.dot(perturbation_matr)

        # get_orthogonality_score(perturbation_matr)
        # get_orthogonality_score(USmat)
        # get_orthogonality_score(USmat.dot(perturbation_matr))


        # self._C_matrix = self._meta_data["C_gamma"]


    
    def set_model_params_to(self, C, d, R):
        """
        """
        self._C_matrix = C.copy()
        self._d_bias = d.copy()
        self._R_noise = R.copy()

    def _E_step(self):
        """
        E step.
        
        """
        raise NotImplementedError("Must override method")
        

    def _M_step(self):
        """
        M step.
        
        """
        raise NotImplementedError("Must override method")

    def _specific_init(self):
        """
        This method is called as an initialisation of model specific variables.

        """
        raise NotImplementedError("Must override method")

    def _recover_path_for_model(self):
        """
        Used to obtain the latent path after the fitting of the model.

        This is a method that is called after the EM iterations.
        
        """
        raise NotImplementedError("Must override method")

    def _recover_path_for_orthonorm_model(self):
        """
        Used to obtain the latent path after the fitting of the model.

        This is a method that is called after the EM iterations.
        
        """

        eigen = np.sqrt(np.linalg.eig(self._C_matrix.transpose().dot(self._C_matrix))[0])
        print(eigen)
        order_of_importance = np.argsort(-eigen)
        print(eigen[order_of_importance])

        self._taus_ordered = self._get_tau_param()[order_of_importance]
        print(self._taus_ordered)
        
        latents = self._recover_path_for_model()

        U, S, V = np.linalg.svd(self._C_matrix)
        m, n = self._C_matrix.shape

        print(S)

        # print(U.shape, S.shape, V.shape)
        Smat = np.zeros_like(self._C_matrix)
        Smat = np.diag(S) # [:n,:n]
        SmatV = Smat.dot(V)

        for trial_i in range(0, self._numb_trials ):
            for bin_i in range(0, self._numb_bins ):

                latents[trial_i,:,bin_i] = SmatV.dot(latents[trial_i,:,bin_i])

        

        return latents



    def _compute_free_energy(self):
        """
        Compute and return the free energy.
        
        """

        F = self._compute_free_energy_para_taus(self._get_tau_param())

        #F = self._compute_free_energy_full()[0]

        return F



    def _get_tau_param(self):
        """
        Returns the kernel time scale parameters in order to input them in the free energy function.
        """

        tau = self._kernel_param["tau"]

        return tau
        

    def _compute_KL_term(self):
        """
        Compute the KL term.
        
        """
        raise NotImplementedError("Must override method")

    def _compute_likelihood_term(self):
        """
        Compute the KL term.
        
        """
        raise NotImplementedError("Must override method")


    def _compute_free_energy_para_taus(self, param):
        """
        Compute and return the free energy from the kernel parameters.

        This function is to be used with auto grad to update the kernel parameters.
        
        """
        raise NotImplementedError("Must override method")



    def _update_kernel_tau(self):
        """

        """
        #print(self._compute_free_energy())
        tau_update_list = []
        print(self._compute_free_energy())

        # get the gradient of the free energy with regard to the parameters
        gradient_func = grad(self._compute_free_energy_para_taus)

        numb_iterations = 4
        learning_rate = 0.01

        x0 = self._kernel_param["tau"]
        result = minimize(self._compute_free_energy_para_taus, x0, method='L-BFGS-B', jac=gradient_func, \
            options={'disp': None, 'maxcor': 6, 'maxfun': 150, 'maxiter': 15, 'iprint': -1, 'maxls': 6})
        
            # options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
        
        self._kernel_param["tau"] = result.x
        # for iter_i in range(0,numb_iterations):
        #     gradients_tau = gradient_func(self._kernel_param["tau"])
        #     self._kernel_param["tau"] += learning_rate * gradients_tau
        #     tau_update_list.append(gradients_tau)
        #     #print(self._compute_free_energy())

        print(self._bin_times[2] - self._bin_times[1], tau_update_list, self._kernel_param["tau"])
        print(self._compute_free_energy())
        # init_params = self._kernel_param["tau"]
        # print(gradient_func(init_params))
        # print(gradient_func(init_params+100))



    def _gather_data_Cupdate(self, numerator, denominator, covars_term):
        """
        """
        self._logging_C_update_numerator = numerator
        self._logging_C_update_denominator = denominator
        self._logging_C_update_covars_term = covars_term


    def _make_stats_Cupdate(self):
        """
        orth_columns_numer

        b_11,b_12,b_21,b_22
        b_11_inv,b_12_inv,b_21_inv,b_22_inv
        b_11_S,b_12_S,b_21_S,b_22_S

        diff_d : the absolute error between d and the original d
        orth_d : the mean angle between d and the two columns of C

        """
        stats_list = []

        orth_columns_numer = orth_res = get_orthogonality_score(self._logging_C_update_numerator, False)[0]
        stats_list.append(orth_columns_numer)

        orth_columns_den = orth_res = get_orthogonality_score(np.linalg.inv(self._logging_C_update_denominator), False)[0]
        stats_list.append(orth_columns_den)

        #true_denom = np.linalg.inv(self._logging_C_update_denominator)

        for i in range(0,3):
            if i == 0:
                matr = np.linalg.inv(self._logging_C_update_denominator)
            elif i == 1:
                matr = (self._logging_C_update_denominator)
            elif i == 2:
                matr = self._logging_C_update_covars_term

            stats_list.append(matr[0,0])
            stats_list.append(matr[0,1])
            stats_list.append(matr[1,0])
            stats_list.append(matr[1,1])

        diff_d = np.mean(np.abs(self._d_bias - self._meta_data["d_bias"]))
        stats_list.append(diff_d)

        orth_d = np.mean([ (180 / np.pi) * np.arccos(np.abs(np.dot(self._C_matrix[:,i], self._d_bias ) / (np.linalg.norm(self._C_matrix[:,i])* np.linalg.norm(self._d_bias)) )) for i in [0,1]])
        stats_list.append(orth_d)


        stats_string = ""
        for st in stats_list:
            stats_string += "," + str(st)

        return stats_string







