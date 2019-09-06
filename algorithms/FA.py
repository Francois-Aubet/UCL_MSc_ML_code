"""

This is just a random thing to try and use Factor Analysis (FA) to extract neural paths.

"""

from algorithms.Algorithm import Algorithm
import numpy as np
import sklearn.decomposition



class FA(Algorithm):
    """ FA """

    algName = "FA"



    def __init__(self, dataset, meta_data, bin_times, numb_latents):
        """ The initialization. """

        super().__init__(dataset, meta_data)

        self._numb_latents = numb_latents



    def extract_neural_path_simple(self, verbose):
        """ 
        
        @:return: the neural path latents
        """

        latents = np.zeros((self._meta_data["numb_trials"],self._numb_latents,len(self._meta_data["bin_times"])))

        for trail_i in range(0,self._meta_data["numb_trials"]):
            fa = sklearn.decomposition.FactorAnalysis(n_components=self._numb_latents)
            fa.fit(np.transpose(self._dataset[trail_i,:,:]))

            latents[trail_i,:,:] = np.transpose(fa.transform(np.transpose(self._dataset[trail_i,:,:])))

        return latents




    def extract_neural_path(self, verbose):
        """ 
        
        @:return: the neural path latents
        """

        self._numb_trials, self._numb_neurons, self._numb_bins = self._dataset.shape
        latents = np.zeros((self._numb_trials,self._numb_latents,len(self._meta_data["bin_times"])))


        temp_data = self._dataset.transpose([0,2,1]).reshape((self._numb_trials*self._numb_bins,self._numb_neurons))


        fa = sklearn.decomposition.FactorAnalysis(n_components=self._numb_latents)
        fa.fit(temp_data)


        # save the params
        self._C_matrix = np.zeros((self._numb_neurons, self._numb_latents))
        self._C_matrix  = np.transpose(fa.components_)

        # perturbation_matr = np.array([[0.8,0.4,0],[0.4,-0.8,0],[0,0,1]])
        # self._C_matrix = self._C_matrix.dot(perturbation_matr)
        
        self._C_matrix_big = self._C_matrix
        # for lat_k in range(0, self._numb_latents):
        #     if np.mean(C_cand[:,lat_k]) > 0.001:
        #         self._C_matrix[:,lat_k] = C_cand[:,lat_k]
        # self._R_noise = np.diag(fa.noise_variance_)

        self._R_noise = np.diag(fa.noise_variance_)
        self._d_bias = np.zeros((self._numb_neurons ))
        
        for trail_i in range(0,self._numb_trials):

#            latents[trail_i,:,:] = np.transpose(fa.transform(np.transpose(self._dataset[trail_i,:,:])))
            # get latent:

            # get reconstruction:
            latent = np.linalg.inv( np.matmul(np.transpose(self._C_matrix), self._C_matrix ) )
            latent = latent.dot(np.matmul(np.transpose(self._C_matrix), self._dataset[trail_i,:,:]) )


            pred = np.matmul(self._C_matrix,latent)
            latents[trail_i,:,:] = latent#self._C_matrix.dot()

        return latents

    def _recover_path_for_model(self):
        """
        Used to obtain the latent path after the fitting of the model.

        This is a method that is called after the EM iterations.
        
        """
        raise NotImplementedError("Must override method")