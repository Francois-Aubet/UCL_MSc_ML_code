"""

This is just a random thing to try and use PCA to extract neural paths.

"""

from algorithms.Algorithm import Algorithm
import numpy as np
import sklearn.decomposition



class PCA(Algorithm):
    """ PCA """

    algName = "PCA"



    def __init__(self, dataset, meta_data, bin_times, numb_latents):
        """ The initialization. """

        super().__init__(dataset, meta_data)

        self._numb_latents = numb_latents



    def extract_neural_path(self, verbose):
        """ 
        
        @:return: the neural path latents
        """

        latents = np.zeros((self._meta_data["numb_trials"],self._numb_latents,len(self._meta_data["bin_times"])))

        for trail_i in range(0,self._meta_data["numb_trials"]):
            pca = sklearn.decomposition.PCA(n_components=self._numb_latents)
            pca.fit(np.transpose(self._dataset[trail_i,:,:]))

            latents[trail_i,:,:] = np.transpose(pca.transform(np.transpose(self._dataset[trail_i,:,:])))

        return latents


