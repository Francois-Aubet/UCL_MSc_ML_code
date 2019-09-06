
"""
A file with a couple of helping functions usefull for most algorithm.


"""
#import numpy as np
import autograd.numpy as np

from scipy.linalg import hadamard, subspace_angles, orth

import pickle


def save_obj(obj, name ):
    with open('datasets/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('datasets/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)





def rbf_kernel(x1, x2, tau = 1):
    return np.exp(-1 * (np.square(x1 - x2)) / (2*tau))

def se_kernel(x1, x2, variance_f = 1, variance_n = 0.001, tau = 100):
    return variance_f * np.exp(-1 * (np.square(x1 - x2) / (2*(tau**2)) ) ) + variance_n * (x1 == x2)

def make_cov_matrix(xs, kernel_func):
    return np.array([[kernel_func(x1,x2) for x2 in xs] for x1 in xs])

def make_se_cov_matrix(xs, variance_n = 0.001, tau = 100):
    return np.array([[se_kernel(x1,x2, 1 - variance_n, variance_n, tau) for x2 in xs] for x1 in xs])

def make_K(T1, T2, kernel_func, param, k):

    kappa = np.outer(T1,T2)
    len_t1, len_t2 = kappa.shape

    # creating matrices from the input so that we can feed them directly in the kernel function
    temp1 = np.repeat(T1.reshape(len_t1,1), len_t2, 1)
    temp2 = np.repeat(T2.reshape(1,len_t2), len_t1, 0)

    if kernel_func is se_kernel:
        # call the function with the good parameters:
        kappa = se_kernel(temp1, temp2, 1 - param["sigma_n"][k], param["sigma_n"][k], param["tau"][k])

    elif kernel_func is rbf_kernel:
        kappa = rbf_kernel(temp1, temp2, param["tau"][k])

    return kappa



def get_orthogonality_score(C_matrix, verbose = True):
    """
    Gets the angle between each subspace and the other ones.

    Note the we leave the diagonal as zeros, because the angles are 1 anyway
    And it helps to have a more representative mean.
    """

    in_degree = True

    len_1, len_2 = C_matrix.shape
    orthogonality_matrix = np.zeros(( len_2 , len_2 ))

    for lat_i in range(0, len_2):
        for lat_j in range(lat_i+1, len_2):
            angle = np.dot(C_matrix[:,lat_i], C_matrix[:,lat_j] ) / (np.dot(np.linalg.norm(C_matrix[:,lat_i]), np.linalg.norm(C_matrix[:,lat_j])) )
            orthogonality_matrix[lat_i, lat_j] = np.arccos(np.abs(angle))
            orthogonality_matrix[lat_j, lat_i] = np.arccos(np.abs(angle))

    if in_degree:
        orthogonality_matrix = 180 * orthogonality_matrix / np.pi

    mean_per_sub_space = np.sum(np.abs(orthogonality_matrix),1)/(len_2-1)

    glob_mean = np.mean(mean_per_sub_space)

    try:
        all_non_diag = orthogonality_matrix.flatten()
        all_non_diag = all_non_diag[np.nonzero(all_non_diag)]

        tenth_percentil = np.percentile(all_non_diag, 25)
        ninetith_percentil = np.percentile(all_non_diag, 75)


        small_avr = np.average(all_non_diag, weights= (all_non_diag <= tenth_percentil).astype(int))
        high_avr = np.average(all_non_diag, weights= (all_non_diag >= ninetith_percentil).astype(int))
    except:
        small_avr = glob_mean
        high_avr = glob_mean

    if verbose:
        print(np.around(orthogonality_matrix,2))
        print("Mean abs angle per subspace: ",  mean_per_sub_space)
        print("Mean abs angle overall: ", glob_mean)
        #print("Std abs angle overall: ", np.std(mean_per_sub_space))

       # print(small_avr, high_avr)
    if len_2 <= 1:
        glob_mean = small_avr = high_avr = 0

    return glob_mean, small_avr, high_avr




def get_angle_subspaces(U, V, verbose = True):
    """
    Gets the angle between two subspaces.
    """

    use_simple_meth = True

    U = orth(U)
    V = orth(V)

    len_1_u, len_2_u = U.shape
    len_1_v, len_2_v = V.shape

    if len_2_u > len_2_v:
        a = U.copy()
        U = V.copy()
        V = a.copy()

        len_1_u, len_2_u = U.shape
        len_1_v, len_2_v = V.shape


    if use_simple_meth:
        print(180 * subspace_angles(U, V) / np.pi)#180 * orthogonality_matrix / np.pi
        angle = np.rad2deg(max(subspace_angles(U, V)))
    
    else:

        # normalizing subspaces:
        for i in range(0, len_2_u):
            U[:,i] *= (1 / np.linalg.norm(U[:,i]) )
        for i in range(0, len_2_v):
            V[:,i] *= (1 / np.linalg.norm(V[:,i]) )


        # constuct M
        M = np.zeros((len_2_u,len_2_v))

        for i in range(0, len_2_u):
            for j in range(0, len_2_v):
                M[i,j] = np.dot(V[:,j], U[:,i])
        
        MMt = M.dot(M.transpose())

        det_MMt = np.linalg.det(MMt)

        # construct u
        uu = np.zeros((len_2_u,len_2_u))

        for i in range(0, len_2_u):
            for j in range(0, len_2_u):
                uu[i,j] = np.dot(U[:,j], U[:,i])
                #print(i,j,np.dot(V[:,j], U[:,i]))
        
        det_uu = np.linalg.det(uu)

        cos_square_theta = det_MMt / (det_uu + 1e-14)
        # print("cos_square_theta",cos_square_theta)
        theta = np.sqrt(cos_square_theta)

        # print("theta",theta)

        # print("angle",np.arccos(theta))
        angle = np.rad2deg(np.arccos(theta))

    if verbose:
        print("angle",angle)

    return angle


def fnorm(matrix):
    """
    get the F norm.
    """
    U, S, V = np.linalg.svd(matrix)
    S = np.diag(S)
    fnorm_s = np.sqrt(np.trace( np.square(S) ))

    return fnorm_s

def is_pos_def(x):
    # print(np.linalg.eigvals(x)[np.linalg.eigvals(x) < 0] * 10**(12))
    return np.all(np.linalg.eigvals(x) > - 10.0**(-9)).astype(int)

def stats_of_covariance(cov_matrix, K_numb_latents, return_as_string = False, verbose = False, simple_return = False):
    """
    get: S_mean_abs,S_mean_diag,S_mean_off,S_fnorm,S_fnorm_diag,S_fnorm_off

    """
    fnorm_cov = fnorm(cov_matrix)
    
    sum_diag = 0
    sum_total = np.sum(np.abs(cov_matrix))
    numb_total_elements = ((cov_matrix.shape[0])**2)
    T = cov_matrix.shape[0] // K_numb_latents
    numb_diag_elements = ((T)**2 * K_numb_latents)
    mean = sum_total / numb_total_elements

    if simple_return:
        return str(mean) + "," + str(fnorm_cov)

    sum_diag_fnorm = 0

    # get the diagonal terms
    sum_diag = 0
    for lat_k in range(0, K_numb_latents):
        sum_diag += np.sum(np.abs(cov_matrix[lat_k*T:lat_k*T+T, lat_k*T:lat_k*T+T]))
        sum_diag_fnorm += fnorm(cov_matrix[lat_k*T:lat_k*T+T, lat_k*T:lat_k*T+T])

    mean_diag = sum_diag / numb_diag_elements
    mean_off = (sum_total - sum_diag) / (numb_total_elements - numb_diag_elements)

    diag_fnorm = sum_diag_fnorm / lat_k

    if return_as_string:
        return str(mean) + "," + str(mean_diag) + "," + str(mean_off) + "," + str(fnorm_cov) + "," + str(diag_fnorm)
    else:
        return mean, mean_diag, mean_off, fnorm_cov, diag_fnorm













## This is just code to test the creation of kernel matrices


# T1 = np.arange(1,5)
# T2 = np.arange(2,4)

# print(T1)
# print(T2)


# kappa = np.outer(T1,T2)
# len_t1, len_t2 = kappa.shape
# kappa = np.zeros((len_t1, len_t2))



# print(len_t1, len_t2)

# #temp1 = np.transpose(np.kron(np.ones((len_t2)), T1)).reshape(len_t1, len_t2)
# temp1 = np.repeat(T1.reshape(len_t1,1), len_t2, 1)
# temp2 = np.repeat(T2.reshape(1,len_t2), len_t1, 0)

# print(temp1)
# print(temp2)

# print(temp1 - temp2)
# #np.array(variance_f * np.exp(-1 * ((x1-x2) ** 2) / (2*(tau**2)) ) + variance_n * (x1 == x2))

# variance_f = 1
# variance_n = 0.001
# tau = 100

# matr = variance_f * np.exp(-1 * ( np.square(temp1 - temp2) / (2*np.square(tau) ) ) ) #+ variance_n * (x1 == x2))
# print(np.square(temp1 - temp2))
# print(np.square(temp1 - temp2) / (2*np.square(tau) ))
# print( matr )

# for  i in range(0,len_t1):
#     for j in range(0, len_t2):
#         kappa[i,j] = se_kernel(T1[i],T2[j])
# print(kappa)











# print(np.array([[se_kernel(x1,x2) for x2 in np.array([T1])] for x1 in np.array([T2])]).shape)
# print(kappa.shape, T1)
















