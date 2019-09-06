"""

This code is used to generate sythentic datasets using only Gaussian things and no non linearity.



inputs:

R: number of trials
N: number of neurons
T: duration of the trails (in ms) as said by different people, we assume that two spikes do not happen in less than 1 ms

K_alpha: number of mean latents
K_beta : number of condition specific latents
K_gamma: number of trial specific latents

time_wrapping : boolean to say if a random time wrapping is generated and used for each trial or not



How to interface it with the rest?
-> will only have the delta and not the firing it self
-> create a new meta data output "binned" : so for pre binned things, we then only generate the number of time points that we want
     25 for  example; but each 20ms (or around that) appart

-> change to have no diagonal element in the covariance matrix (no noise)



output:

a dataset of R trials, each being a N x T matrix.

"""
import numpy as np
import matplotlib.pyplot as plt
from utilis import *
# from matplotlib import rc

# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)




def generate_dataset_G(dataset_name, numb_trials, numb_neurons, total_duration, time_step, numb_conditions, K_alpha, K_beta, \
    K_gamma, R_mean, time_wrapping, plot_creation):


    # kernel params:
    tau_s_alpha = np.linspace(50,125,K_alpha)
    sigma_n_alpha = 0.00 * np.ones((K_alpha))

    tau_s_beta = np.linspace(50,125,K_beta)
    sigma_n_beta = 0.00 * np.ones((K_beta))

    tau_s_gamma = np.linspace(50,200,K_gamma)
    sigma_n_gamma = 0.00 * np.ones((K_gamma))

    if plot_creation:
        print(tau_s_alpha,tau_s_beta,tau_s_gamma)


    # Creating the variables implied from the parameters:
    # get the total number of latent processes:
    numb_total_latents = K_alpha + K_beta * numb_conditions + K_gamma * numb_trials
    numb_latents_per_trial = K_alpha + K_beta + K_gamma


    # generate the condition of each trial:
    trial_conditions = np.zeros((numb_trials))
    numb_trials_per_cond = numb_trials // numb_conditions

    if plot_creation:
        print(numb_trials_per_cond)

    curr_cond = 0
    for trial_i in range(1,numb_trials):
        if np.mod(trial_i,numb_trials_per_cond) == 0 and (numb_trials - trial_i) >= numb_trials_per_cond and curr_cond < numb_conditions - 1:
            curr_cond += 1

        trial_conditions[trial_i] = curr_cond

    if plot_creation:
        print(trial_conditions)


    # generate a random C :
    C_alpha = np.random.rand(numb_neurons,K_alpha) / 2
    C_beta = np.random.rand(numb_neurons,K_beta) / 2
    C_gamma = np.random.rand(numb_neurons,K_gamma) / 2

    print(get_angle_subspaces( C_gamma,  C_beta))
    print(get_orthogonality_score( C_gamma , False))
    print(get_orthogonality_score( C_beta , False))
    print(get_orthogonality_score( np.concatenate([C_beta , C_gamma ], axis=1)))
    # print()

    # generate a random d :
    d_bias = np.random.randn(numb_neurons)
    #d_bias = np.ones(numb_neurons)


    # generate the random processes:

    timestamps = np.arange(0, total_duration, time_step)
    numb_bin = len(timestamps)

    mean = np.zeros_like(timestamps)

    latent_alpha = np.zeros((K_alpha, numb_bin))
    latent_beta = np.zeros((numb_conditions, K_beta, numb_bin))
    latent_gamma = np.zeros((numb_trials, K_gamma, numb_bin))


    l_gram_alpha = []
    l_gram_beta = []
    l_gram_gamma = []

    gram = make_se_cov_matrix(timestamps)

    for i in range(0, K_alpha):
        gram = make_se_cov_matrix(timestamps, variance_n=sigma_n_alpha[i], tau=tau_s_alpha[i])
        l_gram_alpha.append(gram)

    for i in range(0, K_beta):
        gram = make_se_cov_matrix(timestamps, variance_n=sigma_n_beta[i], tau=tau_s_beta[i])
        l_gram_beta.append(gram)

    for i in range(0, K_gamma):
        gram = make_se_cov_matrix(timestamps, variance_n=sigma_n_gamma[i], tau=tau_s_gamma[i])
        l_gram_gamma.append(gram)


    # generate the latents with the gram matrices:

    for i in range(0, K_alpha):
        ys = np.random.multivariate_normal(mean, l_gram_alpha[i])
        latent_alpha[i,:] = ys

    for i in range(0, K_beta):
        for cond_i in range(0, numb_conditions):
            ys = np.random.multivariate_normal(mean, l_gram_beta[i])
            latent_beta[cond_i,i,:] = ys

    for i in range(0, K_gamma):
        for trial_i in range(0, numb_trials):
            ys = np.random.multivariate_normal(mean, l_gram_gamma[i])
            latent_gamma[trial_i,i,:] = ys





    # generate the dataset:
    # - iterate through trials, and create each trial one after the other
    # - get the lambda for each timestamp and neuron :
    # 

    dataset = np.zeros((numb_trials,numb_neurons,numb_bin))

    for trial_i in range(0,numb_trials):

        this_condition = trial_conditions[trial_i].astype(int)

        # adding the mean values:
        dataset[trial_i,:,:] += np.matmul(C_alpha, latent_alpha)

        # adding the condition specific values:
        dataset[trial_i,:,:] += np.matmul(C_beta, latent_beta[this_condition,:,:])

        # adding the trial specific values:
        dataset[trial_i,:,:] += np.matmul(C_gamma, latent_gamma[trial_i,:,:])

        # adding the trial specific values:
        dataset[trial_i,:,:] = ( d_bias.transpose() + dataset[trial_i,:,:].transpose() ).transpose()


    # adding the R noise:
    R_diagonal = np.array([np.random.random(1)[0]*R_mean for i in range(0,numb_neurons)])
    # could do it better

    for trial_i in range(0,numb_trials):
        for bin_i in range(0, numb_bin):
            for neuron_i in range(0, numb_neurons):
                t = 9
                dataset[trial_i,neuron_i,bin_i] += np.random.randn(1) * np.sqrt(R_diagonal[neuron_i])


    if plot_creation:


        # just showing what is being generated:
        # for i in range(0, K_alpha):
        #     plt.plot(latent_alpha[i,:])
        # plt.figure()
        # for i in range(0, K_beta):
        #     for cond_i in range(0, numb_conditions):
        #         plt.plot(latent_beta[cond_i,i,:])
        # plt.figure()
        # for i in range(0, K_gamma):
        #     for trial_i in range(0, numb_trials):
        #         plt.plot(latent_gamma[trial_i,i,:])
        fig, axes = plt.subplots( numb_trials, K_gamma,figsize=(10, 10))
        plt.subplots_adjust(hspace=0.3)
        plt.rcParams.update({'font.size': 12})

        for trial_i in range(0,  numb_trials - 0):
            for lat_i in range(0, K_gamma):
                axes[trial_i, lat_i].plot(timestamps, latent_gamma[trial_i,lat_i,:])
                axes[2, lat_i].set_xlabel("Timesteps")
                axes[trial_i, lat_i].set_title("Trial "+str(trial_i+1)+", latent "+str(lat_i+1)+", time constant "+str((tau_s_gamma[lat_i]).astype(int)))
                #plt.hold(True)
                #axes[1].set_ylim([-2.5, 2.5])
            # for lat_i in range(0,self._numb_trial_lat):
            #     axes[trial_i, lat_i].plot(self._bin_times,self._found_latents[trial_i,lat_i,:])

        # fig.savefig("../MA/figures/c4s1_sample_latents.pdf", bbox_inches='tight')
        plt.show()

        plt.imshow(dataset[0,:,:])
        # plt.xticks([20,40],["200","400"])
        plt.title("The heat map of trial 1")
        plt.ylabel("Neuron indices")
        plt.xlabel("Timesteps")
        # plt.savefig("../MA/figures/c4s1_sample_neuron_dataset.pdf", bbox_inches='tight')
        plt.show()



    # save everything in files:
    #   - create a metadata dictionary and save it
    #   - save the dataset


    # meta data
    my_dictio = {"numb_trials" : numb_trials}
    my_dictio["numb_neurons"] = numb_neurons
    my_dictio["total_duration"] = total_duration
    my_dictio["binned_with"] = time_step
    my_dictio["numb_conditions"] = numb_conditions
    my_dictio["trial_conditions"] = trial_conditions
    my_dictio["K_alpha"] = K_alpha
    my_dictio["K_beta"] = K_beta
    my_dictio["K_gamma"] = K_gamma
    my_dictio["time_wrapping"] = time_wrapping
    my_dictio["generated"] = True

    my_dictio["tau_s_alpha"] = tau_s_alpha
    my_dictio["tau_s_beta"] = tau_s_beta
    my_dictio["tau_s_gamma"] = tau_s_gamma

    my_dictio["C_alpha"] = C_alpha
    my_dictio["C_beta"] = C_beta
    my_dictio["C_gamma"] = C_gamma
    my_dictio["d_bias"] = d_bias

    my_dictio["latent_alpha"] = latent_alpha
    my_dictio["latent_beta"] = latent_beta
    my_dictio["latent_gamma"] = latent_gamma
    
    # my_dictio["S_full_covars"] = S_full_covars
    my_dictio["gen_R"] = np.diag(R_diagonal)

    saving_all = True

    if saving_all:
        save_obj(my_dictio,dataset_name + '_meta')
        save_obj(dataset,dataset_name)
        #save_obj(all_latents,dataset_name + '_latents')


    C = np.concatenate([C_alpha , C_beta , C_gamma ], axis=1)
    
    return C





# Defining all the parameters:

numb_trials = 25#11
numb_neurons = 100
total_duration = 510    # (in ms)
time_step = 10     # (in ms)

numb_conditions = 2

K_alpha = 2
K_beta  = 0
K_gamma = 2

R_mean = 1/6

if np.mod(numb_trials, numb_conditions) != 1:    
        raise EnvironmentError
# not the correct error, but the point is that we assume that we have the same number of classes


# generate dataset name:    (right now set, because test runs)
dataset_name = "G_neuron_data_" + str(K_alpha) + str(K_beta) + str(K_gamma) + "_1"
# dataset_name = "G_neuron_data_demo_data_1_002"

# 
time_wrapping = False
# for now we do not add time wrapping to the code

if True:
    generate_dataset_G(dataset_name, numb_trials, numb_neurons, total_duration, time_step, numb_conditions, K_alpha,\
         K_beta, K_gamma, R_mean, time_wrapping, False)





# # just some helper function
# def rbf_kernel(x1, x2, variance = 1):
#     return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))

# def se_kernel(x1, x2, variance_f = 1, variance_n = 0.001, tau = 100):
#     return variance_f * np.exp(-1 * ((x1-x2) ** 2) / (2*(tau**2)) ) + variance_n * (x1 == x2)

# def make_cov_matrix(xs):
#     return [[se_kernel(x1,x2) for x2 in xs] for x1 in xs]

