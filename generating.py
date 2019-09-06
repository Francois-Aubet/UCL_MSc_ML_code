"""

This code is used to generate sythentic datasets of neuron firings.



inputs:

R: number of trials
N: number of neurons
T: duration of the trails (in ms) as said by different people, we assume that two spikes do not happen in less than 1 ms

K_alpha: number of mean latents
K_beta : number of condition specific latents
K_gamma: number of trial specific latents

time_wrapping : boolean to say if a random time wrapping is generated and used for each trial or not




output:

a dataset of R trials, each being a N x T matrix.

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utilis import *




def save_obj(obj, name ):
    with open('datasets/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def generate_dataset(dataset_name, numb_trials, numb_neurons, total_duration, numb_conditions, K_alpha, K_beta, K_gamma, time_wrapping, plot_creation):

    # kernel params:
    tau_s_alpha = np.linspace(50,175,K_alpha)
    if plot_creation:
        print(tau_s_alpha)
    sigma_n_alpha = 0.001 * np.ones((K_alpha))


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
    C_matrix = np.random.rand(numb_neurons,numb_total_latents) / 2

    # generate a random d :
    d_bias = np.random.randn(numb_neurons)


    # generate the random processes:

    timestamps = np.arange(0, total_duration, 1)

    mean = [0 for t in timestamps]

    gram = make_se_cov_matrix(timestamps)


    all_latents = np.zeros((numb_total_latents,total_duration))

    for i in range(0, K_alpha):
        #gram = make_se_cov_matrix(timestamps, variance_n=sigma_n_alpha[i], tau=tau_s_alpha[i])
        ys = np.random.multivariate_normal(mean, gram)
        all_latents[i,:] = ys

    for i in range(K_alpha, numb_total_latents):
        ys = np.random.multivariate_normal(mean, gram)
        all_latents[i,:] = ys
        #all_latents.extend([timestamps, ys, "k"])

    if plot_creation:
        print(np.array((all_latents)).shape )



    # generate the dataset:
    # - iterate through trials, and create each trial one after the other
    # - get the lambda for each timestamp and neuron :
    # 

    dataset = np.zeros((numb_trials,numb_neurons,total_duration))

    for trial_i in range(0,numb_trials):

        # adding the mean values:
        for latent_k in range(0,K_alpha):
            dataset[trial_i,:,:] += np.outer(C_matrix[:,latent_k], all_latents[latent_k,:])


        # adding the condition specific values:
        this_condition = trial_conditions[trial_i]

        start_cond_latent = np.int(K_alpha + this_condition * K_beta)
        end_cond_latent = np.int(K_alpha + (this_condition + 1) * K_beta)
        #print(start_cond_latent,end_cond_latent)

        for latent_k in range(start_cond_latent,end_cond_latent):
            dataset[trial_i,:,:] += np.outer(C_matrix[:,latent_k], all_latents[latent_k,:])
            #print(latent_k)


        # adding the trial specific values:
        start_cond_latent = np.int(K_alpha + K_beta * numb_conditions + K_gamma * trial_i)
        end_cond_latent = np.int(K_alpha + K_beta * numb_conditions + K_gamma * (trial_i+1) ) 
        #print(start_cond_latent,end_cond_latent)

        for latent_k in range(start_cond_latent,end_cond_latent):
            dataset[trial_i,:,:] += np.outer(C_matrix[:,latent_k], all_latents[latent_k,:])
            #print(latent_k)


    # applying the non linearity to get only positive poisson coef:
    #dataset = np.exp(dataset)
    dataset = np.square(dataset)

    dataset = np.random.poisson(dataset)
    dataset = (dataset >= 1).astype(int)

    all_latents = np.square(all_latents)

    if plot_creation:

        print(np.mean(dataset))

        print(np.sum(dataset[trial_i,:,:] > 1))
        print(np.sum(dataset[trial_i,:,:] <= 1))

        print(np.sum(dataset[trial_i,:,:] > 1))
        print(np.sum(dataset[trial_i,:,:] <= 1))

        plt.imshow(dataset[trial_i,:,:])
        plt.show()

        print(numb_total_latents)

        # just showing what is being generated:
        for i in range(0, K_alpha):
            plt.plot(all_latents[i,:])
        plt.figure()
        for i in range(K_alpha, K_alpha + K_beta * numb_conditions):
            plt.plot(all_latents[i,:])
        plt.figure()
        for i in range(K_alpha + K_beta * numb_conditions, numb_total_latents):
            plt.plot(all_latents[i,:])
        plt.show()


    # save everything in files:
    #   - create a metadata dictionary and save it
    #   - save the dataset


    # meta data
    my_dictio = {"numb_trials" : numb_trials}
    my_dictio["numb_neurons"] = numb_neurons
    my_dictio["total_duration"] = total_duration
    my_dictio["numb_conditions"] = numb_conditions
    my_dictio["trial_conditions"] = trial_conditions
    my_dictio["K_alpha"] = K_alpha
    my_dictio["K_beta"] = K_beta
    my_dictio["K_gamma"] = K_gamma
    my_dictio["tau_s_alpha"] = tau_s_alpha
    my_dictio["time_wrapping"] = time_wrapping
    my_dictio["generated"] = True

    saving_all = True

    if saving_all:
        save_obj(my_dictio,dataset_name + '_meta')
        save_obj(dataset,dataset_name)
        save_obj(all_latents,dataset_name + '_latents')

    return C_matrix




# Defining all the parameters:

numb_trials = 10
numb_neurons = 100
total_duration = 400    # (in ms)

numb_conditions = 3

K_alpha = 1
K_beta  = 0
K_gamma = 0


# generate dataset name:    (right now set, because test runs)
dataset_name = "neuron_data0"

# 
time_wrapping = False
# for now we do not add time wrapping to the code

if False:
    generate_dataset(dataset_name, numb_trials, numb_neurons, total_duration, numb_conditions, K_alpha, K_beta, K_gamma, time_wrapping, True)











# # just some helper function
# def rbf_kernel(x1, x2, variance = 1):
#     return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))

# def se_kernel(x1, x2, variance_f = 1, variance_n = 0.001, tau = 100):
#     return variance_f * np.exp(-1 * ((x1-x2) ** 2) / (2*(tau**2)) ) + variance_n * (x1 == x2)

# def make_cov_matrix(xs):
#     return [[se_kernel(x1,x2) for x2 in xs] for x1 in xs]

