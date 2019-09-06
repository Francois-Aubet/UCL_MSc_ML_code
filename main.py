"""
This is the file that lunches things and lets decide options.

Here we create an algorithm handler and pass it the classes we want and see.


"""
from AlgoHandler import AlgoHandler
# we import all the algorithms
from algorithms.PCA import PCA
from algorithms.FA import FA
from algorithms.GPFA import GPFA
from algorithms.GPFA_sv import GPFA_sv
from algorithms.GPFA_sv_lr import GPFA_sv_lr
from algorithms.GPFA_sv_mc import GPFA_sv_mc
from algorithms.GPFA_sv_mc_lr import GPFA_sv_mc_lr



# the name of the dataset that is to be used from the "./datsets" folder

dataset_name = "G_neuron_data_002_1"


# Any of the algorithm class
algo = GPFA_sv_lr


# define number of latents to extract :
numb_shared_lat = 0
numb_grouped_lat = 0
numb_trial_lat = 2

# if bined, define the duration of the time bin, in milliseconds:
bin_width = 25
max_EM_iterations = 10
numb_inducing = 15#37
learn_kernel_params = False
numb_test_trials = 0
plot_free_energy = 1
save_ortho_step = 15



print("Fitting the ", algo.algName, " on ", dataset_name)

algo_handler = AlgoHandler(algo, dataset_name, numb_shared_lat, numb_grouped_lat, numb_trial_lat, bin_width, max_EM_iterations,
            numb_inducing, learn_kernel_params, numb_test_trials, plot_free_energy, save_ortho_step)

algo_handler.initialise_algo_to_gen_param()

# algo_handler._algo.use_MF_approximation = True
# algo_handler._algo._use_gradient_E_steps = True
# algo_handler._algo._use_lr_E_steps = False




# algo_handler.plot_binned()
algo_handler.extract_path()

algo_handler.plot_path()

# algo_handler._algo.linear_response_correction()

for i in range(0,3):

    # algo_handler.get_orthogonality_scores()

    # algo_handler.compared_generating_and_found()

    # print(algo_handler._algo.get_Lm_for_comparison_porpuses())

    # algo_handler.compute_dataset_errors_from_C()
    algo_handler.get_current_Free_KL_like()

    algo_handler._algo._perturbate_S()

    algo_handler.get_current_Free_KL_like()

    # algo_handler.compare_found_covars_with_GPFA()

    # print(algo_handler._algo.compute_KL_to_true_posterior_full())

    # print(algo_handler._algo.compute_likelihood())
    # print(algo_handler._algo.get_Lm_for_comparison_porpuses())
    # print(algo_handler._algo.compute_the_expected_log_marginal())


    # algo_handler._algo.linear_response_correction()
    if i == 0:
        algo_handler._algo.linear_response_correction()

        # algo_handler._algo._E_step_full()
    elif i == 1:
        algo_handler._algo.update_model_parameters_after_LR()




