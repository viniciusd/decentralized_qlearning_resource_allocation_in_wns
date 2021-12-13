# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)

# EXPERIMENT EXPLANATION:
# By using a simple grid of 4 WLANs sharing 2 channels, we want to test several values of
# gamma, alpha and initial epsilon to evaluate the performance of
# Q-learning for each of them. We compare the obtained results with the
# optimal configurations in terms of proportional fairness and aggregate
# throughput.
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np

from generate_network_3d import GenerateNetwork3D
from qlearning_method import QlearningMethod
import auxiliary_methods as utils
import throughput_calculation as tput


print('****************************************************************************************')
print('* Implications of Decentralized Learning Resource Allocation in WNs                    *')
print('* Copyright (C) 2017-2022, and GNU GPLd, by Francesc Wilhelmi                          *')
print('* GitHub: https://github.com/wn-upf/Decentralized_Qlearning_Resource_Allocation_in_WNs *')
print('`****************************************************************************************')

RUN_EXPERIMENT = False # Should re-run the experiment or just laod pre-saved data?
print_info = False     # print info after implementing Q-learning

# GLOBAL VARIABLES
n_WLANs = 4                    # Number of WLANs in the map
n_agents = 4                   # Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 1000 # 10000
MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME//2 + 1
MAX_LEARNING_ITERATIONS = 1    # Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 100               # Number of TOTAL repetitions to take the average

# WLAN object to be modified for each number of coexistent nodes
nChannels = 2              # Number of available channels (from 1 to NumChannels)
noise = -100               # Floor noise (dBm)

# Definition of actions:
actions_ch = np.arange(1, nChannels+1)   # nChannels possible channels
actions_cca = np.array([-82])            # One CCA level (dBm) -> meaningless (all interferences are considered)
actions_tpc = np.array([5, 10, 15, 20])  # 4 different levels of TPC (dBm)

# DEFINE THE VARIABLES TO BE USED
# Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
possible_actions = np.arange(len(actions_ch)*len(actions_cca)*len(actions_tpc))
# Total number of actions
K = len(possible_actions)
# All possible combinations of configurations for the entire scenario
possible_comb = utils.allcomb(possible_actions,possible_actions,possible_actions,possible_actions)

# Q-learning parameters
initial_epsilon = 1    # Initial Exploration coefficient
updateMode = 1         # 0: epsilon = initial_epsilon / t ; 1: epsilon = epsilon / sqrt(t)

gamma_epsilon_pairs = [(.95, 1), (0.5, 1), (.05, 1), (.95, .5), (.5, .5), (.05, .5)]
alpha = np.arange(0, 1.1, 0.1)                 # Learning Rate
alpha = np.arange(0, 1, 0.1)                 # Learning Rate

# Exploration approach
# seed = datetime.now()
seed = 1
print("Random seed", seed)
random.seed(seed)

experiment_name = Path(__file__).stem


def run_experiment():
    print('-----------------------')
    print('EXPERIMENT 1-1: finding the best parameters (Q-learning)')
    print('-----------------------')

    # Setup the scenario: generate WLANs and initialize states and actions
    wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 2, 0) # SAFE CONFIGURATION
    #DrawNetwork3D(wlan)
        
    # ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE) 
    tpt_evolution_per_wlan_ql = [np.array([]) for _ in range(TOTAL_ROUNDS)]

    avg_tpt_evolution_ql = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    fairness_evolution = [np.array([]) for _ in range(TOTAL_ROUNDS)]

    # avg_tpt_experienced_ql = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    # std_tpt_experienced_ql = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    # aggregate_tpt = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    # avg_fairness_experienced = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    avg_tpt_experienced_ql = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma_epsilon_pairs)))
    std_tpt_experienced_ql = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma_epsilon_pairs)))
    aggregate_tpt = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma_epsilon_pairs)))
    avg_fairness_experienced = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma_epsilon_pairs)))

    for iteration in range(TOTAL_ROUNDS):

        print('------------------------------------')
        print('ROUND', (iteration+1), '/', TOTAL_ROUNDS)
        print('------------------------------------')

        for a in range(len(alpha)):
            for g_e in range(len(gamma_epsilon_pairs)):
                tpt_evolution_per_wlan_ql[iteration], qval = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS,
                                                gamma_epsilon_pairs[g_e][0], gamma_epsilon_pairs[g_e][1], alpha[a], updateMode,
                                                actions_ch, actions_cca, actions_tpc, noise, print_info)
                                            
                for j in range(MAX_CONVERGENCE_TIME):
                    avg_tpt_evolution_ql[iteration] = np.append(avg_tpt_evolution_ql[iteration], np.mean(tpt_evolution_per_wlan_ql[iteration][j]))
                    fairness_evolution[iteration] = np.append(fairness_evolution[iteration], utils.JainsFairness(tpt_evolution_per_wlan_ql[iteration][j]))

                avg_tpt_experienced_ql[iteration][a][g_e] = np.mean(avg_tpt_evolution_ql[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME])
                std_tpt_experienced_ql[iteration][a][g_e] = np.std(avg_tpt_evolution_ql[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME])

                aggregate_tpt[iteration][a][g_e] = np.mean(np.sum(tpt_evolution_per_wlan_ql[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME], axis=1))
                avg_fairness_experienced[iteration][a][g_e] = np.mean(fairness_evolution[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME])
    return wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt, avg_fairness_experienced


# PLOT THE RESULTS
def plot_experiment(wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt, avg_fairness_experienced):
    images_dir = f"images/{experiment_name}"
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    # Compute the optimal configuration to compare the approaches
    maximum_achievable_throughput = tput.computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise)
    # Find the best configuration for each WLAN and display it
    best_conf_tpt = []
    best_conf_fairness = []
    for i in range(len(maximum_achievable_throughput)):
        best_conf_tpt.append(np.sum(maximum_achievable_throughput[i]))
        best_conf_fairness.append(np.sum(np.log(maximum_achievable_throughput[i])))

    ix = np.argmax(best_conf_fairness)
    optimal_prop_fairness = best_conf_fairness[ix]
    print('---------------')
    print('Best proportional fairness: ', optimal_prop_fairness)
    print('Best configurations: ', possible_comb[ix,:])
    for i in range(n_WLANs):
        a, _, c = utils.val2indexes(possible_comb[ix,i], nChannels, len(actions_cca), len(actions_tpc))
        print('   * WLAN', i, ':')
        print('       - Channel:', a)
        print('       - TPC:', actions_tpc[c])

    ix2 = np.argmax(best_conf_tpt)
    optimal_agg_tpt = best_conf_tpt[ix2]
    print('---------------')
    print('Best aggregate throughput:', optimal_agg_tpt, 'Mbps')
    print('Best configurations:', possible_comb[ix2,:])
    for i in range(n_WLANs):
        a, _, c = utils.val2indexes(possible_comb[ix2,i], nChannels, len(actions_cca), len(actions_tpc))
        print('   * WLAN', i, ':')
        print('       - Channel:', a)
        print('       - TPC:', actions_tpc[c])

    # Compute the average results found for the aggregated throughput
    mean_aggregate_tpt = np.mean(aggregate_tpt, axis=0)
    print('mean_aggregate_tpt')
    print(mean_aggregate_tpt)
    # Compute the standard deviation for the aggregated throughput
    std_aggregate_tpt = np.std(aggregate_tpt, axis=0)
    print('std_aggregate_tpt')
    print(std_aggregate_tpt)
     
    # Plot the results
    l = []
    plt.figure()
    # axis([1 20 30 70]) ???
    for i in range(len(gamma_epsilon_pairs)):
        # plt.plot(alpha, mean_aggregate_tpt[:,i], r[i])
        plt.plot(alpha, mean_aggregate_tpt[:,i], '-o')
        l.append(r'$\gamma$ = '+str(gamma_epsilon_pairs[i][0])+' $\epsilon_{0}$ = '+str(gamma_epsilon_pairs[i][1]))
        plt.errorbar(alpha, mean_aggregate_tpt[:,i], yerr=std_aggregate_tpt[:,i])
        plt.xticks(alpha)
    plt.plot(alpha, optimal_agg_tpt*np.ones(len(alpha)),'--r')
    plt.plot(alpha, optimal_prop_fairness*np.ones(len(alpha)),'--r')
    plt.legend(l)
    plt.ylabel('Network Throughput (Mbps)')
    plt.xlabel(r'$\alpha$')
    plt.xlim((0, 1))
    plt.ylim((0, 1.2 * np.max(mean_aggregate_tpt)))
    plt.grid()
    plt.savefig(f"{images_dir}/mean_aggregate_tput.eps")


if RUN_EXPERIMENT:
    wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt, avg_fairness_experienced = run_experiment()

    data_dir = f"data/{experiment_name}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{data_dir}/wlan.npy", wlan)
    np.save(f"{data_dir}/avg_tpt_experienced_ql.npy", avg_tpt_experienced_ql)
    np.save(f"{data_dir}/std_tpt_experienced_ql.npy", std_tpt_experienced_ql)
    np.save(f"{data_dir}/aggregate_tpt.npy", aggregate_tpt)
    np.save(f"{data_dir}/aggregate_tpt.npy", aggregate_tpt)
    np.save(f"{data_dir}/avg_fairness_experienced.npy", avg_fairness_experienced)
else:
    wlan = np.load(f"data/{experiment_name}/wlan.npy", allow_pickle=True)
    avg_tpt_experienced_ql = np.load(f"data/{experiment_name}/avg_tpt_experienced_ql.npy", allow_pickle=True)
    std_tpt_experienced_ql = np.load(f"data/{experiment_name}/std_tpt_experienced_ql.npy", allow_pickle=True)
    aggregate_tpt = np.load(f"data/{experiment_name}/aggregate_tpt.npy", allow_pickle=True)
    avg_fairness_experienced = np.load(f"data/{experiment_name}/avg_fairness_experienced.npy", allow_pickle=True)

plot_experiment(wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt, avg_fairness_experienced)
