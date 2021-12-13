# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)

# EXPERIMENT EXPLANATION:
# By using a simple grid of 4 WLANs sharing 2 channels, we want to test several values of
# gamma and alpha to see their relation.
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pylab as plt
from matplotlib import cm
import numpy as np

from generate_network_3d import GenerateNetwork3D
from qlearning_method import QlearningMethod
import auxiliary_methods as utils

print('****************************************************************************************')
print('* Implications of Decentralized Learning Resource Allocation in WNs                    *')
print('* Copyright (C) 2017-2022, and GNU GPLd, by Francesc Wilhelmi                          *')
print('* GitHub: https://github.com/wn-upf/Decentralized_Qlearning_Resource_Allocation_in_WNs *')
print('****************************************************************************************')


RUN_EXPERIMENT = False # Should re-run the experiment or just laod pre-saved data?
print_info = False     # print info after implementing Q-learning

# DEFINE THE VARIABLES TO BE USED

#  GLOBAL VARIABLES
n_WLANs = 4                    # Number of WLANs in the map
n_agents = 4                   # Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 1000 #10000
MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME//2 + 1
MAX_LEARNING_ITERATIONS = 1    # Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 100             # Number of TOTAL repetitions to take the average

# WLAN object to be modified for each number of coexistent nodes
nChannels = 2              # Number of available channels (from 1 to NumChannels)
noise = -100               # Floor noise (dBm)

# Definition of actions:
actions_ch = np.arange(1, nChannels+1)       # nChannels possible channels
actions_cca = np.array([-82])            # One CCA level (dBm) -> meaningless (all interferences are considered)
actions_tpc = np.array([5, 10, 15, 20])     # 4 different levels of TPC (dBm)

# Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
possible_actions = np.arange(len(actions_ch)*len(actions_cca)*len(actions_tpc))
# Total number of actions
K = len(possible_actions)
possible_comb = utils.allcomb(possible_actions,possible_actions,possible_actions,possible_actions)

# Q-learning parameters
initial_epsilon = 1    # Initial Exploration coefficient
updateMode = 1         # 0: epsilon = initial_epsilon / t  1: epsilon = epsilon / sqrt(t) 

gamma = np.arange(0, 1, .1)         # Discount factor
alpha = np.arange(0, 1, .1)         # Learning Rate

# Exploration approach
# seed = datetime.now()
seed = 1
print("Random seed", seed)
random.seed(seed)

experiment_name = Path(__file__).stem


def run_experiment():
    print('-----------------------')
    print('EXPERIMENT 1-2: Alpha vs Gamma performance (Q-learning)')
    print('-----------------------')

    # Setup the scenario: generate WLANs and initialize states and actions
    wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 2, 0) # SAFE CONFIGURATION
    # DrawNetwork3D(wlan)
        
    # ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE)
    tpt_evolution_per_wlan_ql = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    avg_tpt_evolution_ql = [np.array([]) for _ in range(TOTAL_ROUNDS)]
    avg_tpt_experienced_ql = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma)))
    std_tpt_experienced_ql = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma)))
    aggregate_tpt = np.zeros((TOTAL_ROUNDS, len(alpha), len(gamma)))
        
    for iteration in range(TOTAL_ROUNDS):

        print('------------------------------------')
        print('ROUND', (iteration+1), '/', TOTAL_ROUNDS)
        print('------------------------------------')

        for a in range(len(alpha)):
            for g in range(len(gamma)):
                tpt_evolution_per_wlan_ql[iteration], qval = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS,
                        gamma[g], initial_epsilon, alpha[a], updateMode,
                        actions_ch, actions_cca, actions_tpc, noise, print_info)
                for j in range(MAX_CONVERGENCE_TIME):
                    avg_tpt_evolution_ql[iteration] = np.append(avg_tpt_evolution_ql[iteration], np.mean(tpt_evolution_per_wlan_ql[iteration][j]))

                avg_tpt_experienced_ql[iteration][a][g] = np.mean(avg_tpt_evolution_ql[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME])
                std_tpt_experienced_ql[iteration][a][g] = np.std(avg_tpt_evolution_ql[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME])

                aggregate_tpt[iteration][a][g] = np.mean(np.sum(tpt_evolution_per_wlan_ql[iteration][MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME], axis=1))
    return wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt


def plot_experiment(wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt):
    images_dir = f"images/{experiment_name}"
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    # PLOT THE RESULTS

    # Compute the average results found for the aggregated throughput
    mean_aggregate_tpt = np.mean(aggregate_tpt, axis=0)
    print('mean_aggregate_tpt')
    print(mean_aggregate_tpt)
    # Compute the standard deviation for the aggregated throughput
    std_aggregate_tpt = np.std(aggregate_tpt, axis=0)
    print('std_aggregate_tpt')
    print(std_aggregate_tpt)

    # PLOT THE RESULTS
    # ix_alpha = np.argmax(mean_aggregate_tpt)
    ix_alpha = np.unravel_index(np.argmax(mean_aggregate_tpt), mean_aggregate_tpt.shape)
    maxVal = mean_aggregate_tpt[ix_alpha]
    # m, n = ind2sub(len(mean_aggregate_tpt), ix_alpha)
    # m, n = len(mean_aggregate_tpt)//ix_alpha, len(mean_aggregate_tpt)%ix_alpha
    m, n = ix_alpha
    print('Best alpha-gamma values:', alpha[n], '-', gamma[m])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    X, Y = np.meshgrid(alpha, gamma)
    surf = ax.plot_trisurf(X.flatten(), Y.flatten(), mean_aggregate_tpt.flatten(), cmap=cm.jet)
    fig.colorbar(surf)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\alpha$')
    ax.set_zlabel('Network Throughput (Mbps)')
    #plot3(alpha(n), gamma(m), maxVal, 'ro')
    plt.savefig(f"{images_dir}/tput.eps")

    # ix_alpha = np.argmax(std_aggregate_tpt)
    ix_alpha = np.unravel_index(np.argmax(std_aggregate_tpt), std_aggregate_tpt.shape)
    maxVal = std_aggregate_tpt[ix_alpha]
    # [m, n] = ind2sub(size(std_aggregate_tpt), ix_alpha)
    # m, n = len(std_aggregate_tpt)//ix_alpha, len(std_aggregate_tpt)%ix_alpha
    m, n = ix_alpha

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    surf = ax.plot_trisurf(X.flatten(), Y.flatten(), std_aggregate_tpt.flatten(), cmap=cm.jet)
    fig.colorbar(surf)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\alpha$')
    ax.set_zlabel('Standard Deviation (Mbps)')
    #plot3(alpha(n), gamma(m), maxVal, 'ro')
    plt.savefig(f"{images_dir}/std.eps")


if RUN_EXPERIMENT:
    wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt = run_experiment()

    data_dir = f"data/{experiment_name}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{data_dir}/wlan.npy", wlan)
    np.save(f"{data_dir}/avg_tpt_experienced_ql.npy", avg_tpt_experienced_ql)
    np.save(f"{data_dir}/std_tpt_experienced_ql.npy", std_tpt_experienced_ql)
    np.save(f"{data_dir}/aggregate_tpt.npy", aggregate_tpt)
    np.save(f"{data_dir}/aggregate_tpt.npy", aggregate_tpt)
else:
    wlan = np.load(f"data/{experiment_name}/wlan.npy", allow_pickle=True)
    avg_tpt_experienced_ql = np.load(f"data/{experiment_name}/avg_tpt_experienced_ql.npy", allow_pickle=True)
    std_tpt_experienced_ql = np.load(f"data/{experiment_name}/std_tpt_experienced_ql.npy", allow_pickle=True)
    aggregate_tpt = np.load(f"data/{experiment_name}/aggregate_tpt.npy", allow_pickle=True)

plot_experiment(wlan, avg_tpt_experienced_ql, std_tpt_experienced_ql, aggregate_tpt)
