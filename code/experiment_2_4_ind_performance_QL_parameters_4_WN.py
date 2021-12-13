# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)

# EXPERIMENT EXPLANATION:
# By using a simple grid of 4 WLANs sharing 2 channels, we want to test the
# Q-learning method if using different numbers of iterations. We fix alpha,
# gamma and initial epsilon to the values that generated better results in
# terms of proportional fairness in the Experiment_1
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np

from generate_network_3d import GenerateNetwork3D
from qlearning_method import QlearningMethod
import auxiliary_methods as utils


print('****************************************************************************************')
print('* Implications of Decentralized Learning Resource Allocation in WNs                    *')
print('* Copyright (C) 2017-2022, and GNU GPLd, by Francesc Wilhelmi                          *')
print('* GitHub: https://github.com/wn-upf/Decentralized_Qlearning_Resource_Allocation_in_WNs *')
print('****************************************************************************************')

print('-----------------------')
print('EXPERIMENT 2-4: Individual performance Q-learning')
print('-----------------------')

RUN_EXPERIMENT = False  # Should re-run the experiment or just laod pre-saved data?
print_info = False          # print info when calling QlearningMethod

# DEFINE THE VARIABLES TO BE USED

# GLOBAL VARIABLES
n_WLANs = 4                    # Number of WLANs in the map
n_agents = 4                   # Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 10000   # Maximum convergence time (one period implies the participation of all WLANs)
MIN_SAMPLE_CONSIDER = 1        # Iteration from which to consider the obtained results
MAX_LEARNING_ITERATIONS = 1    # Maximum number of learning iterations done by each WLAN inside a general iteration

# WLAN object to be modified for each number of coexistent nodes
nChannels = 2              # Number of available channels (from 1 to NumChannels)
noise = -100               # Floor noise (dBm)

# Definition of actions:
actions_ch = np.arange(1, nChannels+1)      # nChannels possible channels
actions_cca = np.array([-82])               # One CCA level (dBm) -> meaningless (all interferences are considered)
actions_tpc = np.array([5, 10, 15, 20])     # 4 different levels of TPC (dBm)

# Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
possible_actions = np.arange(len(actions_ch)*len(actions_cca)*len(actions_tpc))
# Total number of actions
K = len(possible_actions)
possible_comb = utils.allcomb(possible_actions,possible_actions,possible_actions,possible_actions)

# Q-learning parameters
gamma = .05            # Discount rate
initial_epsilon = .1   # Initial Exploration coefficient
updateMode = 1         # 0: epsilon = initial_epsilon / t  1: epsilon = epsilon / sqrt(t)
alpha  = .1            # Learning rate

# Setup the scenario: generate WLANs and initialize states and actions
wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 2, 0) # SAFE CONFIGURATION

# Exploration approach
# seed = datetime.now()
seed = 1
print("Random seed", seed)
random.seed(seed)

experiment_name = Path(__file__).stem

  
def run_experiment():
    # Compute the throughput experienced per WLAN at each iteration                             
    tpt_evolution_per_wlan_ql, _ = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS,
            gamma, initial_epsilon, alpha, updateMode, actions_ch, actions_cca, actions_tpc, noise, print_info)
    tpt_evolution_per_wlan_ql = np.array(tpt_evolution_per_wlan_ql)

    return tpt_evolution_per_wlan_ql

def plot_experiment(tpt_evolution_per_wlan_ql):
    global MIN_SAMPLE_CONSIDER # It gets reassigned later on
    images_dir = f"images/{experiment_name}"
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    # PLOT THE RESULTS
    print('Aggregate throughput experienced on average:', np.mean(np.sum(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME], axis=1)))
    print('Fairness on average:', np.mean(utils.JainsFairness(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME])))
    print('Proportional fairness experienced on average:', np.mean(np.sum(np.log(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME]), axis=1)))

    # Throughput experienced by each WLAN for each EXP3 iteration
    plt.figure()
    for i in range(n_WLANs):
        plt.subplot(n_WLANs//2, n_WLANs//2, i+1)
        tpt_per_iteration = tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME, i]
        plt.plot(np.arange(MIN_SAMPLE_CONSIDER, MAX_CONVERGENCE_TIME), tpt_per_iteration)
        plt.title(f'WN {i}')
        plt.xlim((MIN_SAMPLE_CONSIDER-1, MAX_CONVERGENCE_TIME))
        plt.ylim((0, 1.1 * np.max(tpt_per_iteration)))
        plt.ylabel('Throughput')
        plt.xlabel('Q-learning iteration')
    plt.tight_layout()
    plt.savefig(f"{images_dir}/tput_per_wlan.eps")

    # Aggregated throughput experienced for each QL iteration
    plt.figure()
    agg_tpt_per_iteration = np.sum(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME], axis=1)
    plt.plot(np.arange(MIN_SAMPLE_CONSIDER, MAX_CONVERGENCE_TIME), agg_tpt_per_iteration)
    plt.xlabel('Q-learning Iteration')
    plt.ylabel('Network Throughput (Mbps)')
    plt.xlim((MIN_SAMPLE_CONSIDER-1, MAX_CONVERGENCE_TIME))
    plt.ylim((0, 1.1 * np.max(agg_tpt_per_iteration)))
    plt.savefig(f"{images_dir}/agg_tpt.eps")

    # Proportional fairness experienced for each QL iteration
    plt.figure()
    proprotional_fairness_per_iteration = np.sum(np.log(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME]), axis=1)
    plt.plot(np.arange(MIN_SAMPLE_CONSIDER, MAX_CONVERGENCE_TIME), proprotional_fairness_per_iteration)
    plt.xlabel('Q-learning Iteration')
    plt.ylabel('Proportional Fairness')
    plt.xlim((MIN_SAMPLE_CONSIDER-1, MAX_CONVERGENCE_TIME))
    plt.ylim((0, 1.1 * np.max(proprotional_fairness_per_iteration)))
    plt.savefig(f"{images_dir}/proportional_fairness.eps")

    # Average tpt experienced per WLAN
    MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME//2 + 1
    mean_tpt_per_wlan = np.mean(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME], axis=0)
    std_per_wlan = np.std(tpt_evolution_per_wlan_ql[MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME], axis=0)
    plt.figure()
    plt.xlabel('WN id')
    plt.ylabel('Mean throughput (Mbps)')
    plt.bar(np.arange(1, len(wlan)+1), mean_tpt_per_wlan, width=0.5)
    plt.errorbar(np.arange(1, len(wlan)+1), mean_tpt_per_wlan,yerr=std_per_wlan, fmt=".r")
    plt.xlim((0, 5))
    plt.ylim((0, 350))
    plt.savefig(f"{images_dir}/errorbar.eps")


if RUN_EXPERIMENT:
    tpt_evolution_per_wlan_ql = run_experiment()

    data_dir = f"data/{experiment_name}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{data_dir}/tpt_evolution_per_wlan_ql.npy", tpt_evolution_per_wlan_ql)
else:
    tpt_evolution_per_wlan_ql = np.load(f"data/{experiment_name}/tpt_evolution_per_wlan_ql.npy", allow_pickle=True)

plot_experiment(tpt_evolution_per_wlan_ql)
