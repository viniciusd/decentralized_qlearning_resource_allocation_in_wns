# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)
import copy
import random

import numpy as np

import throughput_calculation as tput
import auxiliary_methods as utils
import power_management as power

import warnings
warnings.filterwarnings("error")


def selectActionQLearning(Qval, actions_ch, actions_cca, actions_tpc, e):
# selectActionQLearning: returns the best possible action given the current state
#   OUTPUT:
#        * selected_action - contains the selected channel, CCA and TPC
#   INPUT:
#       * Qval - Q-values matrix for a given agent (maps actions with rewards)
#       * actions_ch - set of channels available
#       * actions_cca - set of CCA values available
#       * actions_tpc - set of TPC values available
#       * e - epsilon-greedy approach for exploration

    indexes = []

    if random.random() > e:
        val = np.max(Qval)

        # Check if there is more than one occurrence in order to select a value randomly
        if sum(Qval==val) > 1:
            for i in range(len(Qval)):
                if Qval[i] == val:
                    indexes.append(i)
            if not indexes:
                index = np.argmax(Qval)
            else:
                index = random.choice(indexes)
        else:
            index = np.argmax(Qval)
    else:
        index = random.randrange(len(Qval))

    a, b,c = utils.val2indexes(index, len(actions_ch), len(actions_cca), len(actions_tpc))
    selected_action = (a, b, c)

    return selected_action


def QlearningMethod(wlan,
    MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, gamma, initial_epsilon,
    alpha, updateMode, actions_ch, actions_cca, actions_tpc, noise, print_info=False):

# QlearningMethod - Given an OBSS, applies QLearning to maximize the
# experienced throughput
#   OUTPUT: 
#       * tpt_experienced_by_WLAN - throughput experienced by each WLAN
#         for each of the iterations done
#   INPUT: 
#       * wlan - wlan object containing information about all the WLANs
#       * MAX_CONVERGENCE_TIME - maximum number of iterations
#       * MAX_LEARNING_ITERATIONS - maximum number of iterations that an
#       agent performs at the same time
#       * gamma - discount factor (Q-learning)
#       * initial_epsilon - exploration coefficient (Q-learning)
#       * alpha - learning rate (Q-learning)
#       * actions_ch - set of channels
#       * actions_cca - set of carrier sense thresholds
#       * actions_tpc - set of transmit power values
#       * noise - floor noise in dBm
#       * print_info - to print information at the end of the simulation

    # Use a copy of wlan to make operations
    wlan_aux = copy.deepcopy(wlan)

    n_WLANs = len(wlan)
    # Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
    possible_actions = np.arange(len(actions_ch)*len(actions_cca)*len(actions_tpc))
    # Total number of actions
    K = len(possible_actions)
    
    # Find the index of the initial action taken by each WLAN
    initial_action_ix_per_wlan = np.zeros(n_WLANs, dtype=int)
    for i in range(n_WLANs):
        index_cca = np.where(actions_cca==wlan_aux[i].CCA)[0][0]
        index_tpc = np.where(actions_tpc==wlan_aux[i].PTdBm)[0][0]
        initial_action_ix_per_wlan[i] = utils.indexes2val(wlan_aux[i].channel, index_cca, index_tpc, len(actions_ch), len(actions_cca))

    # Initialize the indexes of the taken action
    action_ix_per_wlan = initial_action_ix_per_wlan
    
    # Compute the maximum achievable throughput per WLAN
    power_matrix = power.power_matrix(wlan_aux)
    upper_bound_tpt_per_wlan = tput.computeMaxBoundThroughput(wlan_aux, noise, np.max(actions_tpc))

    # Fill the Q-table of each node with 0's 
    Qval = np.zeros((n_WLANs, len(possible_actions)), dtype=np.longdouble)
       
    selected_arm = copy.deepcopy(action_ix_per_wlan)              # Initialize arm selection for each WLAN by using the initial action
    current_action = np.zeros(n_WLANs)
    previous_action = selected_arm
    times_arm_is_seleceted = np.zeros((n_WLANs, K))
    transitions_counter = np.zeros((n_WLANs, K*K))
    allcombs = utils.allcomb(range(K), range(K))

    # ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME               
    t = 1
    epsilon = initial_epsilon 
    cumulative_tpt_experienced_per_WLAN = 0
    cumulative_fairness = 0

    tpt_experienced_by_WLAN = []
    while t < MAX_CONVERGENCE_TIME + 1:
        # Assign turns to WLANs randomly 
        order = list(range(n_WLANs))
        random.shuffle(order)

        for i in range(n_WLANs): # Iterate sequentially for each agent in the random order                      

            learning_iteration = 1
            while learning_iteration <= MAX_LEARNING_ITERATIONS:
                # Select an action according to Q-learning policy
                selected_action = selectActionQLearning(Qval[order[i]],
                    actions_ch, actions_cca, actions_tpc, epsilon)

                ix_action = utils.indexes2val(selected_action[0], selected_action[1], selected_action[2], len(actions_ch), len(actions_cca))

                current_action[order[i]] = ix_action
                ix = np.where((allcombs[:,0] == previous_action[order[i]]) & (allcombs[:,1] == current_action[order[i]]))[0][0]
                previous_action[order[i]] = current_action[order[i]]
                transitions_counter[order[i], ix] += 1

                times_arm_is_seleceted[order[i], ix_action] += 1

                # Change parameters according to the action obtained
                wlan_aux[order[i]].channel = selected_action[0]
                # wlan_aux[order[i]].CCA = actions_cca[selected_action[1]]
                wlan_aux[order[i]].PTdBm = actions_tpc[selected_action[2]]

                # Prepare the next state according to the actions performed on the current state
                index_cca = np.where(actions_cca==wlan_aux[order[i]].CCA)[0][0]
                index_tpc = np.where(actions_tpc==wlan_aux[order[i]].PTdBm)[0][0]
                action_ix_per_wlan[order[i]] =  utils.indexes2val(wlan_aux[order[i]].channel, index_cca,
                    index_tpc, len(actions_ch), len(actions_cca))

                learning_iteration += 1

        power_matrix = power.power_matrix(wlan_aux)
        tpt_experienced_by_WLAN.append(tput.computeThroughputFromSINR(wlan_aux, power_matrix, noise))  # bps 

        # Update Q:
        for wlan_i in range(n_WLANs):
            rw = (tpt_experienced_by_WLAN[t-1][wlan_i] / upper_bound_tpt_per_wlan[wlan_i])
            Qval[wlan_i][action_ix_per_wlan[wlan_i]] = (1 - alpha) * Qval[wlan_i][action_ix_per_wlan[wlan_i]] +  \
                                                        (alpha * rw + gamma * (np.max(Qval[wlan_i])))
        cumulative_tpt_experienced_per_WLAN = cumulative_tpt_experienced_per_WLAN +  tpt_experienced_by_WLAN[t-1]
        cumulative_fairness = cumulative_fairness + utils.JainsFairness(tpt_experienced_by_WLAN[t-1])

        # Update the exploration coefficient according to the inputted mode
        if updateMode == 0:
            epsilon = initial_epsilon / t
        elif updateMode == 1:
            epsilon = initial_epsilon / np.sqrt(t)

        # Increase the number of 'learning iterations' of a WLAN
        t = t + 1 
    

    if print_info:
        import matplotlib.pylab as plt

        print('+++++++++++++++++')
        print('Q-learning execution results per WN:')
        
        # Throughput experienced by each WLAN for each EXP3 iteration
        plt.xlim((1, 20))
        plt.ylim((30, 70))
        # Print the preferred action per wlan
        h = []
        for i in range(n_WLANs):

            times_arm_is_seleceted[i, :]/MAX_CONVERGENCE_TIME

            ix = np.argmax(Qval[i])
            a, _, c = utils.val2indexes(possible_actions[ix], len(actions_ch), len(actions_cca), len(actions_tpc))
            print('   * WN', i, ':')
            print('       - Channel:', a)
            print('       - TPC:', actions_tpc[c])
            print('       - Transitions probabilities (top-3):')
            plt.tight_layout()
            h.append(plt.subplot(2, 2, i+1))
            plt.bar(range(K), times_arm_is_seleceted[i, :]/MAX_CONVERGENCE_TIME)
            plt.xlim((0, 9))
            plt.ylim((0, 1))
            # xticks(1:8)
            # xticklabels(1:8)
            plt.title(f'WN {i}')
            plt.xlabel('Action Index')
            plt.ylabel('Probability')


            a = transitions_counter[i,:]
            # Max value
            ix1 = np.argmax(a)
            val1 = a[ix1]
            ch1_1, _, x = utils.val2indexes(possible_actions[allcombs[ix1,0]], len(actions_ch), len(actions_cca), len(actions_tpc))
            tpc1_1 = actions_tpc[x]
            ch1_2, _, x = utils.val2indexes(possible_actions[allcombs[ix1,1]], len(actions_ch), len(actions_cca), len(actions_tpc))
            tpc1_2 = actions_tpc[x]
            print('              . prob. of going from', allcombs[ix1,0], ' (ch = ', ch1_1, '/tpc =', tpc1_1, ')',
                ' to ', allcombs[ix1,1], ' (ch =', ch1_2, '/tpc = ', tpc1_2, ')',
                '=', val1/MAX_CONVERGENCE_TIME)
            # Second max value
            ix2 = np.argmax(a[a < np.max(a)])
            val2 = a[ix2]
            ch2_1, _, x = utils.val2indexes(possible_actions[allcombs[ix2,0]], len(actions_ch), len(actions_cca), len(actions_tpc)) 
            tpc2_1 = actions_tpc[x]
            ch2_2, _, x = utils.val2indexes(possible_actions[allcombs[ix2,1]], len(actions_ch), len(actions_cca), len(actions_tpc)) 
            tpc2_2 = actions_tpc[x]
            print('              . prob. of going from', allcombs[ix2,0], ' (ch =', ch2_1, '/tpc =', tpc2_1, ')',
                ' to ', allcombs[ix2,1], ' (ch=', ch2_2, '/tpc=', tpc2_2, ')',
                ' = ', val2/MAX_CONVERGENCE_TIME)
            # Third max value
            if (a < max(a[a < np.max(a)])).any():
                ix3 = np.argmax(a[a < max(a[a < np.max(a)])])
                val3 = a[ix3]
                ch3_1, _, x = utils.val2indexes(possible_actions[allcombs[ix3,0]], len(actions_ch), len(actions_cca), len(actions_tpc)) 
                tpc3_1 = actions_tpc[x]
                ch3_2, _, x = utils.val2indexes(possible_actions[allcombs[ix3,1]], len(actions_ch), len(actions_cca), len(actions_tpc))
                tpc3_2 = actions_tpc[x] 
                print('              . prob. of going from', allcombs[ix3,0], ' (ch =', ch3_1, '/tpc =', tpc3_1, ')',
                    ' to ', allcombs[ix3,1], ' (ch =', ch3_2, '/tpc =', tpc3_2, ')',
                    ' = ', val3/MAX_CONVERGENCE_TIME)

        print('+++++++++++++++++')
        plt.show()

    return tpt_experienced_by_WLAN, Qval    
