import copy

import numpy as np 

import auxiliary_methods as utils
import power_management as power 


def computeMaxBoundThroughput(wlan, noise, maxPower):
# Given an WLAN (AP+STA), compute the maximum capacity achievable according
# to the power obtained at the receiver without interference
#
# OUTPUT:
#   * optimal_tpt - maximum achievable throughput per WLAN (Mbps)
# INPUT:
#   * wlan - object containing all the WLANs information 
#   * powMat - power received from each AP
#   * noise - floor noise in dBm
    wlan_aux = copy.deepcopy(wlan)
    optimal_tpt = []
    for i in range(len(wlan_aux)):
        wlan_aux[i].PTdBm = maxPower
    power_matrix = power.power_matrix(wlan_aux)
    for i in range(len(wlan_aux)):
        optimal_tpt.append(computeTheoreticalCapacity(wlan_aux[i].BW, utils.db2pow(powMat[i,i] - noise + 30))/1e6) # Mbps

    return optimal_tpt


def computeTheoreticalCapacity(B, sinr): 
# Computes the theoretical capacity given a bandwidth and a SINR 
# 
# OUTPUT: 
#   * C - capacity in bps 
# INPUT: 
#   * B - Available Bandwidth (Hz)  
#   * sinr - Signal to Interference plus Noise Ratio (-) 
    C = B * np.log2(1+sinr) 
 
    return C 
 
 
def computeThroughputFromSINR(wlan, powMat, noise): 
# Computes the throughput of each WLAN in wlan according to the 
# interferences sensed  
#  * Assumption: all the devices transmit at the same time and the 
#    throughput is computed as the capacity obtained from the total SINR  
# 
# OUTPUT: 
#   * tpt - tpt achieved by each WLAN (Mbps) 
# INPUT: 
#   * wlan - object containing all the WLANs information  
#   * powMat - power received from each AP 
#   * noise - floor noise in dBm 
    N_WLANs = len(wlan) 
    sinr = np.zeros(N_WLANs) 
    tpt = np.zeros(N_WLANs) 
    # Activate all the WLANs 
    for j in range(N_WLANs): 
        wlan[j].transmitting = True 
    # Compute the tpt of each WLAN according to the sensed interferences 
    for i in range(N_WLANs): 
        interferences = power.interferences(wlan, powMat) # dBm 
        sinr[i] = powMat[i,i] - utils.pow2db((interferences[i] + utils.db2pow(noise))) # dBm 
        tpt[i] = computeTheoreticalCapacity(wlan[i].BW, utils.db2pow(sinr[i])) / 1e6 # Mbps 
    return tpt 


def computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise): 
# Computes the throughput experienced by each WLAN for all the possible 
# combinations of Channels, CCA and TPC  
# 
#   NOTE: the "allcomb" function does not hold big amounts of combinations  
#   (a reasonable limit is 4 WLANs with 2 channels and 4 levels of TPC) 
# 
# OUTPUT: 
#   * tpt - tpt achieved by each WLAN for each configuration (Mbps) 
# INPUT: 
#   * wlan - object containing all the WLANs information  
#   * actions_ch - set of channels 
#   * actions_cca - set of carrier sense thresholds 
#   * actions_tpc - set of transmit power values 
#   * noise - floor noise in dBm 
 
    print('      - Computing the throughput for all the combinations...') 
 
    # Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower" 
    possible_actions = np.arange(len(actions_ch)*len(actions_cca)*len(actions_tpc.shape)) 
    # Set of possible combinations of configuration   
    possible_comb = utils.allcomb(possible_actions,possible_actions,possible_actions,possible_actions) 
 
    n_WLANs = len(wlan) 
    nChannels = len(actions_ch) 
     
    wlan_aux = copy.deepcopy(wlan)    # Generate a copy of the WLAN object to make modifications 
 
    # Try all the combinations 
    tpt_wlan_per_conf = [] 
    for i in range (possible_comb.shape[0]): 
        # Change WLANs configuration  
        for j in range(n_WLANs): 
            ch, _, tpc_ix = utils.val2indexes(possible_comb[i,j], nChannels, len(actions_cca), len(actions_tpc)) 
            wlan_aux[j].channel = ch    
            wlan_aux[j].PTdBm = actions_tpc[tpc_ix]             
        # Compute the Throughput and store it 
        power_matrix = power.power_matrix(wlan_aux)  
        tpt_wlan_per_conf.append(computeThroughputFromSINR(wlan_aux, power_matrix, noise)) 
     
    # Find the best configuration for each WLAN and display it 
    agg_tpt = [] 
    fairness = [] 
    prop_fairness = [] 
    for i in range(len(tpt_wlan_per_conf)): 
        agg_tpt.append(np.sum(tpt_wlan_per_conf[i])) 
        fairness.append(utils.JainsFairness(tpt_wlan_per_conf[i])) 
        prop_fairness.append(np.sum(np.log(tpt_wlan_per_conf[i]))) 
     
    ix = np.argmax(prop_fairness) 
    val = prop_fairness[ix] 
    print('---------------') 
    print('Best proportional fairness:', val) 
    print('Aggregate throughput:', agg_tpt[ix], 'Mbps') 
    print('Fairness:', fairness[ix]) 
    print('Best configurations:', possible_comb[ix, :]) 
    for i in range(n_WLANs): 
        a, _, c = utils.val2indexes(possible_comb[ix, i], nChannels, len(actions_cca), len(actions_tpc)) 
        print('   * WLAN', i, ':') 
        print('       - Channel:', a) 
        print('       - TPC:', actions_tpc[c]) 
     
     
    ix2 = np.argmax(agg_tpt) 
    val2 = agg_tpt[ix2] 
    print('---------------') 
    print('Best aggregate throughput:', val2, 'Mbps') 
    print('Fairness:', fairness[ix2]) 
    print('Best configurations:', possible_comb[ix2, :]) 
    for i in range(n_WLANs): 
        a, _, c = utils.val2indexes(possible_comb[ix2, i], nChannels, len(actions_cca), len(actions_tpc)) 
        print('   * WLAN', i, ':') 
        print('       - Channel:', a) 
        print('       - TPC:', actions_tpc[c]) 
 
    return tpt_wlan_per_conf     
