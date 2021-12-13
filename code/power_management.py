import numpy as np

import auxiliary_methods as utils



def interferences(wlan, powMat):
# Returns the interferences power received at each WLAN
#   OUTPUT:
#       * intMat: 1xN array (N is the number of WLANs) with the
#       interferences noticed on each AP in mW
#   INPUT:
#       * wlan: contains information of each WLAN in the map. For instance,
#       wlan(1) corresponds to the first one, so that it has unique
#       parameters (x,y,z,BW,CCA,etc.).
#       * powMat: matrix NxN (N is the number of WLANs) with the power
#       received at each AP in dBm.

# We assume that overlapping channels also create an interference with lower level (20dB/d) - 20 dB == 50 dBm
    wlans = len(wlan)
    interferences = np.zeros(wlans)
    
    for i in range(wlans):
        for j in range(wlans):
            if i != j and wlan[j].transmitting:
                interferences[i] += utils.db2pow(powMat[i,j] - utils.db2pow(20 * (abs(wlan[i].channel - wlan[j].channel))))
    return interferences
#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)


def power_matrix(wlan):
# Returns the power received by each AP from all the others
#   OUTPUT:
#       - powMat: matrix NxN (N is the number of WLANs) with the power
#       received at each AP in dBm
#   INPUT:
#       - wlan: contains information of each WLAN in the map. For instance,
#       wlan(1) corresponds to the first one, so that it has unique
#       parameters (x,y,z,BW,CCA,etc.)
    N_WLANs = len(wlan)     # Number of WLANs (obtained from the input)
    PLd1=5                     # Path-loss factor
    shadowing = 9.5            # Shadowing factor
    obstacles = 30             # Obstacles factor

    powMat = np.zeros((N_WLANs, N_WLANs))
    # Compute the received power on all the APs from all the others
    for i in range(N_WLANs):
        for j in range(N_WLANs):
            if i != j:
                # Distance between APs of interest
                d_AP_AP = np.sqrt((wlan[i].x - wlan[j].x)**2 + (wlan[i].y - wlan[j].y)**2 + (wlan[i].z - wlan[j].z)**2) 
                # Propagation model
                alfa = 4.4
                PL_AP = PLd1 + 10 * alfa * np.log10(d_AP_AP) + shadowing / 2 + (d_AP_AP/10) * obstacles / 2
                powMat[i,j] = wlan[j].PTdBm - PL_AP
            else:
                # Calculate Power received at the STA associated to the AP
                d_AP_STA = np.sqrt((wlan[i].x - wlan[j].xn)**2 + (wlan[i].y - wlan[j].yn)**2 + (wlan[i].z - wlan[j].zn)**2) 
                # Propagation model
                alfa = 4.4
                PL_AP = PLd1 + 10 * alfa * np.log10(d_AP_STA) + shadowing / 2 + (d_AP_STA / 10) * obstacles / 2
                powMat[i,j] = wlan[i].PTdBm - PL_AP
    return powMat
