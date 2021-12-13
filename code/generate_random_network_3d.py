# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)
import random
import numpy as np

def GenerateRandomNetwork3D(N_WLANs, NumChannels, printMap=False):
    # DrawNetwork3D  Calculate interferences on WLANs.
    #   Inputs:
    #       * N_WLANs: number of WLANs on the studied environment
    #       * NumChannels: number of available channels
    #       * B: bandwidth available per WLAN (Hz)
    #   Output:
    #       * wlan: object containing the information of each wlan drawn

    # Dimensions of the 3D map
    MaxX=10
    MaxY=5 
    MaxZ=10
    # Maximum range for a STA
    MaxRangeX = 1
    MaxRangeY = 1
    MaxRangeZ = 1
    # AP density
    print('Density of APs', N_WLANs / (MaxX * MaxY * MaxZ))

    # Locate elements on the map randomly
    for j in range(N_WLANs):
        # Assign Tx Power and CCA on the WLAN
        wlan[j].PTdBm = 20
        wlan[j].CCA = -82
        # Assign channel to the AP randomly
        wlan[j].channel = ceil(NumChannels*rand())
        # Assign location to the AP on the 3D map
        wlan[j].x = MaxX*random.random()
        wlan[j].y = MaxY*random.random()
        wlan[j].z = MaxZ*random.random()  
        # Build arrays of locations for each AP
        x[j]=wlan[j].x
        y[j]=wlan[j].y
        z[j]=wlan[j].z
        # Assign a STA to each AP for throughput analysis
        if random.random() < 0.5:
            xc = MaxRangeX*random.random()   #dnode*random.random() #what xc represents ? B: Is just an auxiliary variable to fix the position of the node around the AP, see below
        else:
            xc = -MaxRangeX*random.random()
        if random.random() < 0.5:
            yc = MaxRangeY*random.random()
        else:
            yc = -MaxRangeY*random.random()
        if random.random() < 0.5:
            zc = MaxRangeZ*random.random()
        else:
            zc = -MaxRangeZ*random.random()
        wlan[j].xn = min(abs(wlan[j].x+xc), MaxX)  
        wlan[j].yn = min(abs(wlan[j].y+yc), MaxY)
        wlan[j].zn = min(abs(wlan[j].z+zc), MaxZ)
        xn[j]=wlan[j].xn  #what is xn[j] B: the "x" position of node j
        yn[j]=wlan[j].yn
        zn[j]=wlan[j].zn        
        wlan[j].BW = 20e6 

    # Plot map of APs and STAs
    if printMap:
        print('Channels selected per WLAN')
        for i in range(N_WLANs):
            channels[i] = wlan[i].channel       
        sumChannels = np.zeros(NumChannels)
        for i in range(NumChannels):
            sumChannels[i] = np.sum(channels==i)
        print(channels)
        print('Times a channel is occupied')
        print(sumChannels)
        DrawNetwork3D(wlan)

    return wlan
