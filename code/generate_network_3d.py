# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Wlan:
    PTdBm: Optional[float] = None
    CCA: Optional[float] = None
    channel: Optional[float] = None
    BW: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    xn: float = 0
    yn: float = 0
    zn: float = 0


def GenerateNetwork3D(N_WLANs, NumChannels, topology, stas_position, printMap=False):
# GenerateNetwork3D - Generates a 3D network 
#   OUTPUT: 
#       * wlan - contains information of each WLAN in the map. For instance,
#       wlan(1) corresponds to the first one, so that it has unique
#       parameters (x,y,z,BW,CCA,etc.)
#   INPUT: 
#       * N_WLANs: number of WLANs on the studied environment
#       * NumChannels: number of available channels
#       * topology: topology of the network ('ring', 'line' or 'grid')
#       * stas_position: way STAs are placed (1 - "random", 2 - "safe" or 3 - "exposed")
#       * printMap: flag for calling DrawNetwork3D at the end
    actions_tpc = [5, 10, 15, 20]
    actions_cca = [-82]

    # Dimensions of the 3D map
    MaxX=10
    MaxY=5 
    MaxZ=10
    # Maximum range for a STA
    MaxRangeX = 1
    MaxRangeY = 1
    MaxRangeZ = 1
    MaxRange = np.sqrt(3)
    # AP density
    print('Density of APs', N_WLANs/(MaxX*MaxY*MaxZ))
    
    wlans = []
    x = []
    y = []
    z = []
    xn = []
    yn = []
    zn = []

    if topology == 'ring':
        x0 = MaxX/2
        y0 = MaxY/2
        r = (MaxY-1)/2
        n=N_WLANs
        tet=linspace(-np.pi,np.pi,n+1)                
        posX = r*np.cos(tet)+x0
        posY = r*np.sin(tet)+y0
    
    gridPositions4 = np.array([
            [(MaxX)/N_WLANs, (MaxY)/N_WLANs, MaxZ/2],
            [(MaxX)/N_WLANs, 3*(MaxY)/N_WLANs, MaxZ/2],
            [3*(MaxX)/N_WLANs, (MaxY)/N_WLANs, MaxZ/2],
            [3*(MaxX)/N_WLANs, 3*(MaxY)/N_WLANs, MaxZ/2],
            ])

    gridPositions8 = np.array([
            [(MaxX)/(N_WLANs/2), (MaxY)/(N_WLANs/2), MaxZ/(N_WLANs/2)],
            [(MaxX)/(N_WLANs/2), 3*(MaxY)/(N_WLANs/2), MaxZ/(N_WLANs/2)],
            [3*(MaxX)/(N_WLANs/2), (MaxY)/(N_WLANs/2), MaxZ/(N_WLANs/2)],
            [3*(MaxX)/(N_WLANs/2), 3*(MaxY)/(N_WLANs/2), MaxZ/(N_WLANs/2)],
            [(MaxX)/(N_WLANs/2), (MaxY)/(N_WLANs/2), 3*MaxZ/(N_WLANs/2)],
            [(MaxX)/(N_WLANs/2), 3*(MaxY)/(N_WLANs/2), 3*MaxZ/(N_WLANs/2)],
            [3*(MaxX)/(N_WLANs/2), (MaxY)/(N_WLANs/2), 3*MaxZ/(N_WLANs/2)],
            [3*(MaxX)/(N_WLANs/2), 3*(MaxY)/(N_WLANs/2), 3*MaxZ/(N_WLANs/2)],
            ])

    gridPositions12 = np.array([
            [(MaxX)/(N_WLANs/3), (MaxY)/(N_WLANs/3), MaxZ/(N_WLANs/3)],
            [(MaxX)/(N_WLANs/3), 3*(MaxY)/(N_WLANs/3), MaxZ/(N_WLANs/3)],
            [3*(MaxX)/(N_WLANs/3), (MaxY)/(N_WLANs/3), MaxZ/(N_WLANs/3)],
            [3*(MaxX)/(N_WLANs/3), 3*(MaxY)/(N_WLANs/3), MaxZ/(N_WLANs/3)],
            [(MaxX)/(N_WLANs/3), (MaxY)/(N_WLANs/3), 2*MaxZ/(N_WLANs/3)],
            [(MaxX)/(N_WLANs/3), 3*(MaxY)/(N_WLANs/3), 2*MaxZ/(N_WLANs/3)],
            [3*(MaxX)/(N_WLANs/3), (MaxY)/(N_WLANs/3), 2*MaxZ/(N_WLANs/3)],
            [3*(MaxX)/(N_WLANs/3), 3*(MaxY)/(N_WLANs/3), 2*MaxZ/(N_WLANs/3)],
            [(MaxX)/(N_WLANs/3), (MaxY)/(N_WLANs/3), 3*MaxZ/(N_WLANs/3)],
            [(MaxX)/(N_WLANs/3), 3*(MaxY)/(N_WLANs/3), 3*MaxZ/(N_WLANs/3)],
            [3*(MaxX)/(N_WLANs/3), (MaxY)/(N_WLANs/3), 3*MaxZ/(N_WLANs/3)],
            [3*(MaxX)/(N_WLANs/3), 3*(MaxY)/(N_WLANs/3), 3*MaxZ/(N_WLANs/3)],
            ])
 
    for j in range(N_WLANs):
        wlan = Wlan()
        wlan.PTdBm = random.choice(actions_tpc)  # Assign Tx Power
        wlan.CCA = random.choice(actions_cca)  # Assign CCA
        wlan.channel = random.randrange(NumChannels) # Assign channels
        wlan.BW = 20e6 
        if topology == 'ring':
            wlan.x = posX[j]
            wlan.y = posY[j]
            wlan.z = MaxZ/2 
        elif topology == 'line':
            wlan.x = j*((MaxX-2)/N_WLANs)
            wlan.y = MaxY/2
            wlan.z = MaxZ/2 
        elif topology == 'grid':
            if N_WLANs == 4:
                wlan.x = gridPositions4[j,0]
                wlan.y = gridPositions4[j,1]
                wlan.z = gridPositions4[j,2]
            elif N_WLANs == 8:
                wlan.x = gridPositions8[j,0]
                wlan.y = gridPositions8[j,1]
                wlan.z = gridPositions8[j,2] 
            elif N_WLANs == 12:
                wlan.x = gridPositions12[j,0]
                wlan.y = gridPositions12[j,1]
                wlan.z = gridPositions12[j,2]
            else:
                print('error, only 4, 8 and 12 WLANs allowed')

        # Build arrays of locations for each AP
        x.append(wlan.x)
        y.append(wlan.y)
        z.append(wlan.z)  
        
        if topology == 'grid':
            # Add the listening STA to each AP randomly
            if stas_position == 1: # RANDOM
                if random.random() < 0.5:
                    wlan.xn = wlan.x + MaxRangeX*random.random()
                else :
                    wlan.xn = wlan.x - MaxRangeX*random.random()

                if random.random() < 0.5:
                    wlan.yn = wlan.y + MaxRangeY*random.random()
                else:
                    wlan.yn = wlan.y - MaxRangeY*random.random()

                if random.random() < 0.5:
                    wlan.zn = wlan.z + MaxRangeZ*random.random()
                else:
                    wlan.zn = wlan.z - MaxRangeZ*random.random()
            elif stas_position == 2: # SAFE
                if j == 1:
                    wlan.xn = wlan.x - MaxRangeX
                    wlan.yn = wlan.y - MaxRangeY
                elif j == 2:
                    wlan.xn = wlan.x - MaxRangeX
                    wlan.yn = wlan.y + MaxRangeY
                elif j == 3:
                    wlan.xn = wlan.x + MaxRangeX
                    wlan.yn = wlan.y - MaxRangeY
                elif j == 4:
                    wlan.xn = wlan.x + MaxRangeX
                    wlan.yn = wlan.y + MaxRangeY
                wlan.zn = wlan.z  
            elif stas_position == 3: # EXPOSED
                if j == 1:
                    wlan.xn = wlan.x + MaxRangeX
                    wlan.yn = wlan.y + MaxRangeY
                elif j == 2:
                    wlan.xn = wlan.x + MaxRangeX
                    wlan.yn = wlan.y - MaxRangeY
                elif j == 3:
                    wlan.xn = wlan.x - MaxRangeX
                    wlan.yn = wlan.y + MaxRangeY
                elif j == 4:
                    wlan.xn = wlan.x - MaxRangeX
                    wlan.yn = wlan.y - MaxRangeY
                wlan.zn = wlan.z                         
        elif topology == 'line':
            # Add the listening STA to each AP randomly
            if stas_position == 1: # RANDOM
                if random.random() < 0.5:
                    wlan.xn = wlan.x + MaxRangeX*random.random()
                else:
                    wlan.xn = wlan.x - MaxRangeX*random.random()

                if random.random() < 0.5:
                    wlan.yn = wlan.y + MaxRangeY*random.random()
                else:
                    wlan.yn = wlan.y - MaxRangeY*random.random()

                if random.random() < 0.5:
                    wlan.zn = wlan.z + MaxRangeZ*random.random()
                else:
                    wlan.zn = wlan.z - MaxRangeZ*random.random()
            elif stas_position == 2: # SAFE
                wlan.xn = wlan.x
                wlan.yn = wlan.y + MaxRangeY
                wlan.zn = wlan.z
            elif stas_position == 3: # EXPOSED
                wlan.xn = wlan.x + ((MaxX-2)/N_WLANs)/2
                wlan.yn = wlan.y
                wlan.zn = wlan.z                    
        if topology == 'ring':
            # Add the listening STA to each AP randomly
            if stas_position == 1: # RANDOM
                if random.random() < 0.5:
                    wlan.xn = wlan.x + MaxRangeX*random.random()
                else:
                    wlan.xn = wlan.x - MaxRangeX*random.random()

                if random.random() < 0.5:
                    wlan.yn = wlan.y + MaxRangeY*random.random()
                else:
                    wlan.yn = wlan.y - MaxRangeY*random.random()

                if random.random() < 0.5:
                    wlan.zn = wlan.z + MaxRangeZ*random.random()
                else:
                    wlan.zn = wlan.z - MaxRangeZ*random.random()
            elif stas_position == 2: # SAFE
                wlan.xn = wlan.x
                wlan.yn = wlan.y
                wlan.zn = wlan.z - MaxRangeZ  
            elif stas_position == 3: # EXPOSED
                if j == 1:
                    wlan.xn = wlan.x + MaxRangeX
                    wlan.yn = wlan.y + MaxRangeY
                elif j == 2:
                    wlan.xn = wlan.x + MaxRangeX
                    wlan.yn = wlan.y - MaxRangeY
                elif j == 3:
                    wlan.xn = wlan.x - MaxRangeX
                    wlan.yn = wlan.y + MaxRangeY
                elif j == 4:
                    wlan.xn = wlan.x - MaxRangeX
                    wlan.yn = wlan.y - MaxRangeY
                wlan.zn = wlan.z                         
        xn.append(wlan.xn)  # what is xn[j] B: the "x" position of node j
        yn.append(wlan.yn)
        zn.append(wlan.zn)
        wlans.append(wlan)
        
#     print('Channels selected per WLAN')
#     for i in range(N_WLANs):
#         channels[i] = wlan[i].channel       
#     for i in range(NumChannels)
#         sumChannels[i] = sum(channels==i)
#     print(channels)
#     print('Times a channel is occupied')
#     print(sumChannels)
#   
    # Plot map of APs and STAs
    if printMap:
        DrawNetwork3D(wlan)

    return wlans
