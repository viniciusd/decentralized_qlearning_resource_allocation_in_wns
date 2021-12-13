# Decentralized_Qlearning_Resource_Allocation_in_WNs

#   Francesc Wilhelmi, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Boris Bellalta, Wireless Networking Research Group (WN-UPF), Universitat Pompeu Fabra
#   Cristina Cano, Wireless Networks Research Group (WINE-UOC), Universitat Oberta de Catalunya (UOC)
#   Anders Jonsson, Artificial Intelligence and Machine Learning Research Group (AIML-UPF), Universitat Pompeu Fabra (UPF)
import matplotlib.pylab as plt

def DrawNetwork3D(wlan):
# DrawNetwork3D - Plots a 3D of the network 
#   INPUT: 
#       * wlan - contains information of each WLAN in the map. For instance,
#       wlan(1) corresponds to the first one, so that it has unique
#       parameters (x,y,z,BW,CCA,etc.)
    MaxX = 10
    MaxY = 5 
    MaxZ = 10
    for j in range(wlan.shape[1]):
        x[j] = wlan[j].x
        y[j] = wlan[j].y
        z[j] = wlan[j].z

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    labels = np.arange(len(y))
    for i in range(wlan.shape[1]):
        ax.scatter(wlan[i].x, wlan[i].y, wlan[i].z, s=70)
        ax.scatter(wlan[i].xn, wlan[i].yn, wlan[i].zn, s=30)
        ax.line([wlan[i].x, wlan[i].xn], [wlan[i].y, wlan[i].yn], [wlan[i].z, wlan[i].zn], ':')
    ax.text(x,y,z,labels)
    ax.set_xlabel('x [meters]')
    ax.set_ylabel('y [meters]')
    ax.set_zlabel('z [meters]')
    ax.set_xlim(0, MaxX)
    ax.set_ylim(0, MaxY)
    ax.set_zlim(0, MaxZ)

    plt.show()
