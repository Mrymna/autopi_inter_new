############################################################################################
## common plotting functions that might be used in several different notebooks or figures ##
############################################################################################
import numpy as np

def plotMap(ax,myMap,title="",titleY=0.95,titleFontSize=10,transpose=True,cmap="jet",vmin=0,alpha=1):
    """
    Plot one 2D map
    """
    if transpose:
        ax.imshow(myMap.T,origin="lower",cmap=cmap,interpolation=None, vmin=vmin,alpha=alpha)
    else:
        ax.imshow(myMap,origin="lower",cmap=cmap,interpolation=None, vmin=vmin,alpha=alpha)
    ax.set_title(title,y=titleY,fontsize=titleFontSize)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

def plotHDHist(ax,oneHist,bins,title=""):
    """
    Plot one polar histogram
    """
    ownBins = bins.copy()
    ownHist = oneHist.copy()
    
    ownBins = np.append(ownBins,ownBins[0])
    ownHist = np.append(ownHist,ownHist[0])
    ax.plot(ownBins,
            ownHist)
    ax.set_xticklabels([])
    ax.set_title(title,y=0.98)
    ax.grid(True)
    
    
