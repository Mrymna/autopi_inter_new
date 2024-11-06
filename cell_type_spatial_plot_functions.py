#################################################################
####plotting function for cell_Type_spatial_categories.ipynb ####
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime

#########################
### information score ###
#########################

def plotInfoDistribution(ax,info):
    ax.hist(info,bins=np.linspace(0,3,50))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("Info score")
    ax.set_xticks(np.linspace(0,3,4))
    ax.set_ylabel("Neurons")
def plotInfoShuffleExample(ax,dfInfoScore,cluId,trialNo=0):
    
    trialCode = "circ80_{}".format(trialNo)
    
    observed = dfInfoScore.infoScore[(dfInfoScore.shuffling==False)&
                                      (dfInfoScore.cluId==cluId)&
                                      (dfInfoScore.trialCode==trialCode)].values.item()
    shuf = dfInfoScore.infoScore[(dfInfoScore.shuffling==True)&
                              (dfInfoScore.cluId==cluId)&
                              (dfInfoScore.trialCode==trialCode)].values
    
    h = ax.hist(shuf,label="Shuf.")
    ax.plot([observed,observed],[0,np.max(h[0])],label="Obs.")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("Info score")
    ax.set_xticks(np.linspace(0,1,3))
    ax.set_ylabel("Shuffles")
    ax.legend()
def plotInfoScoreStability(ax,dfInfoScore,usableCellIndices):
    info = dfInfoScore.infoScore[(dfInfoScore.shuffling==False)&(dfInfoScore.trialCode=="circ80_0")]
    info0 = info[usableCellIndices]
    info = dfInfoScore.infoScore[(dfInfoScore.shuffling==False)&(dfInfoScore.trialCode=="circ80_1")]
    info1 = info[usableCellIndices]
    
    ax.scatter(info0,info1,s=2,alpha=0.3)
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)
    ax.set_xticks(np.linspace(0,3,3))
    ax.set_yticks(np.linspace(0,3,3))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    ax.set_xlabel("Info score trial 0")
    
    ax.set_ylabel("Info score trial 1")

def summaryPlotInfo(gs,fig,dfInfoScore,cells):
    nPlots = 5
    spec = gridspec.GridSpecFromSubplotSpec(1,nPlots, subplot_spec=gs[:])
    
    usableCellIndices = cells.usable.to_numpy()
    
    
    ax = fig.add_subplot(spec[0]) # add an axes to the figure
    # Distribution of all info scores in the first circ80 trial, for all clean cells
    info = dfInfoScore.infoScore[(dfInfoScore.shuffling==False)&(dfInfoScore.trialCode=="circ80_0")]
    info = info[usableCellIndices]
    plotInfoDistribution(ax,info)

    # Stability
    ax = fig.add_subplot(spec[1]) # add an axes to the figure
    plotInfoScoreStability(ax,dfInfoScore,usableCellIndices)

    # Example
    ax = fig.add_subplot(spec[2]) # add an axes to the figure
    cluId = "mn8578-30112021-0107_22"
    plotInfoShuffleExample(ax,dfInfoScore,cluId)


    
    
    
    ax = fig.add_subplot(spec[3]) # add an axes to the figure
    spa = cells.spatiallySelective_AND[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["SS","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
    else:
        labels = ["Ns", "SS"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]
    ax.pie(spa,labels = labels,autopct='%.0f%%',colors=colors)
    ax.text(-1,-1.8, "Spatially\nselective (AND)")

    ax = fig.add_subplot(spec[4]) # add an axes to the figure
    spa = cells.spatiallySelective_OR[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["SS","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
    else:
        labels = ["Ns", "SS"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]
    ax.pie(spa,labels = labels,autopct='%.0f%%',colors=colors)
    ax.text(-1,-1.8,"Spatially\nselective (OR)")

######################################
## grid scores #######################
######################################
def plotGridDistribution(ax,grid):
    ax.hist(grid,bins=np.linspace(-0.5,2.5,50))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("Grid score")
    ax.set_xlim(-0.5,1.7)
    ax.set_xticks(np.linspace(-0.5,1.5,3))
    ax.set_ylabel("Neurons")
def plotGridShuffleExample(ax,dfGridScore,cluId,trialNo=0):
    
    trialCode = "circ80_{}".format(trialNo)
    
    observed = dfGridScore.gridScore[(dfGridScore.shuffling==False)&
                                      (dfGridScore.cluId==cluId)&
                                      (dfGridScore.trialCode==trialCode)].values.item()
    shuf = dfGridScore.gridScore[(dfGridScore.shuffling==True)&
                              (dfGridScore.cluId==cluId)&
                              (dfGridScore.trialCode==trialCode)].values
    
    h = ax.hist(shuf,label="Shuf.")
    ax.plot([observed,observed],[0,np.max(h[0])],label="Obs.")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("Grid score")
    ax.set_xticks(np.linspace(-0.5,1.5,3))
    ax.set_ylabel("Shuffles")
    ax.legend()
def plotGridScoreStability(ax,dfGridScore,usableCellIndices):
    grid = dfGridScore.gridScore[(dfGridScore.shuffling==False)&(dfGridScore.trialCode=="circ80_0")]
    grid0 = grid[usableCellIndices]
    grid = dfGridScore.gridScore[(dfGridScore.shuffling==False)&(dfGridScore.trialCode=="circ80_1")]
    grid1 = grid[usableCellIndices]
    
    ax.scatter(grid0,grid1,s=2,alpha=0.3)
    ax.set_xlim(-0.5,1.7)
    ax.set_ylim(-0.5,1.7)
    ax.set_xticks(np.linspace(-0.5,1.5,3))
    ax.set_yticks(np.linspace(-0.5,1.5,3))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    ax.set_xlabel("Grid score trial 0")
    
    ax.set_ylabel("Grid score trial 1")
    
def summaryPlotGrid(gs,fig,dfGridScore,cells):
    nPlots = 5
    spec = gridspec.GridSpecFromSubplotSpec(1,nPlots, subplot_spec=gs[:])
    
    
    usableCellIndices = cells.usable.to_numpy()
    
    ax = fig.add_subplot(spec[0]) # add an axes to the figure

    # Distribution of all info scores in the first circ80 trial, for all clean cells
    grid = dfGridScore.gridScore[(dfGridScore.shuffling==False)&(dfGridScore.trialCode=="circ80_0")]
    grid = grid[usableCellIndices]
    plotGridDistribution(ax,grid)

    # Stability
    ax = fig.add_subplot(spec[1]) # add an axes to the figure
    plotGridScoreStability(ax,dfGridScore,usableCellIndices)

    # Example
    ax = fig.add_subplot(spec[2]) # add an axes to the figure
    cluId = "mn8578-30112021-0107_22"
    plotGridShuffleExample(ax,dfGridScore,cluId)

    ax = fig.add_subplot(spec[3]) # add an axes to the figure
    spa = cells.gridCell_AND[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["G","Ns"]
    else:
        labels = ["Ns", "G"]

    ax.pie(spa,labels = labels,autopct='%.0f%%')
    ax.text(-1,-1.8, "Grid cells\n(AND)")
    

    ax = fig.add_subplot(spec[4]) # add an axes to the figure
    spa = cells.gridCell_OR[cells.usable].value_counts()
 
    if spa.index[0]:
        labels = ["Grid","Ns"]
    else:
        labels = ["Ns", "Grid"]
    ax.pie(spa,labels = labels,autopct='%.0f%%')
    ax.text(-1,-1.8, "Grid cells\n(OR)")
    
##################################################
## HD scores #####################################
##################################################

def plotHDDistribution(ax,hd):
    ax.hist(hd,bins=np.linspace(0,1,50))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("HD score")
    ax.set_xlim(0,1)
    ax.set_xticks(np.linspace(0,1,3))
    ax.set_ylabel("Neurons")
def plotHDShuffleExample(ax,dfHDScore,cluId,trialNo=0):
    
    trialCode = "circ80_{}".format(trialNo)
    
    observed = dfHDScore.HDScore[(dfHDScore.shuffling==False)&
                                      (dfHDScore.cluId==cluId)&
                                      (dfHDScore.trialCode==trialCode)].values.item()
    shuf = dfHDScore.HDScore[(dfHDScore.shuffling==True)&
                              (dfHDScore.cluId==cluId)&
                              (dfHDScore.trialCode==trialCode)].values
    
    h = ax.hist(shuf,label="Shuf.")
    ax.plot([observed,observed],[0,np.max(h[0])],label="Obs.")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("HD score")
    ax.set_xticks(np.linspace(0,1,3))
    ax.set_ylabel("Shuffles")
    ax.legend()
def plotHDScoreStability(ax,dfHDScore,usableCellIndices):
    HD = dfHDScore.HDScore[(dfHDScore.shuffling==False)&(dfHDScore.trialCode=="circ80_0")]
    HD0 = HD[usableCellIndices]
    HD = dfHDScore.HDScore[(dfHDScore.shuffling==False)&(dfHDScore.trialCode=="circ80_1")]
    HD1 = HD[usableCellIndices]
    
    ax.scatter(HD0,HD1,s=2,alpha=0.3)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks(np.linspace(0,1,3))
    ax.set_yticks(np.linspace(0,1,3))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    ax.set_xlabel("HD score trial 0")
    
    ax.set_ylabel("HD score trial 1")

def summaryPlotHD(gs,fig,dfHDScore,cells):
    nPlots = 5
    spec = gridspec.GridSpecFromSubplotSpec(1,nPlots, subplot_spec=gs[:])
    
    usableCellIndices = cells.usable.to_numpy()
    
    ax = fig.add_subplot(spec[0]) # add an axes to the figure

    # Distribution of all info scores in the first circ80 trial, for all clean cells
    HD = dfHDScore.HDScore[(dfHDScore.shuffling==False)&(dfHDScore.trialCode=="circ80_0")]
    HD = HD[usableCellIndices]
    plotHDDistribution(ax,HD)

    # Stability
    ax = fig.add_subplot(spec[1]) # add an axes to the figure
    plotHDScoreStability(ax,dfHDScore,usableCellIndices)

    # Example
    ax = fig.add_subplot(spec[2]) # add an axes to the figure
    cluId = "jp3129-10062022-0107_159"
    plotHDShuffleExample(ax,dfHDScore,cluId)

    ax = fig.add_subplot(spec[3]) # add an axes to the figure
    spa = cells.HDCell_AND[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["HD","Ns"]
    else:
        labels = ["Ns", "HD"]

    ax.pie(spa,labels = labels,autopct='%.0f%%')
    ax.text(-1,-1.8, "HD cells\n(AND)")

    ax = fig.add_subplot(spec[4]) # add an axes to the figure
    spa = cells.HDCell_OR[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["HD","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
        
    else:
        labels = ["Ns", "HD"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]
        
    ax.pie(spa,labels = labels,autopct='%.0f%%', colors = colors)
    ax.text(-1,-1.8, "HD cells\n(OR)")
    
#############################################################################
######## Border functions ###################################################
#############################################################################

def plotBorderDistribution(ax,border,score="borderScore"):
    ax.hist(border,bins=np.linspace(-1,1,25))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    if score == "borderScore":
        ax.set_xlabel("Border score")
    elif score == "borderHalfScore":
        ax.set_xlabel("Border half score")
    else:
        pass
    ax.set_xlim(0,1)
    ax.set_xticks(np.linspace(-1,1,3))
    ax.set_ylabel("Neurons")
def plotBorderShuffleExample(ax,dfBorderScore,cluId,trialNo=0,score="borderScore"):
    
    trialCode = "circ80_{}".format(trialNo)
    
    observed = dfBorderScore[score][(dfBorderScore.shuffling==False)&
                                      (dfBorderScore.cluId==cluId)&
                                      (dfBorderScore.trialCode==trialCode)].values.item()
    shuf = dfBorderScore[score][(dfBorderScore.shuffling==True)&
                              (dfBorderScore.cluId==cluId)&
                              (dfBorderScore.trialCode==trialCode)].values
    
    h = ax.hist(shuf,label="Shuf.")
    ax.plot([observed,observed],[0,np.max(h[0])],label="Obs.")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    if score == "borderScore":
        ax.set_xlabel("Border score")
    elif score == "borderHalfScore":
        ax.set_xlabel("Border half score")
    else:
        pass
    ax.set_xticks(np.linspace(-0,1,3))
    ax.set_ylabel("Shuffles")
    ax.legend()
def plotBorderScoreStability(ax,dfBorderScore,score, usableCellIndices):
    Border = dfBorderScore[score][(dfBorderScore.shuffling==False)&(dfBorderScore.trialCode=="circ80_0")]
    Border0 = Border[usableCellIndices]
    Border = dfBorderScore[score][(dfBorderScore.shuffling==False)&(dfBorderScore.trialCode=="circ80_1")]
    Border1 = Border[usableCellIndices]
    
    ax.scatter(Border0,Border1,s=2,alpha=0.3)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks(np.linspace(-1,1,3))
    ax.set_yticks(np.linspace(-1,1,3))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    if score == "borderScore":
        ax.set_xlabel("Border score trial 0")
        ax.set_ylabel("Border score trial 1")
    elif score == "borderHalfScore":
        ax.set_xlabel("Border half score t0")
        ax.set_ylabel("Border half score t1")
    else:
        pass
    
def summaryPlotBorder(gs,fig,dfBorderScore,cells):
    nPlots = 5
    spec = gridspec.GridSpecFromSubplotSpec(1,nPlots, subplot_spec=gs[:])
    usableCellIndices = cells.usable.to_numpy()
    
    ax = fig.add_subplot(spec[0]) # add an axes to the figure

    # Distribution of all info scores in the first circ80 trial, for all clean cells
    Border = dfBorderScore.borderScore[(dfBorderScore.shuffling==False)&(dfBorderScore.trialCode=="circ80_0")]
    Border = Border[usableCellIndices]
    plotBorderDistribution(ax,Border)

    # Stability
    ax = fig.add_subplot(spec[1]) # add an axes to the figure
    plotBorderScoreStability(ax,dfBorderScore, "borderScore", usableCellIndices)

    # Example
    ax = fig.add_subplot(spec[2]) # add an axes to the figure
    cluId = "jp3129-10062022-0107_159"
    plotBorderShuffleExample(ax,dfBorderScore,cluId)

    ax = fig.add_subplot(spec[3]) # add an axes to the figure
    spa = cells.borderCell_AND[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["B","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
    else:
        labels = ["Ns", "B"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]

    ax.pie(spa,labels = labels,autopct='%.0f%%')
    ax.text(-1,-1.8, "Border cells\n(AND)")


    ax = fig.add_subplot(spec[4]) # add an axes to the figure
    spa = cells.borderCell_OR[cells.usable].value_counts()
    if spa.index[0]:
        labels = ["B","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
    else:
        labels = ["Ns", "B"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]
    ax.pie(spa,labels = labels,autopct='%.0f%%',colors=colors)
    ax.text(-1,-1.8, "Border cells\n(OR)")
    
    
###################
### Speed scores
###################

def plotSpeedDistribution(ax,speed):
    ax.hist(speed,bins=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("Speed score")
    ax.set_ylabel("Neurons")
    ax.set_xlim(-1,1)
def plotspeedShuffleExample(ax,dfspeedScore,cluId,trialNo=0):

    trialCode = "circ80_{}".format(trialNo)
    observed = dfspeedScore.speedScore[(dfspeedScore.shuffling==False)&
                                      (dfspeedScore.cluId==cluId)&
                                      (dfspeedScore.trialCode==trialCode)].values.item()
    shuf = dfspeedScore.speedScore[(dfspeedScore.shuffling==True)&
                              (dfspeedScore.cluId==cluId)&
                              (dfspeedScore.trialCode==trialCode)].values
    
    h = ax.hist(shuf,label="Shuf.")
    ax.plot([observed,observed],[0,np.max(h[0])],label="Obs.")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("Speed score")
    #ax.set_xticks(np.linspace(0,1,2))
    ax.set_ylabel("Shuffles")
    ax.legend()
    
def plotSpeedScoreStability(ax,dfspeedScore,usableCellIndices):
    speed = dfspeedScore.speedScore[(dfspeedScore.shuffling==False)&(dfspeedScore.trialCode=="circ80_0")]
    speed0 = speed[usableCellIndices]
    speed = dfspeedScore.speedScore[(dfspeedScore.shuffling==False)&(dfspeedScore.trialCode=="circ80_1")]
    speed1 = speed[usableCellIndices]
    
    ax.scatter(speed0,speed1,s=2,alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.set_xlabel("speed score trial 0")
    ax.set_ylabel("speed score trial 1")

def summaryPlotSpeed(gs,fig,dfspeedScore,cells):
    nPlots = 5
    spec = gridspec.GridSpecFromSubplotSpec(1,nPlots, subplot_spec=gs[:])
    
    usableCellIndices = cells.usable_auto_RF.to_numpy()
    ### plot the speed distribution 
    ax = fig.add_subplot(spec[0]) # add an axes to the figure
    # Distribution of all info scores in the first circ80 trial, for all clean cells
    speed = dfspeedScore.speedScore[(dfspeedScore.shuffling==False)&(dfspeedScore.trialCode=="circ80_0")]
    speed = speed[usableCellIndices]
    plotSpeedDistribution(ax,speed)
    
    
    # Stability
    ax = fig.add_subplot(spec[1]) # add an axes to the figure
    plotSpeedScoreStability(ax,dfspeedScore,usableCellIndices)

    # Example
    ax = fig.add_subplot(spec[2]) # add an axes to the figure
    cluId = "mn8578-30112021-0107_22"
    plotspeedShuffleExample(ax,dfspeedScore,cluId)


    ax = fig.add_subplot(spec[3]) # add an axes to the figure
    spa = cells.speedCell_AND[cells.usable_auto_RF].value_counts()
    if spa.index[0]:
        labels = ["SS","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
    else:
        labels = ["Ns", "SS"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]
    ax.pie(spa,labels = labels,autopct='%.0f%%',colors=colors)
    ax.text(-1,-1.8, "Speed\nselective (AND)")

    ax = fig.add_subplot(spec[4]) # add an axes to the figure
    spa = cells.speedCell_OR[cells.usable_auto_RF].value_counts()
    if spa.index[0]:
        labels = ["SS","Ns"]
        colors = [sns.color_palette()[1],sns.color_palette()[0]]
    else:
        labels = ["Ns", "SS"]
        colors = [sns.color_palette()[0],sns.color_palette()[1]]
    ax.pie(spa,labels = labels,autopct='%.0f%%',colors=colors)
    ax.text(-1,-1.8,"Speed\nselective (OR)")
