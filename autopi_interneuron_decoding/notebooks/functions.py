import cv2
import torch
import numpy as np
from spikeA.Neuron import Simulated_place_cell, Simulated_grid_cell
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import datetime
import pandas as pd
import os

# Location of animal position files
firstRF = "../data/jp451_lstm_firstRF.npy"
lastRF = "../data/jp451_lstm_lastRF.npy"

# GLOBAL VARIABLES
GLOBALFONTSIZE = 12
plt.rcParams['figure.dpi'] = 100

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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.axis('off')


class RigidGridCellModel(torch.nn.Module):
    """
    Model with one spacing, one orientation and one peak rate
    """
    def __init__(self,period,orientation,peak_rate,offset):
        super().__init__()

        period = torch.tensor([period],requires_grad =True,dtype=torch.float32) # model parameters
        offset = torch.tensor(offset,requires_grad =True,dtype=torch.float32) # model parameters
        peak_rate = torch.tensor([peak_rate],requires_grad =True,dtype=torch.float32) # model parameters
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        

        self.period = torch.nn.Parameter(period)
        self.offset = torch.nn.Parameter(offset)
        self.peak_rate = torch.nn.Parameter(peak_rate)
        
        # orientation
        self.ori_scalling= 0.01 # to make the gradient for orientation similar to other parameters
        ori = torch.tensor([orientation/self.ori_scalling], requires_grad=True,dtype=torch.float32) # start with 60 degree orientation
        self.ori = torch.nn.Parameter(ori)
       
        
        ## matrix to get the cos and sin component for our 2,1 projection matrix
        self.myMatCos = torch.tensor([[1],[0]],dtype=torch.float32)
        self.myMatSin = torch.tensor([[0],[1]],dtype=torch.float32)
       
        
    def length_to_angle(self,x,period):
        xr = x/period*np.pi*2
        return (torch.atan2(torch.sin(xr), torch.cos(xr)))

    def forward(self, X):
       
        # matrix to project onto the x axis and keep only the x coordinate
        self.sori = self.ori * self.ori_scalling
       
        self.Rx0 = self.myMatCos @ torch.cos(-self.sori[0].reshape(1,1))+ self.myMatSin @ -torch.sin(-self.sori[0].reshape(1,1)) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
        self.Rx1 = self.myMatCos @ torch.cos(- (self.sori[0].reshape(1,1)+self.pi/3)) + self.myMatSin @ -torch.sin(-(self.sori[0].reshape(1,1)+self.pi/3))
        self.Rx2 = self.myMatCos @ torch.cos(- (self.sori[0].reshape(1,1)+self.pi/3*2)) + self.myMatSin @ -torch.sin(-(self.sori[0].reshape(1,1)+self.pi/3*2))
         
        
        # distance in cm along each axis
        d0 = X @ self.Rx0
        d1 = X @ self.Rx1
        d2 = X @ self.Rx2
        
        c0 = self.length_to_angle(d0, self.period[0])
        c1 = self.length_to_angle(d1, self.period[0]) 
        c2 = self.length_to_angle(d2, self.period[0])

        # deal with the offset, project on each vector 
        d0 = self.offset @ self.Rx0
        d1 = self.offset @ self.Rx1
        d2 = self.offset @ self.Rx2

        # offset as angle
        a0 = self.length_to_angle(d0, self.period[0])
        a1 = self.length_to_angle(d1, self.period[0])
        a2 = self.length_to_angle(d2, self.period[0])
      
        rateC0 = torch.cos(c0-a0)
        rateC1 = torch.cos(c1-a1)
        rateC2 = torch.cos(c2-a2)

        rate = (rateC0+rateC1+rateC2+1.5)/4.5*self.peak_rate
        return rate

    def modelParamToGridParam(self):
        """
        Return the grid cell parameters in a dictionary
        """
        myIter = iter(self.parameters())
        pred_grid_param = {}
        period = next(myIter).detach().numpy()
        pred_grid_param["period"] = np.array([period[0],period[0],period[0]])
        pred_grid_param["offset"] = next(myIter).detach().numpy()
        pred_grid_param["peak_rate"] = next(myIter).detach().numpy()
        ori = next(myIter).detach().numpy() * self.ori_scalling
        pred_grid_param["orientation"] = np.array([ori[0],ori[0]+np.pi/3,ori[0]+np.pi/3*2])
        return pred_grid_param


def training_loop_grid_parameters(n_epochs, model, optimizer, loss_fn, X,y,verbose=True):
    
    for epoch in range (n_epochs):

        #for X,y in data_loader:
        optimizer.zero_grad()
        yhat = model(X)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()

        if epoch % 1000 ==0:
            if verbose:
                print("Epoch: {}, Loss: {}".format(epoch,loss))
                print("Parameters")
                pred_grid_param = model.modelParamToGridParam()
                print(pred_grid_param)
                for param in model.parameters():
                    print(param)
                print("Gradients")
                for param in model.parameters():
                    print(param.grad)
                print("")

        if loss < 0.0001:
            if verbose:
                print("Final loss:", loss)
            return loss


    return loss


class GridCellModel(torch.nn.Module):
    """
    Model with 3 period, 3 orientation and 1 peak rate.
    """
    def __init__(self,period,orientation,peak_rate,offset):
        super().__init__()

        period = torch.tensor(period,requires_grad =True,dtype=torch.float32) # model parameters
        offset = torch.tensor(offset,requires_grad =True,dtype=torch.float32) # model parameters
        peak_rate = torch.tensor(peak_rate,requires_grad =True,dtype=torch.float32) # model parameters
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        

        self.period = torch.nn.Parameter(period)
        self.offset = torch.nn.Parameter(offset)
        self.peak_rate = torch.nn.Parameter(peak_rate)
        
        # orientation
        self.ori_scalling= 0.01 # to make the gradient for orientation similar to other parameters
        ori = torch.tensor(orientation/self.ori_scalling, requires_grad=True,dtype=torch.float32) # start with 60 degree orientation
        self.ori = torch.nn.Parameter(ori)
       
        
        ## matrix to get the cos and sin component for our 2,1 projection matrix
        self.myMatCos = torch.tensor([[1],[0]],dtype=torch.float32)
        self.myMatSin = torch.tensor([[0],[1]],dtype=torch.float32)
       
        
    def length_to_angle(self,x,period):
        xr = x/period*np.pi*2
        return (torch.atan2(torch.sin(xr), torch.cos(xr)))

    def forward(self, X):
       
        # matrix to project onto the x axis and keep only the x coordinate
        self.sori = self.ori * self.ori_scalling
       
        self.Rx0 = self.myMatCos @ torch.cos(-self.sori[0].reshape(1,1))+ self.myMatSin @ -torch.sin(-self.sori[0].reshape(1,1)) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
        self.Rx1 = self.myMatCos @ torch.cos(-self.sori[1].reshape(1,1)) + self.myMatSin @ -torch.sin(-self.sori[1].reshape(1,1))
        self.Rx2 = self.myMatCos @ torch.cos(-self.sori[2].reshape(1,1)) + self.myMatSin @ -torch.sin(-self.sori[2].reshape(1,1))
         
        
        # distance in cm along each axis
        d0 = X @ self.Rx0
        d1 = X @ self.Rx1
        d2 = X @ self.Rx2
        
        c0 = self.length_to_angle(d0, self.period[0])
        c1 = self.length_to_angle(d1, self.period[1]) 
        c2 = self.length_to_angle(d2, self.period[2])

        # deal with the offset, project on each vector 
        d0 = self.offset @ self.Rx0
        d1 = self.offset @ self.Rx1
        d2 = self.offset @ self.Rx2

        # offset as angle
        a0 = self.length_to_angle(d0, self.period[0])
        a1 = self.length_to_angle(d1, self.period[1])
        a2 = self.length_to_angle(d2, self.period[2])
      
        rateC0 = torch.cos(c0-a0)
        rateC1 = torch.cos(c1-a1)
        rateC2 = torch.cos(c2-a2)

        rate = (rateC0+rateC1+rateC2+1.5)/4.5*self.peak_rate
        return rate

    def modelParamToGridParam(self):
        """
        Return the model parameters as a dictionary
        """
        myIter = iter(self.parameters())
        pred_grid_param = {}
        pred_grid_param["period"] = next(myIter).detach().numpy()
        pred_grid_param["offset"] = next(myIter).detach().numpy()
        pred_grid_param["peak_rate"] = next(myIter).detach().numpy()

        ori = next(myIter).detach().numpy() * self.ori_scalling
        orientation = np.array([ori[0],ori[1],ori[2]])

        pred_grid_param["orientation"] = orientation
        return pred_grid_param


def figurePanelDefaultSize():
    """
    Use to keep the size of panels similar across figures
    """
    return (1.8,1.8)



def invalidate_surrounding(myMap,cm_per_bin=3, valid_radius_cm=50):
    xs,ys = np.meshgrid(np.arange(0,myMap.shape[0]),np.arange(0,myMap.shape[1]))
    midPoint=(myMap.shape[0]/2,myMap.shape[1]/2)
    distance = np.sqrt((xs.T-midPoint[0])**2 + (ys.T-midPoint[1])**2) * cm_per_bin
    myMap[distance>valid_radius_cm]=np.nan


def fit_grid_parameter_from_grid_cell_activity(n,ap,apSim,cm_per_bin = 3,xy_range=np.array([[-50,-90],[50,60]]),n_epochs=5000):
    """
    Function that finds the best grid cell model parameters (period, orientation, peak rate, offset) that predict the firing rate of the neuron
    
    The model takes x,y position as input and try to predict the firing rate of the neuron.
    
    We start with an estimate from firing rate map (offset) and spatial autocorrelation (orientation, spacing), then fit some grid cell models with gradient descent.
    
    
    """
    interval = n.ap.intervals.inter

    rowSize,colSize= 2,2
    
    ap.set_intervals(interval)
    n.spike_train.set_intervals(interval)
    n.spatial_properties.set_intervals(interval)
    n.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    n.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)
    n.spatial_properties.grid_score()
    n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,outside_interval_solution="remove")

    gcSpacing = n.spatial_properties.grid_info()[0]
    gcPeakLocation = n.spatial_properties.firing_rate_map_peak_location()

    # get a template spatial autocorrelation with orientation set at 0
    grid_param = {}
    period = gcSpacing * np.cos(np.pi/6)
    grid_param["period"] = np.array([period,period,period])
    grid_param["offset"] = gcPeakLocation[1]
    grid_param["peak_rate"] = 25
    grid_param["orientation"] = np.array([0,np.pi/3,np.pi/3*2]) # 30 degree orientation means that the field to the right will be at 1,0
    apSim.set_intervals(interval)

    print(grid_param)
    sgc = Simulated_grid_cell(name="pc1",
                              offset=grid_param["offset"],
                              orientation=grid_param["orientation"],
                              period=grid_param["period"],
                              peak_rate=grid_param["peak_rate"],
                              ap=apSim)

    apSim.set_intervals(interval)
    sgc.spike_train.set_intervals(interval)
    sgc.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    sgc.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)

    refAuto = n.spatial_properties.spatial_autocorrelation_map.copy()
    invalidate_surrounding(myMap=refAuto,cm_per_bin=cm_per_bin,valid_radius_cm=gcSpacing+gcSpacing*0.4)
    refAutoSgc = sgc.spatial_properties.spatial_autocorrelation_map.copy()
    invalidate_surrounding(myMap=refAutoSgc,cm_per_bin=cm_per_bin,valid_radius_cm=gcSpacing+gcSpacing*0.4)
    autoStackReal = np.expand_dims(refAuto,0)
    autoStackSim = np.expand_dims(refAutoSgc,0)
    
    rot,cor = rotation_correlations(autoStackReal,autoStackSim,minRotation=0,maxRotation=np.pi+0.174,nRotations=180+10)
    peak_indices,_ = find_peaks(cor,height=0.3,distance=10)
    deltas = rot[peak_indices]
    
    print("Results after rotating the spatial autocorrelation")
    print("First axis at ",deltas[0])
    ncols=1
    nrows=1
    fig = plt.figure(figsize=(ncols*4, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    ax = fig.add_subplot(mainSpec[0])
    ax.plot(rot,cor)
    ax.scatter(rot[peak_indices],cor[peak_indices],color="red",s=10)
    ax.set_xlabel("Rotation")
    ax.set_ylabel("r value")
    plt.show()
    
    
    if len(deltas) > 4:
        raise ValueError("Expect less than 5 peaks while rotating spatial autocorrelation but got {}".format(len(deltas)))
    
    if len(deltas) < 3:
        raise ValueError("Expect at least 5 peaks while rotating spatial autocorrelation but got {}".format(len(deltas)))
    grid_param["orientation"] = np.array(deltas[0:3]) # take the first 3
    
   
    # get a simulated grid cells with our initial grid parameters
    sgc = Simulated_grid_cell(name="pc1",
                              offset=grid_param["offset"],
                              orientation=grid_param["orientation"],
                              period=grid_param["period"],
                              peak_rate=grid_param["peak_rate"],
                              ap=apSim)

    apSim.set_intervals(interval)
    sgc.spike_train.set_intervals(interval)
    sgc.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    
    print("Visualize real and simulated grid pattern before fitting")
    ncols=2
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    ax = fig.add_subplot(mainSpec[0])
    plotMap(ax,n.spatial_properties.firing_rate_map,
            title="Data - {:.2f} Hz".format(np.nanmax(n.spatial_properties.firing_rate_map)),
            titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
    ax = fig.add_subplot(mainSpec[1])
    plotMap(ax,sgc.spatial_properties.firing_rate_map,
            title="Sim - {:.2f} Hz".format(np.nanmax(sgc.spatial_properties.firing_rate_map)),
            titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
    plt.show()
    
    # get the data that will be used for modelling
    ap.set_intervals(interval)
    
    # trick to get aligned ifr and pose data
    modInterval = interval.copy()
    modInterval[0,0] = ap.pose[0,0]
    modInterval[0,1] = ap.pose[-1,0]+0.00000001
    
    n.spike_train.set_intervals(modInterval)
    n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,shift_start_time = -0.0099999999,
                                            outside_interval_solution="remove")

    poseLong = ap.pose[:,1:3]
    rateLong = n.spike_train.ifr[0].copy()
        
    # remove np.nan
    keepIndices = ~np.any(np.isnan(poseLong),1)
    rate = rateLong[keepIndices]
    pose = poseLong[keepIndices]
    #print(rate.shape, pose.shape)
    
    # transform to tensors
    tpose = torch.tensor(pose,dtype=torch.float32)
    trate = torch.tensor(np.expand_dims(rate,1),dtype=torch.float32)
    print("Shape of tensors used for training:",tpose.shape,trate.shape)
    
    
    # get a rigid grid model
    rgcModel = RigidGridCellModel(period=grid_param["period"][0], 
                              peak_rate=grid_param["peak_rate"],
                              orientation=grid_param["orientation"][0],
                              offset=grid_param["offset"])
    
    grid_param_model_start = rgcModel.modelParamToGridParam()
    grid_param_model_start
    print("Fitting rigid grid cell model")
    loss_rigid = training_loop_grid_parameters(n_epochs = n_epochs,
                          model=rgcModel,
                          optimizer=torch.optim.Adam(rgcModel.parameters(),lr=0.01),
                          loss_fn = torch.nn.MSELoss(),
                          X = tpose,
                          y = trate,
                         verbose=False)
    loss_rigid = loss_rigid.detach().numpy()
    print("Loss after rigid model fitting:",loss_rigid)
    
    grid_param_model_rigid = rgcModel.modelParamToGridParam()
    sgcRigid = Simulated_grid_cell(name="pc1",
                          offset=grid_param_model_rigid["offset"],
                          orientation=grid_param_model_rigid["orientation"],
                          period=grid_param_model_rigid["period"],
                          peak_rate=grid_param_model_rigid["peak_rate"],
                          ap=apSim)
    
    sgcRigid.spatial_properties.firing_rate_map_2d(cm_per_bin=cm_per_bin,smoothing_sigma_cm=3, xy_range=xy_range)
    rStart = sgcRigid.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                            map2=sgc.spatial_properties.firing_rate_map)
    rRigid = sgcRigid.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                            map2=sgcRigid.spatial_properties.firing_rate_map)
    print("Improvement of firing rate maps correlation after rigid fitting: from {:.3f} to {:.3f}".format(rStart, rRigid))
    
    gcModel = GridCellModel(period=grid_param_model_rigid["period"], 
                        peak_rate=grid_param_model_rigid["peak_rate"], 
                        orientation=grid_param_model_rigid["orientation"],
                        offset=grid_param_model_rigid["offset"])
    print("Fitting more flexible grid cell model")
    loss_flexible = training_loop_grid_parameters(n_epochs = n_epochs,
                          model=gcModel,
                          optimizer=torch.optim.Adam(gcModel.parameters(),lr=0.01),
                          loss_fn = torch.nn.MSELoss(),
                          X = tpose,
                          y = trate,
                         verbose=False)
    loss_flexible = loss_flexible.detach().numpy()
    print("Loss after flexible model fitting:",loss_flexible)
    grid_param_model_flexible = gcModel.modelParamToGridParam()
    
    sgcFlexible = Simulated_grid_cell(name="pc1",
                          offset=grid_param_model_flexible["offset"],
                          orientation=grid_param_model_flexible["orientation"],
                          period=grid_param_model_flexible["period"],
                          peak_rate=grid_param_model_flexible["peak_rate"],
                          ap=apSim)
    sgcFlexible.spatial_properties.firing_rate_map_2d(cm_per_bin=cm_per_bin,smoothing_sigma_cm=3, xy_range=xy_range)
    rFlexible = sgc.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                                map2=sgcFlexible.spatial_properties.firing_rate_map)
    print("Improvement of firing rate maps correlation after flexible fitting: from {:.3f} to {:.3f}".format(rRigid,rFlexible))
    
    print("Comparison of firing rate maps after fitting different models")
    
    ncols=4
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=1)

    ax = fig.add_subplot(mainSpec[0])
    plotMap(ax,n.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Original",fontsize=9)

    ax = fig.add_subplot(mainSpec[1])
    plotMap(ax,sgc.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated not fitted\n {:.3f}".format(rStart),fontsize=9)

    ax = fig.add_subplot(mainSpec[2])
    plotMap(ax,sgcRigid.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated rigid fit\n {:.3f}".format(rRigid),fontsize=9)

    ax = fig.add_subplot(mainSpec[3])
    plotMap(ax,sgcFlexible.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated flexible fit\n {:.3f}".format(rFlexible),fontsize=9)

    plt.show()
    
    res = {"name":n.name,
           "grid_param_initial":grid_param,
           "grid_param_model_rigid": grid_param_model_rigid,
           "loss_rigid": loss_rigid,
           "r_rigid": rRigid,
           "grid_param_model_flexible": grid_param_model_flexible,
           "loss_flexible": loss_flexible,
           "r_flexible": rFlexible}

    n.spike_train.unset_intervals()
    n.spike_train.set_intervals(interval)
    
    #n.spatial_properties.firing_rate_map_2d()
    
    return res


def map_cor(a,b):
    """
    Correlation coefficient between two firing rate maps
    
    Arguments:
    a: 2D np.array (map1)
    b: 2D np.array (map2)
    
    Returns:
    Pearson correlation coefficient between a and b
    """
    a = a.flatten()
    b = b.flatten()
    indices1 = np.logical_and(~np.isnan(a), ~np.isnan(b))
    indices2 = np.logical_and(~np.isinf(a), ~np.isinf(b))
    indices = np.logical_and(indices1,indices2)
    if np.sum(indices)<2:
        return np.nan
    r,p = pearsonr(a[indices],b[indices])
    return r


def rotate_map(a,rotation_radian=np.pi/6):
    """
    Rotate the values in a map around the center of the map
    
    Arguments:
    a: 2D Numpy array
    rotation_radian: angle of the rotation in radian, positive is anti-clockwise
    
    Return:
    2D Numpy array with values rotated
    """

    (h, w) = a.shape
    (cX, cY) = (w // 2, h // 2) # center of the rotation
    degree = rotation_radian/(2*np.pi)*360

    # rotate by degree°, same scale
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = degree, scale = 1.0)
    a_rotated = cv2.warpAffine(a, M, (w, h), borderValue = np.nan)
    return a_rotated

def rotation_correlations(ccStack1,ccStack2,minRotation=-np.pi,maxRotation=np.pi,nRotations=72):
    """
    We rotate the maps of ccStack2 and perform a correlation coefficient between individual cc.
    We create a 2D array of correlation coefficients (rotation x cell_pairs)
    Then we get the mean coefficient at each rotation
    """
    rotations = np.linspace(minRotation,maxRotation,nRotations)
    rotatedStack = np.empty_like(ccStack2)
    corrValues = np.empty((rotations.shape[0],ccStack2.shape[0]))

    for i,r in enumerate(rotations):
        for j in range(ccStack2.shape[0]): # rotate individual maps
            rotatedStack[j,:,:] = rotate_map(ccStack2[j,:,:],rotation_radian=r)
            corrValues[i,j] = map_cor(ccStack1[j,:,:],rotatedStack[j,:,:])

    peaks = np.mean(corrValues,axis=1)
    return rotations, peaks


def find_grid_cell_parameters(neuron_list,save=False):
    cm_per_bin = 3
    xy_range=np.array([[-50,-90],[50,60]])
    

    # Calculate firing rate maps and spatial autocorrelation to have a reasonable estimate of the grid parameters
    for n in neuron_list:
        n.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
        n.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)
        n.spatial_properties.grid_score()
        n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,outside_interval_solution="remove")
    
    # our rough estimate of spacing, orientation and offset, used as a starting point
    gcSpacing = [n.spatial_properties.grid_info()[0] for n in neuron_list]
    gcOrientation = [n.spatial_properties.grid_info()[1] for n in neuron_list]
    gcPeakLocation = [n.spatial_properties.firing_rate_map_peak_location() for n in neuron_list]

    print(n.spatial_properties.grid_info())
    
    # show the firing rate maps
    print("Firing rate maps used to find grid cell parameters")
    rowSize,colSize= 1.6,1.6
    ncols=6
    nrows=int(np.ceil(len(neuron_list)/ncols))
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    for i,n in enumerate(neuron_list):
        c= int(i % ncols)
        r= int(i / ncols)
        ax = fig.add_subplot(mainSpec[r,c])
        plotMap(ax,n.spatial_properties.firing_rate_map,title="{} - {:.2f} Hz".format(n.name,np.nanmax(n.spatial_properties.firing_rate_map)),titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
        xy,grid_peak_location =  gcPeakLocation[i]
        ax.scatter(xy[0],xy[1],color="black", s = 35)
        ax.scatter(xy[0],xy[1],color="red", s = 10)
    
    # plot the distribution of spacing
    rowSize,colSize= 1.8,1.8
    ncols=1
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)

    ax = fig.add_subplot(mainSpec[0])
    ax.hist(gcSpacing)
    ax.set_xlim(30,50)
    ax.set_xlabel("Grid spacing (cm)")
    ax.set_ylabel("Neurons")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    

    # do the model fitting, the function is in the file function of the path_reconstruction directory
    res = [fit_grid_parameter_from_grid_cell_activity(n,n.ap,n.ap) for n in tqdm(neuron_list)]

    if save:
        #save the results into a file for quick access
        fn = "../data/grid_cell_parameters.pkl"
        print("Saving:",fn)
        with open(fn, 'wb') as handle:
            pickle.dump(res, handle)
    
    
    return res


def poseToGridSpace(pose,period=np.array([40,40,40]),orientation=np.array([0,np.pi/3,np.pi/3*2])):
    """
    Function to transfrom the x,y position of the mouse to 
    a position within the internal representation of grid cells. 
    
    The internal representation is 3 angles (x,y,z) which represents the distance along 3 axes
    The 3 axes are at 60 degrees of each other.
    To get from distance to angle, we get the modulo of the distance and the underlying spacing.
    Then set the range to -np.pi to pi. 
    Each angle is represented by a cos and sin component to avoid discontinuity (0-360).
    
    Arguments:
    pose: 2D numpy array with x and y position, 2 columns
    period: spacing of the underlying band pattern
    orientation: angle of the 3 main axes of the grid pattern
    """
    
    Rx0 = np.array([[np.cos(-orientation[0])],[-np.sin(-orientation[0])]]) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0 
    Rx1 = np.array([[np.cos(-orientation[1])],[-np.sin(-orientation[1])]])
    Rx2 = np.array([[np.cos(-orientation[2])],[-np.sin(-orientation[2])]])
    
    d0 = pose @ Rx0
    d1 = pose @ Rx1
    d2 = pose @ Rx2
        
    c0 = (d0 % period[0])/period[0] * np.pi*2 - np.pi
    c1 = (d1 % period[1])/period[1] * np.pi*2 - np.pi
    c2 = (d2 % period[2])/period[2] * np.pi*2 - np.pi
    
    c0c = np.cos(c0)
    c0s = np.sin(c0)
    c1c = np.cos(c1)
    c1s = np.sin(c1)
    c2c = np.cos(c2)
    c2s = np.sin(c2)
    
    return np.stack([c0c.flatten(),
                     c0s.flatten(),
                     c1c.flatten(),
                     c1s.flatten(),
                     c2c.flatten(),
                     c2s.flatten()]).T

class NeuralDataset(torch.utils.data.Dataset):
    """
    Represent our pose and neural data.
    
    """
    def __init__(self, ifr, pose, time, seq_length,ifr_normalization_means=None,ifr_normalization_stds=None):
        """
        ifr: instantaneous firing rate
        pose: position of the animal
        seq_length: length of the data passed to the network
        """
        super(NeuralDataset, self).__init__()
        self.ifr = ifr.astype(np.float32)
        self.pose = pose.astype(np.float32)
        self.time = time.astype(np.float32)
        self.seq_length = seq_length
        
        self.ifr_normalization_means=ifr_normalization_means
        self.ifr_normalization_stds=ifr_normalization_stds
        
        self.normalize_ifr()
        
        self.validIndices = np.argwhere(~np.isnan(self.pose[:,0]))
        self.validIndices = self.validIndices[self.validIndices>seq_length] # make sure we have enough neural dat leading to the pose
   
        
    def normalize_ifr(self):
        """
        Set the mean of each neuron to 0 and std to 1
        Neural networks work best with inputs in this range
        Set maximal values at -5.0 and 5 to avoid extreme data points
        
        ###########
        # warning #
        ###########
        
        In some situation, you should use the normalization of the training set to normalize your test set.
        For instance, if the test set is very short, you might have a very poor estimate of the mean and std, or the std might be undefined if a neuron is silent.
        """
        if self.ifr_normalization_means is None:
            self.ifr_normalization_means = self.ifr.mean(axis=0)
            self.ifr_normalization_stds = self.ifr.std(axis=0)
            
        self.ifr = (self.ifr-np.expand_dims(self.ifr_normalization_means,0))/np.expand_dims(self.ifr_normalization_stds,axis=0)
        self.ifr[self.ifr> 5.0] = 5.0
        self.ifr[self.ifr< -5.0] = -5.0
        
        
    def __len__(self):
        return len(self.validIndices)
    
    def __getitem__(self,index):
        """
        Function to get an item from the dataset
        
        Returns pose, neural data
        
        """
        neuralData = self.ifr[self.validIndices[index]-self.seq_length:self.validIndices[index],:]
        pose = self.pose[self.validIndices[index]:self.validIndices[index]+1,:] #
        time = self.time[self.validIndices[index]:self.validIndices[index]+1]
        
        return torch.from_numpy(neuralData), torch.from_numpy(pose).squeeze(), torch.from_numpy(time) # we only need one channel for the mask



class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs, sequence_length,device):
        super(LSTM,self).__init__()
        """
        For more information about nn.LSTM -> https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size,hidden_size, num_layers, batch_first=True)
        # input : batch_size x sequence x features
        self.device = device
        self.fc = torch.nn.Linear(hidden_size*sequence_length, num_outputs) # if you onely want to use the last hidden state (hidden_state,num_classes)
        
    def forward(self,x):
        
        h0 =  torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(self.device)
        c0 =  torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(self.device) 
        out, _ = self.lstm(x,(h0,c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out) #if you want to use only the last hidden state, remove previous line, # out = self.fc(out[:,-1,:])
        
        return out

def lossOnTestDataset(model,test_data_loader,device,loss_fn):
    model.eval()
    loss_test = 0
    with torch.no_grad():
        for imgs, labels, time in test_data_loader: # mini-batches with data loader, imgs is sequences of brain activity, labels is position of mouse
            imgs = imgs.to(device=device) # batch x chan x 28 x 28 to batch x 28 x 28
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            loss_test += loss.item()
    a = model.train()
    return loss_test/len(test_data_loader)

def training_loop(n_epochs,
                 optimizer,
                 model,
                 loss_fn,
                 train_data_loader,
                 test_data_loader,
                 config,
                  device,
                 verbose=False,best_loss=float('inf'),
                 best_model_state=None):
    
    if verbose:
        print("Training starting at {}".format(datetime.datetime.now()))
    testLoss =  lossOnTestDataset(model,test_data_loader,device,loss_fn)
    trainLoss = lossOnTestDataset(model,train_data_loader,device,loss_fn)
    if verbose:
        print("Test loss without training: {}".format(testLoss))
    
    df = pd.DataFrame({"epochs": [0],
                       "seq_length": config["seq_length"],
                       "n_cells": config["n_cells"],
                       "hidden_size": config["hidden_size"],
                       "num_layers": config["num_layers"],
                      "learning_rate": config["learning_rate"],
                      "batch_size": config["batch_size"],
                      "train_loss": trainLoss,
                      "test_loss": testLoss})

    for epoch in range(1,n_epochs+1):
        loss_train = 0
        for imgs, labels, time in train_data_loader: # mini-batches with data loader, imgs is sequences of brain activity, labels is position of mouse
            imgs = imgs.to(device=device) # batch x chan x 28 x 28 to batch x 28 x 28
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        testLoss = lossOnTestDataset(model,test_data_loader,device,loss_fn)
        if verbose:
            print("{} Epoch: {}/{}, Training loss: {}, Testing loss: {}".format(datetime.datetime.now(),epoch,n_epochs,loss_train/len(train_data_loader), testLoss))
        df1 = pd.DataFrame({"epochs": [epoch],
                       "seq_length": config["seq_length"],
                       "n_cells": config["n_cells"],
                       "hidden_size": config["hidden_size"],
                       "num_layers": config["num_layers"],
                      "learning_rate": config["learning_rate"],
                      "batch_size": config["batch_size"],
                      "train_loss": loss_train/len(train_data_loader),
                           "test_loss": testLoss})
        
        df = pd.concat([df, df1])

        if testLoss < best_loss:
            best_loss = testLoss
            best_model_state = model.state_dict()

    return df, best_model_state


# Function to load the train and test data
def load_position_ifr_dataset(dirPath="../data"):
    fn = os.path.join(dirPath,"train_ifr.pkl")
    ifrTime = pickle.load(open(fn,"rb"))
    train_ifr, train_time = ifrTime

    fn = os.path.join(dirPath,"train_pose.pkl")
    train_pose = pickle.load(open(fn,"rb"))

    fn = os.path.join(dirPath,"test_ifr.pkl")
    ifrTime = pickle.load(open(fn,"rb"))
    test_ifr, test_time = ifrTime

    fn = os.path.join(dirPath,"test_pose.pkl")
    test_pose = pickle.load(open(fn,"rb"))
    
    fn = os.path.join(dirPath,"grid_cell_parameters.pkl")
    grid_param = pickle.load(open(fn,"rb"))
    print("train_ifr.shape:",train_ifr.shape)
    print("train_pose.shape:",train_pose.shape)

    oriRigid = np.stack([p["grid_param_model_rigid"]["orientation"] for p in grid_param])
    oriFlexible = np.stack([p["grid_param_model_flexible"]["orientation"] for p in grid_param])
    periodRigid = np.stack([p["grid_param_model_rigid"]["period"] for p in grid_param])
    periodFlexible = np.stack([p["grid_param_model_flexible"]["period"] for p in grid_param])


    grid_param = {
    "period": np.median(periodFlexible,axis=0),
    "orientation": np.median(oriFlexible,axis=0),
    }
    
    
    if train_ifr.shape[0] != train_pose.shape[0]:
        raise ValueError("Problem with the shape of ifr and pose object")
    if test_ifr.shape[0] != test_pose.shape[0]:
        raise ValueError("Problem with the shape of ifr and pose object")
        
    train_grid_coord = poseToGridSpace(pose=train_pose[:,1:3],
                             period=grid_param["period"],
                             orientation=grid_param["orientation"])
    
    
    test_grid_coord = poseToGridSpace(pose=test_pose[:,1:3],
                             period=grid_param["period"],
                             orientation=grid_param["orientation"])
    
    
    
    return train_ifr, train_pose, train_grid_coord, test_ifr, test_pose, test_grid_coord, grid_param, train_time, test_time



def gridSpaceToMovementPath(grid_coord,grid_period=40,orientation=0):
    """
    Function to go from grid cell coordinate (2 angles) to movement path

    gridSpace is a representation of the internal activity of the grid manifold. It has 3 dimensions that are circular. But we are only using 2 dimensions here
    When the active representation in grid space changes, we can transform this into movement in the real world.
    We don't know the absolute position of the animal, but we can recreate the movement path.

    We use 2 of the 3 components of the grid space to reconstruct the movement path.
    For each time sample, we know the movement in the grid cells space along these 2 directions.
    If we know that the mouse moved 2 cm along the first grid vector, the mouse can be at any position on a line that passes by 2*unitvector0 and is perpendicular to unitvector0
    If we know that the mouse moved 3 cm along the second grid vector, the mouse can be at any position on a line that passes by 3*unitvector1 and is perpendicular to unitvector1
    We just find the intersection of the two lines to know the movement of the mouse in x,y space.


    Arguments:
    grid_coord: is a 2D numpy array with the cos and sin component of the first 2 axes of the grid (4 columns)
    """

    # get angle from the cos and sin components
    ga0 = np.arctan2(grid_coord[:,1],grid_coord[:,0])
    ga1 = np.arctan2(grid_coord[:,3],grid_coord[:,2])

    # get how many cm per radian

    cm_per_radian = grid_period/(2*np.pi)

    # get the movement along the 3 vector of the grid
    dga0=mvtFromAngle(ga0,cm_per_radian[0])
    dga1=mvtFromAngle(ga1,cm_per_radian[1])


    # unit vector and unit vector perpendicular to the grid module orientation vectors
    uv0 = np.array([[np.cos(orientation[0]),np.sin(orientation[0])]]) # unit vector v0
    puv0 = np.array([[np.cos(orientation[0]+np.pi/2),np.sin(orientation[0]+np.pi/2)]]) # unit vector perpendicular to uv0
    uv1 = np.array([[np.cos(orientation[1]),np.sin(orientation[1])]]) # unit vector v1
    puv1 = np.array([[np.cos(orientation[1]+np.pi/2),np.sin(orientation[1]+np.pi/2)]]) # unit vector perpendicular to uv1

    # two points in the x,y coordinate system that are on a line perpendicular to v0
    p1 = np.expand_dims(dga0,1)*uv0 # x,y coordinate of movement along v0
    p2 = p1+ puv0 # a second x,y coordinate that is p1 plus a vector perpendicular to uv0

    # two points in the x,y coordinate system that are on a line perpendicular to v1
    p3 = np.expand_dims(dga1,1)*uv1 # coordinate of the point 1 on line 1
    p4 = p3+ puv1 # coordinate of point 2 on line 1

    # find the intersection between 2 lines, using 2 points that are part of line 1 and 2 points that are part of line 2
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    px_num = (p1[:,0]*p2[:,1] - p1[:,1]*p2[:,0]) * (p3[:,0]-p4[:,0]) - (p1[:,0]-p2[:,0]) * (p3[:,0]*p4[:,1] - p3[:,1]*p4[:,0]) 
    px_den = ((p1[:,0]-p2[:,0]) * (p3[:,1]-p4[:,1]) - (p1[:,1]-p2[:,1]) * (p3[:,0]-p4[:,0]))
    reconstructedX = px_num/px_den
    py_num = (p1[:,0]*p2[:,1] - p1[:,1]*p2[:,0]) * (p3[:,1]-p4[:,1]) - (p1[:,1]-p2[:,1]) * (p3[:,0]*p4[:,1] - p3[:,1]*p4[:,0]) 
    py_den = ((p1[:,0]-p2[:,0]) * (p3[:,1]-p4[:,1]) - (p1[:,1]-p2[:,1]) * (p3[:,0]-p4[:,0]))
    reconstructedY = py_num/py_den

    return np.stack([reconstructedX,reconstructedY]).T



def mvtFromAngle(ga,cm_per_radian):
    """
    Go from an angle in the one grid coordinate (one of the 3 axes) to a change in position along this axis
    """
    dga = np.diff(ga,prepend=np.nan) # this is the change in the angle
    dga = np.where(dga>np.pi,dga-2*np.pi,dga) # correct for positive jumps because of circular data
    dga = np.where(dga<-np.pi,dga+2*np.pi,dga) # correct for negative jumps
    dga = dga* cm_per_radian # transform from change in angle to change in cm
    return dga


def plot_spatial_autocorrelation(ax,sgc):
    sgc.spatial_properties.spatial_autocorrelation_map_2d()
    im = ax.imshow(sgc.spatial_properties.spatial_autocorrelation_map, cmap='jet')

    sgc.spatial_properties.calculate_doughnut()
    grid_score = sgc.spatial_properties.grid_score()
    ax.set_title("Grid score: {:.3}".format(grid_score))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Autocorrelation')
