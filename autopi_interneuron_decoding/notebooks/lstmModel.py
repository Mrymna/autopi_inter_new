import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import spikeA
from spikeA.Spike_train import Spike_train
from spikeA.Animal_pose import Animal_pose
from spikeA.Spatial_properties import Spatial_properties
from spikeA.Neuron import Simulated_place_cell, Simulated_grid_cell
from scipy.stats import poisson
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy import stats
from scipy import ndimage
from astropy.stats import circcorrcoef
from astropy import units as u
from functions import *
from scipy.ndimage import gaussian_filter1d

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import mean_squared_error 
import datetime
import traceback

from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor
from tqdm import tqdm  # Assuming you're using tqdm for progress tracking
import multiprocessing
import os

from spikeA.Session import Kilosort_session
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train_loader import Spike_train_loader
from spikeA.Cell_group import Cell_group
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Spike_train import Spike_train
from spikeA.Spike_waveform import Spike_waveform
from spikeA.Spike_waveform import Spike_waveform

from concurrent.futures import ProcessPoolExecutor
save_directory ='/home/kevin/repo/autopi_inter_new/autopi_interneuron_decoding/notebooks/Shuffled_values/'

os.makedirs(save_directory, exist_ok=True)

class NeuralDataset(torch.utils.data.Dataset):
    """
    Represent our pose and neural data.
    
    """
    def __init__(self, ifr, sin, cos, time, seq_length,ifr_normalization_means=None,ifr_normalization_stds=None):    
        """
        ifr: instantaneous firing rate
        angle: angle of the animal around the lever from -pi to pi 
        seq_length: length of the data passed to the network
        """
        super(NeuralDataset, self).__init__()
        self.ifr = ifr.astype(np.float32)
        #self.sin_cos_angles = sin_cos_angles.astype(np.float32)
        self.sin = sin.astype(np.float32)
        self.cos= cos.astype(np.float32)
        self.time = time.astype(np.float32)
        self.seq_length = seq_length
        
        self.ifr_normalization_means=ifr_normalization_means
        self.ifr_normalization_stds=ifr_normalization_stds
        
        self.normalize_ifr()
        
        #self.validIndices = np.argwhere(~np.isnan(self.angel[:]))
        self.validIndices = np.argwhere(~np.isnan(self.sin[:]))
        self.validIndices = self.validIndices[self.validIndices>seq_length] # make sure we have enough neural dat leading to the angles
   
        
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
        
        Returns angles, neural data
        
        """
        neuralData = self.ifr[self.validIndices[index]-self.seq_length:self.validIndices[index],:]
        #sin_cos_angles = self.angle[self.validIndices[index]:self.validIndices[index]+1,:]#. 2d array 
        sin= self.sin[self.validIndices[index]:self.validIndices[index]+1].squeeze()  # Squeeze to get shape [1] instead of [1, 1]
        cos= self.cos[self.validIndices[index]:self.validIndices[index]+1].squeeze()  # Squeeze to get shape [1] instead of [1, 1]
        time = self.time[self.validIndices[index]:self.validIndices[index]+1]
        
        #return torch.from_numpy(neuralData), torch.from_numpy(sin_cos_angles).squeeze(), torch.from_numpy(time) # if I return sin_cos_angle in a 2d array
        return torch.from_numpy(neuralData), torch.from_numpy(sin), torch.from_numpy(cos), torch.from_numpy(time)
    
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs, sequence_length, device):
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
        #self.fc = torch.nn.Linear(hidden_size * sequence_length, 2)  # Output two values for sin and cos but in a single 2d array  
        
    def forward(self,x):
        
        h0 =  torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(self.device)
        c0 =  torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(self.device) 
        out, _ = self.lstm(x,(h0,c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out) #if you want to use only the last hidden state, remove previous line, # out = self.fc(out[:,-1,:])
        
        return out
    
def split_to_test_andTraining(ifr, pose, sin_cos_angles, time, angles):

    split_index = int(ifr.shape[1] * 0.8)  # 80% of the data along the second axis
    train_ifr = ifr[:, :split_index]  # First 80% for training
    test_ifr = ifr[:, split_index:]
    
    train_pose= pose[:split_index, :]
    test_pose= pose[split_index:, :]
    
    train_sin_cos_angles= sin_cos_angles[:split_index, :]
    test_sin_cos_angles= sin_cos_angles[split_index:, :]
    
    train_time=time[:split_index]
    test_time= time[split_index:]
    
    train_angles=angles[:split_index]
    test_angles= angles[split_index:]
    
    return train_pose,test_pose, train_ifr, test_ifr, train_sin_cos_angles, test_sin_cos_angles, train_time,test_time, train_angles, test_angles


def get_shuffled_data(interName, sin_cos_angles, ses, min_roll_sec=20):
    # Get the interval times for the specified interName
    intervals = ses.intervalDict[interName]
    total_time_sec = intervals[:, 1][-1] - intervals[:, 0][0]

    # Ensure the total interval duration is larger than 2 * min_roll_sec
    if total_time_sec < 2 * min_roll_sec:
        raise ValueError("Total time in intervals should be larger than 2 * min_roll_sec")

    # Generate a random shift within the allowed range
    time_shift = np.random.default_rng().uniform(min_roll_sec, total_time_sec - min_roll_sec, 1)
    angles = np.arctan2(sin_cos_angles[:, 0], sin_cos_angles[:, 1])

    # Calculate time per data point
    time_per_datapoint = angles[1] - angles[0]
    shift = int(time_shift / time_per_datapoint)

    # Apply the shift to the angles
    rolled_angles = np.roll(angles, shift=shift, axis=0)

    # Convert the shifted angles back to sin and cos components
    rolled_sin_cos = np.column_stack((np.sin(rolled_angles), np.cos(rolled_angles)))

    return rolled_sin_cos

def get_test_training_datas_oneSession(sSes, ses, cells,interName, ctype= 'fs', sigma_ifr= 5, maxDistance= 18, rotationType="none", shuffle=True):

    # 1. Split the time interval for training and testing
    intervals= ses.intervalDict[interName]
    ifr, pose, time, angles, sin_cos_angles = get_session_ifr_pose_angle_aroundlever_for_model(sSes.name,ses,sSes,cells, interName= interName, ctype= ctype,  sigma_ifr= sigma_ifr, maxDistance= maxDistance, rotationType= rotationType )
    
    if shuffle: 
        sin_cos_angles_shuffled= get_shuffled_data( interName, sin_cos_angles,ses, min_roll_sec=20)
        train_pose,test_pose, train_ifr, test_ifr, train_sin_cos_angles, test_sin_cos_angles, train_time,test_time, train_angles, test_angles = split_to_test_andTraining(ifr,
                                pose, sin_cos_angles_shuffled, time, angles)
        train_ifr = train_ifr.T 
        test_ifr = test_ifr.T

    
    else: 
        
        train_pose,test_pose, train_ifr, test_ifr, train_sin_cos_angles, test_sin_cos_angles, train_time,test_time, train_angles, test_angles = split_to_test_andTraining(ifr, pose, sin_cos_angles, time, angles)
        train_ifr = train_ifr.T 
        test_ifr = test_ifr.T
    
    ####################
    ### Get the config file 
    #############
    ###################

    config = {"seq_length":20, ## is this the length of the 
              "n_cells": train_ifr.shape[1], ## it was the num of cells for one neuron is always 1
              "hidden_size" :256,
              "num_layers" : 2,
              "num_outputs" : 2, # 2 number of vector unit in x and y since I want to calculate the movement direction 
              "learning_rate" : 0.001,#0.001, ## 0.001 ( sme was )
              "dropout_rate": 0.05,
              "batch_size" :64, #64,
              "num_epochs": 100}

    #print(datetime.datetime.now(), config)
    #print(datetime.now(), config)
    # 4. Create train and test datasets
    train_dataset = NeuralDataset(
        ifr=train_ifr[:, :config["n_cells"]],
        #angle=train_angles,
        sin= train_sin_cos_angles[:,0], ### instead of 
        cos= train_sin_cos_angles[:,1],
        time=train_time,
        seq_length=config["seq_length"]
    )

    ifr_normalization_means = train_dataset.ifr_normalization_means
    ifr_normalization_stds = train_dataset.ifr_normalization_stds

    myDict = {
        "ifr_normalization_means": ifr_normalization_means,
        "ifr_normalization_stds": ifr_normalization_stds
    }

    test_dataset = NeuralDataset(
        ifr=test_ifr[:, :config["n_cells"]],
        #angle=test_angles,
        sin= test_sin_cos_angles[:,0],
        cos= test_sin_cos_angles[:,1],
        time=test_time,
        seq_length=config["seq_length"],
        ifr_normalization_means=ifr_normalization_means,
        ifr_normalization_stds=ifr_normalization_stds)

    # 5. Create data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"],
        num_workers=0, shuffle=True, pin_memory=False, drop_last=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"],
        num_workers=0, shuffle=False, pin_memory=False)

    return train_dataset, test_dataset, train_data_loader, test_data_loader, myDict, config

def lossOnTestDataset(model, test_data_loader, device, loss_fn):
    model.eval()
    loss_test = 0
    with torch.no_grad():
        for imgs, sin_labels, cos_labels, time in test_data_loader:
            imgs = imgs.to(device)
            sin_labels = sin_labels.to(device)
            cos_labels = cos_labels.to(device)

            outputs = model(imgs)  # Model outputs both sin and cos
            
            # Assume outputs[:, 0] is for sin and outputs[:, 1] is for cos
            sin_outputs = outputs[:, 0]
            cos_outputs = outputs[:, 1]

            # Calculate loss for both sin and cos
            loss_sin = loss_fn(sin_outputs, sin_labels)
            loss_cos = loss_fn(cos_outputs, cos_labels)
            
            # Total loss (sum of both)
            loss = loss_sin + loss_cos
            loss_test += loss.item()
    
    model.train()  # Set model back to training mode
    
    if len(test_data_loader) == 0:  # Handle case where the loader is empty
        return float('inf')  # Or return some placeholder value to indicate no test data
    
    return loss_test / len(test_data_loader)


def training_loop(n_epochs,
                  optimizer,
                  model,
                  loss_fn,
                  train_data_loader,
                  test_data_loader,
                  config,
                  device,
                  verbose=False,
                  best_loss=float('inf'),
                  best_model_state=None):
    
    if verbose:
        print("Training starting at {}".format(datetime.datetime.now()))
    
    # Evaluate initial loss without training
    testLoss = lossOnTestDataset(model, test_data_loader, device, loss_fn)
    trainLoss = lossOnTestDataset(model, train_data_loader, device, loss_fn)
    
    if verbose:
        print(f"Test loss without training: {testLoss}")
    
    df = pd.DataFrame({
        "epochs": [0],
        "seq_length": config["seq_length"],
        "n_cells": config["n_cells"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "train_loss": trainLoss,
        "test_loss": testLoss
    })

    for epoch in range(1, n_epochs + 1):
        loss_train = 0
        model.train()  # Set model to training mode

        for imgs, sin_labels, cos_labels, time in train_data_loader:  # Expect 4 values
            imgs = imgs.to(device)
            sin_labels = sin_labels.to(device).squeeze()  # Ensure sin_labels shape is [64] instead of [64, 1]
            cos_labels = cos_labels.to(device).squeeze()  # Ensure cos_labels shape is [64] instead of [64, 1]

            # Forward pass
            outputs = model(imgs)  # Model outputs both sin and cos
            sin_outputs = outputs[:, 0]  # First output is sin
            cos_outputs = outputs[:, 1]  # Second output is cos

            # Compute losses for both sin and cos predictions
            loss_sin = loss_fn(sin_outputs, sin_labels)
            loss_cos = loss_fn(cos_outputs, cos_labels)

            # Total loss as the sum of both
            loss = loss_sin + loss_cos

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            loss_train += loss.item()  # Accumulate loss
        
        # Calculate loss on test set
        testLoss = lossOnTestDataset(model, test_data_loader, device, loss_fn)

        if verbose:
            print(f"{datetime.datetime.now()} Epoch: {epoch}/{n_epochs}, "
                  f"Training loss: {loss_train/len(train_data_loader)}, "
                  f"Testing loss: {testLoss}")

        # Record training statistics
        df1 = pd.DataFrame({
            "epochs": [epoch],
            "seq_length": config["seq_length"],
            "n_cells": config["n_cells"],
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layers"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "train_loss": loss_train / len(train_data_loader),
            "test_loss": testLoss
        })

        df = pd.concat([df, df1])

        # Save the best model state based on test loss
        if testLoss < best_loss:
            best_loss = testLoss
            best_model_state = model.state_dict()

    return df, best_model_state

def Makeprediction(model, test_data_loader, device):
    model.eval()
    labellists_sin = []
    labellists_cos = []
    outputlists_sin = []
    outputlists_cos = []
    
    with torch.no_grad():
        for imgs, sin_labels, cos_labels, time in test_data_loader:  # mini-batches with data loader
            imgs = imgs.to(device=device)
            
            # Move labels to CPU
            sin_labels = sin_labels.to('cpu').numpy()
            cos_labels = cos_labels.to('cpu').numpy()
            
            # Predict outputs
            outputs = model(imgs)
            sin_outputs = outputs[:, 0].to('cpu').detach().numpy()  # First output: sin
            cos_outputs = outputs[:, 1].to('cpu').detach().numpy()  # Second output: cos

            # Append predictions and labels
            outputlists_sin.append(sin_outputs)
            outputlists_cos.append(cos_outputs)
            labellists_sin.append(sin_labels)
            labellists_cos.append(cos_labels)
    
    return labellists_sin, labellists_cos, outputlists_sin, outputlists_cos

def get_labels_and_outputs_mse(model, test_data_loader, device):
    ''' 
        This is a func to run the models on the slected cell types and
        returns:
        
         1. mean square error of the model
         2. labesl and the outputs (concatenated for the selected cells)
         
    '''
    
    #labellists, outputlists = Makeprediction(model,test_data_loader,device=device)
    labellists_sin, labellists_cos, outputlists_sin, outputlists_cos= Makeprediction(model, test_data_loader, device)
    labels_sin= np.concatenate(labellists_sin)
    labels_cos= np.concatenate(labellists_cos)
    outputs_sin= np.concatenate(outputlists_sin)
    outputs_cos= np.concatenate(outputlists_cos)
        
    outputs= np.column_stack((outputs_sin, outputs_cos))
    labels= np.column_stack((labels_sin, labels_cos))
    
    mse_sin= mean_squared_error(labels_sin,outputs_sin) 
    mse_cos= mean_squared_error(labels_cos,outputs_cos)
    
    #mse= np.column_stack((mse_sin, mse_cos))
    mse = (mse_sin, mse_cos)
    
    return mse, outputs, labels



def plot_shuffled_rvalues_oneSession(gs,df_sSes, Angle_test, Angle_test_pred ,iteration= 500 ):

    
    ax= fig.add_subplot(gs[0])
    ax.hist(df_sSes.r_test)
    threshold= np.percentile(df_sSes.r_test,95)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label='thr.')
    r_original,p= pearsonr(Angle_test, Angle_test_pred)
    ax.axvline(x=r_original, color='red', linestyle='-', linewidth=2, label='ori. r')
    ax.set_title(f'Test dataset \n itteration:{iteration}, n_fs:{df_sSes.df.iloc[0].n_cells.iloc[0]}', fontsize=9)
    ax.set_xlabel('Pearson r ')##
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.9, 1.0), fancybox=True, shadow=True)


    ax= fig.add_subplot(gs[1])
    ax.hist(abs(df_sSes.circr_test))
    threshold= np.percentile(abs(df_sSes.circr_test),95)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label='thr.')
    circr= abs(circcorrcoef(Angle_test, Angle_test_pred))
    ax.axvline(x=circr, color='red', linestyle='-', linewidth=2, label='ori. circr')
    ax.set_title(f'Test dataset \n itteration:{iteration}, n_fs:{df_sSes.df.iloc[0].n_cells.iloc[0]}', fontsize=9)
    ax.set_xlabel('Circular r ')##
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.9, 1.0), fancybox=True, shadow=True)

def get_session_ifr_pose_angle_aroundlever_for_model(sessionName,ses,sSes,cells,interName= 'all_light', ctype= 'fs',  sigma_ifr= 5, maxDistance= 18, rotationType="none" ):
    """
    Get ifr  and the movement direction of the animal that matches the ap.pose samples

    """
    # load session files
    #sSes, ap, apSim, cg = load_session_files_for_modelling(sessionName,cells,sSessions,intervals, ctype= ctype)
    ap, cg = load_pose_around_lever_for_modeling(ses,sSes, cells,interName= interName,maxDistance= maxDistance, rotationType=rotationType, 
                                        invalidateSmallBridgeAngle= False,invalidateMaxAngle=np.pi/12, ctype= ctype)

    # trick to get aligned ifr and pose data
    time = ap.pose[:,0]
    bin_size_sec = np.diff(time)[0]

    pose = ap.pose[:,1:3]
    keepIndices = ~(np.isnan(pose[:,1]))
    pose= pose[keepIndices]

    angles = np.arctan2(pose[:,1], pose[:,0]) ## get values between -3,3 
    magnitude = np.sqrt(pose[:,0]**2 + pose[:,1]**2)
    #angles = np.mod(angles + 2 * np.pi, 2 * np.pi) ## get values between 0, 2*pi
    # Compute the sine and cosine of the angle
    sin_angle = pose[:,1] / magnitude
    cos_angle = pose[:,0] / magnitude

    sin_cos_angles = np.column_stack((sin_angle, cos_angle))

    if len(cg.sc_list)!= 0:

        for n in cg.sc_list:
            n.spike_train.set_intervals(ses.intervalDict[interName])
            n.spike_train.instantaneous_firing_rate(bin_size_sec = bin_size_sec, sigma= sigma_ifr, time_start=min(time)-bin_size_sec/2, 
                                                time_end=max(time), outside_interval_solution="remove")


        ifr = np.stack([n.spike_train.ifr[0][keepIndices] for n in cg.sc_list])

        #### aligne the shape 
        time = time[keepIndices]
        ifr= ifr[:,:-1]
        time= time[:-1]

    else:
        ifr= np.nan
        time= time[:-1]

    return ifr, pose[:-1,:], time, angles[:-1], sin_cos_angles[:-1,:]

def load_pose_around_lever_for_modeling(ses,sSes, cells,interName= 'all_light' ,maxDistance= 18, rotationType="none", 
                                        invalidateSmallBridgeAngle= False,invalidateMaxAngle=np.pi/12, ctype= 'fs' ):

    sSes.load_parameters_from_files()
    sSes.load_parameters_from_files()
    sSes.ap.load_pose_from_file() # get original hd data,

    ## Transfer the x,y position of the animal to the Lever Reference frame 
    toLeverReferenceFrame(ses=ses,sSes=sSes,maxDistance=maxDistance, rotationType="none", invalidateSmallBridgeAngle= False,invalidateMaxAngle=np.pi/12)

    sSes.ap.set_intervals(ses.intervalDict[interName])
    stl = Spike_train_loader()
    stl.load_spike_train_kilosort(sSes)
    cg = Cell_group(stl, sSes.ap)

    # create a list of cells (spikeA.Neuron)
    if ctype== 'fs' :
        cids = cells[(cells.session == sSes.name) & (cells.mrate_RF1> 10) & (cells.interneuron)].cluId

    elif ctype== 'gc':
        cids = cells[(cells.session == sSes.name) & (cells.gridCell_FIRST)].cluId
    else:
        cids = cells[(cells.session == sSes.name)].cluId

    cIds = [cid.split("_")[1] for cid in cids]
    cg.sc_list = [n for n in cg.neuron_list if n.name in cIds]

    return sSes.ap, cg


def toLeverReferenceFrame(ses,sSes,maxDistance=30, rotationType="none",
                         invalidateSmallBridgeAngle=False,invalidateMaxAngle=np.pi/12):
    """
    Change the reference frame of the position data so that the lever is at 0,0.
        
    The data in sSes.ap.pose will be modified 
    Columns 1 and 2 are x and y are relative to 0,0
    Column 4 is the direction of the position vector (column 1,2) relative to 1,0
    
    Arguments:
    
    ses: autopipy session
    sSes: spikeA session
    interName: name of intervals to use (from ses.intervalDict)
    maxDistance: max distance from the center of the lever box
    rotationType: can be "none","bridge","lever", once centered on the lever,
                    we can rotate the position to have different reference frame (cartesian (none), relative to bridge direction, relative to lever orientation)
    invalidateSmallBridgeAngle: whether to invalidate lever position when the bridge angle is small (for which cartesian and brdige reference frames are the same)
                                This is done to eliminate data when the none and brdige rotations are the same
                                This can be used to better contrast the prediction of "bridge" and "none" rotation
    invalidateMaxAngle: angle below which we invalidate   
    
    See this jupyter notebook to explain how the rotation were done: directional_reference_frame_rotation_example.ipynb
    
    """
    
    
    leverX,leverY,leverOri = getLeverPosition(ses)
    
    ## angle between lever and bridge
    fn = ses.path+"/bridgeCoordinatesCm"
    if os.path.exists(fn):
        b = np.loadtxt(fn)
        xy = b.mean(axis=0)
        bridgeX=xy[0]
        bridgeY=xy[1]
    else:
        bridgeX=0
        bridgeY=-42
    
    leverToBridgeVX = bridgeX - leverX
    leverToBridgeVY = bridgeY - leverY
    
    leverToBridgeAngle = np.arctan2(leverToBridgeVY,leverToBridgeVX) # this angle is relative to a vector pointing east
    
    if invalidateSmallBridgeAngle: # invalidate lever position when the bridge angle is very close to -np.pi (South)
        # This is done to eliminate data when the none and brdige rotations are the same
        # This can be used to better contrast the prediction of "bridge" and "none" rotation
        # get the angle between (0,-1) vector and the leverToBridge vector.
        v = np.vstack([leverToBridgeVX,leverToBridgeVY]).T # 2D numpy array, one vector per row
        ang = vectorAngle(v=v,rv=np.array([[0,-1.0]])) # angle relative to 0,-1 vector (south)
        invalidateIndices = ang < invalidateMaxAngle
        leverX[invalidateIndices]=np.nan
        leverY[invalidateIndices]=np.nan
        
    
    # transform the animal position so that it is centered on lever
    mouseX = sSes.ap.pose[:,1]-leverX
    mouseY = sSes.ap.pose[:,2]-leverY
    
    if rotationType == "lever":    
        # original vector for each pixel in the map
        rotation=leverOri + np.pi/2 # The angles were from a vector poinint east. Adding np.pi/2 change the reference vector to a vector poining south.
    elif rotationType == "bridge":
        rotation=leverToBridgeAngle + np.pi/2 # #we need to rotate by the negative of the lever to bridge angle
    else : # don't rotate
        rotation = np.zeros_like(mouseX)
        
    oriVectors = np.vstack([mouseX,mouseY]).T # x by 2 matrix, one vector per row
       
    # this is a rotation matrix for our vectors, one per data points in path
    rotMat = np.array([[np.cos(rotation), -np.sin(rotation)],
                       [np.sin(rotation), np.cos(rotation)]])
    
    # rotate the vectors
    rotVectors = np.empty_like(oriVectors)
        
    for i in range(rotVectors.shape[0]): # for points in path
        rotVectors[i]= oriVectors[i,:]@rotMat[:,:,i] # apply the rotation

    # this should be the rotVectors!!!!
    mouseX = rotVectors[:,0]
    mouseY = rotVectors[:,1]
    
    D = np.sqrt(mouseX**2+mouseY**2)
    
    mouseX[D>maxDistance]= np.nan
    mouseY[D>maxDistance]= np.nan
    
    
    # replace x and y by mouseX and mouseY in ap.pose_ori #
    sSes.ap.pose_ori[:,1] = mouseX
    sSes.ap.pose_ori[:,2] = mouseY

    
    # replace the head direction data with the angle between the vector of the animal position (origin 0,0) and the vector 1,0
    v = np.vstack([mouseX,mouseY]).T # 2D numpy array, one vector per row
    sSes.ap.pose_ori[:,4] = np.arctan2(mouseY,mouseX)
    
def getLeverPosition(ses):
    """
    return the lever position at each frame of the ap.pose
    
    nan are interpolated and the data is smooth; since the lever is stable during trials, this should only improve the quality of the data
    """
    # calculate lever position
    fn = ses.path+"/leverPose"
    leverPose = pd.read_csv(fn)
    
    # middle point at the back of the lever
    midBackX = (leverPose.leverBoxPLX + leverPose.leverBoxPRX)/2
    midBackY = (leverPose.leverBoxPLY + leverPose.leverBoxPRY)/2
    
    ## lever position is mid point between midBAck and leverPress
    leverX = (leverPose.leverPressX + midBackX)/2
    leverY = (leverPose.leverPressY + midBackY)/2 
    
    # Fill in NaN's in lever position (lever does not move during trials anyway)
    mask = np.isnan(leverX)
    leverX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), leverX[~mask])
    mask = np.isnan(leverY)
    leverY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), leverY[~mask])
    # Smooth lever position as it is not moving anyway
    leverX = gaussian_filter1d(leverX, 30)
    leverY = gaussian_filter1d(leverY, 30)

    ## lever orientation is center of lever box to the press
    ovX = leverPose.leverPressX-leverX
    ovY = leverPose.leverPressY-leverY
    leverOri = np.arctan2(ovY,ovX)
    
    return leverX,leverY,leverOri  

interNames = ['atLever_light']


def run_and_evaluate_model_shuffle(sSes, ses, cells,interName, ctype='fs', sigma_ifr=5, shuffle=True):
    
    ## Get the test and train Dataset for each shuffle
    train_dataset, test_dataset, train_loader, test_loader, _, config = get_test_training_datas_oneSession(
        sSes, ses, cells,interName, ctype=ctype, sigma_ifr=sigma_ifr, shuffle=shuffle
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(config["n_cells"], config["hidden_size"], config["num_layers"], config["num_outputs"], config["seq_length"], device=device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    # Train the model
    df, _ = training_loop(
        n_epochs=config["num_epochs"], optimizer=optimizer, model=model,
        loss_fn=loss_fn, train_data_loader=train_loader, test_data_loader=test_loader,
        config=config, device=device, verbose=False
    )

    # Get labels and predictions
    mse_test, outputs_test, labels_test = get_labels_and_outputs_mse(model, test_loader, device)
    mse_train, outputs_train, labels_train = get_labels_and_outputs_mse(model, train_loader, device)

    # Calculate angles
    angles = {}
    
    for split, labels, outputs in zip(['train', 'test'], [labels_train, labels_test], [outputs_train, outputs_test]):
        angles[f'{split}_shuffled'] = np.arctan2(labels[:, 0], labels[:, 1])
        angles[f'{split}_pred_shuffled'] = np.arctan2(outputs[:, 0], outputs[:, 1])

    # Calculate correlations
    metrics = {
        'r_train': pearsonr(angles['train_shuffled'], angles['train_pred_shuffled'])[0],
        'circr_train': circcorrcoef(angles['train_shuffled'], angles['train_pred_shuffled']),
        'r_test': pearsonr(angles['test_shuffled'], angles['test_pred_shuffled'])[0],
        'circr_test': circcorrcoef(angles['test_shuffled'], angles['test_pred_shuffled']),
        'df': df,
    }


    return {**angles, **metrics}

def Get_shuffle_values_one_session(sSes, ses, cells,interName, iteration= 500): 
    results = []
    for _ in tqdm(range(iteration)):
        angles_and_metrics = run_and_evaluate_model_shuffle(sSes, ses, cells,interName, ctype='fs', sigma_ifr=5, shuffle=True)
        results.append(angles_and_metrics)

    # Convert results to a DataFrame
    df_sSes = pd.DataFrame(results)
      
    return df_sSes
    
def process_session_intervals_withShuffles(sSes, ses, interNames, cells, iteration=500):
    # Check if there are more than 1 FS cells in the session
    FS_count = cells[(cells.interneuron) & (cells.mrate_RF1 > 10) & (cells.session == sSes.name)].shape[0]
    
    if FS_count >= 7 :
        for interName in interNames:
            # Define the file path
            filename = f"{sSes.name}_{interName}_shuffled_values.pkl"
            file_path = os.path.join(save_directory, filename)
            
            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"File {filename} already exists. Skipping.")
                continue  # Skip to the next interval if file exists
            
            # Process each interval and save the DataFrame
            print(sSes.name)
            df_sSes = Get_shuffle_values_one_session(sSes, ses,cells, interName, iteration=iteration)
            df_sSes.to_pickle(file_path)
            print(f"Saved data for {filename} at {file_path}")
            del df_sSes
            gc.collect()
            

def process_session_wrapper(session_pair, interNames, cells, iteration):
    sSes, ses = session_pair  # Unpack the session pair
    #print(sSes)
    process_session_intervals_withShuffles(sSes, ses, interNames, cells, iteration=iteration)
    pass
