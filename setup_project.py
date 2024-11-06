"""
This files sets my working environment to work with this project.

It imports some packages that I often use, creates list of recording session (autopipy and spikeA).

I also added some useful functions that I use when processing data

Simply put %run setup_project.py in a jupyter notebook to run the code
"""

from numba import jit
import pandas as pd
import numpy as np
import os.path
import shutil
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from itertools import compress
import socket
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from autopipy.project import Project # a for autopipy
from autopipy.session import Session # a for autopipy
from autopipy.trialElectro import TrialElectro # a for autopipy

from spikeA.Session import Kilosort_session
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train_loader import Spike_train_loader
from spikeA.Cell_group import Cell_group
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Spike_train import Spike_train
from spikeA.Spike_waveform import Spike_waveform
from spikeA.Spike_waveform import Spike_waveform

def RosToResTime(rosTimes, resRosTime):
    """
    Function to go from ROS time to res time
    
    Argument
    rosTimes: 1D array with the ROS times for which you want the res time
    resRosTime: 2D array (:,2) that is used to build the interpolation model (first column is res time, second column is ROS time)
    
    Return
    1D array the same size of rosTime input. It contains the res times for the ros times given.
    
    Example to transfrom some intervals (2D arrays):
    RosToResTime(journey_intervals.flatten(),np.stack([sSes.ap.pose[:,0],sSes.ap.pose[:,7]],axis=1)).reshape((-1,2))
    
    """
    fx = interp1d(resRosTime[:,1],resRosTime[:,0], bounds_error=False) # x we will start at 0 until the end of the file
    return fx(rosTimes)

def resToRosTime(resTimes, rosResTime):
    """
    Function to go from res time to ROS time
    
    Argument
    resTimes: 1D array with the res times for which you want the ROS time
    rosResTime: 2D array (:,2) that is used to build the interpolation model (first column is the ros time, second column is the res time)
    
    Return
    1D array the same size of resTime input. It contains the ros times for the res times given.
    
    Example to transform some intervals (2D arrays):
    resToRosTime(res.flatten(),np.stack([sSes.ap.pose[:,7],sSes.ap.pose[:,0]],axis=1)).reshape((-1,2))
    """
    fx = interp1d(rosResTime[:,1],rosResTime[:,0], bounds_error=False) # x we will start at 0 until the end of the file
    return fx(resTimes)

#def setup_project_session_lists(projectName="autopi_inter",dataPath="/data/projects/autopi_inter",dlcModelPath="/adata/models",sessionFileName="sessionList"):
def setup_project_session_lists(projectName="autopi_inter",dataPath="/data/projects/autopi_inter",dlcModelPath="/adata/models",sessionFileName="sessionList"):

    """
    Set variables, projectName, dataPath, dlcModelPath
    Create an autopipy.Project object called myProject
    Create a list of spikeA.Kilosort_sessions called sSessions
    """
    # a mixed class
    #from neuronAutopi import NeuronAutopi
    print("creating myProject, an autopipy.project object")
    myProject = Project(name=projectName,dataPath=dataPath,dlcModelPath=dlcModelPath)
    fn=myProject.dataPath+"/"+ sessionFileName
    print("Reading " + fn)
    sessionNames = pd.read_csv(fn) # this will be a pandas dataframe
    print("We have {} testing sessions in the list".format(len(sessionNames)))
    myProject.createSessionList(sessionNameList=sessionNames.sessionName.to_list()) 
    print("spikeA.Kilosort_session objects are in sSessions")
    sSessions = [ Kilosort_session(ses.name,ses.path) for ses in myProject.sessionList ] # spikeA session object
    return projectName, dataPath, dlcModelPath, myProject, sSessions



def load_parameters_from_files_project(sSesList):
    for ses in tqdm(sSesList):
        ses.load_parameters_from_files()

def load_spike_train_project(sSesList):    
    for ses in tqdm(sSesList):
        ses.load_parameters_from_files()
        stl = Spike_train_loader()
        ses.stl= stl
        stl.load_spike_train_kilosort(ses)
        ## give a name to your neurons
        neuronNames = [ "{}_{}".format(ses.name,clu) for clu in stl.clu_ids]
        ses.cg = Cell_group(stl,names=neuronNames)      
        
        
def prepareAutopipySession(ses):
    fn = ses.path+"/"+"trial_table_simple"
    ses.trials = pd.read_csv(fn)
        
def prepareSessionsForSpatialAnalysisProject(spikeASesList,autopipySesList,pose_file_extension = ".pose.npy"):
    print("Loading Animal_pose and Spike_train, sSes.ap and sSes.cg")
    print("Will load", pose_file_extension)
    
    sSes = spikeASesList[0]
    if not isinstance(sSes, Kilosort_session):
        raise typeError("spikeASesList should contain spikeA.Session.Kilosort_session objects")
    
    for ses in tqdm(spikeASesList):
        ses.load_parameters_from_files() 
        ap = Animal_pose(ses)
        ap.pose_file_extension = pose_file_extension # This means that the ap will always load from this extension
        ap.load_pose_from_file()
        stl = Spike_train_loader()
        stl.load_spike_train_kilosort(ses)
        cg = Cell_group(stl,ap)
        # we want to keep ap and cg after the function ends, one way is to store it in our sSes
        ses.ap = ap
        ses.cg = cg
    print("Loading ses.trial_table_simple as ses.trials")
    for ses in autopipySesList:
        prepareAutopipySession(ses)
    
    print("Create condition intervals in ses.intervalDict")
    for ses,sSes in zip(autopipySesList,spikeASesList):
        getSessionConditionIntervals(ses,sSes) # function defined in setup.py
        if ses.trialList is not None:
            getSearchHomingIntervals(ses,sSes)

def data_location(sSesList):
    """
    Arguments:
    sSesList: list of spikeA sessions
    
    returns a dataframe with session, mouse, hd, ip columns for all sessions of the sSesList
    """
    res = [ session_data_location(sSes) for sSes in sSesList]
    df = pd.DataFrame({"session": [ s for s,m,h,ip in res],
                 "mouse": [ m for s,m,h,ip in res],
                  "hd": [ h for s,m,h,ip in res],
                  "ip": [ ip for s,m,h,ip in res]})    
    return df
def session_data_location(sSes):
    """
    Find on which hard drive and computer (ip) the data are located
    """
    mouse_dir,session = os.path.split(sSes.path)
    first_level = os.readlink(mouse_dir)
    first_level =os.path.normpath(first_level)
    second_level = os.readlink(first_level)
    hd = second_level.split("/")[2]
    autom = pd.read_csv("/etc/auto.ext_drives",sep='\t', lineterminator='\n',header=None)
    autom.columns = ["hd","param","ip"]
    ip = autom.loc[autom.hd==hd].ip.item().split(":")[0].split(".")[0]
    return sSes.name,sSes.subject,hd,ip

def get_local_indices(dat_loc,ip=None):
    """
    returns a numpy array of boolean, whether sessions are on a specific computer
    """
    return (dat_loc.ip==ip).to_numpy()

def get_local_sessions(sSesList,ip=None):
    """
    Get a list of sessions that are local to the computer running this code
    
    If ip is left to None, it will get the local ip for you
    """
    if ip is None:
        ip=socket.gethostname()
    dat_loc =data_location(sSesList)
    i = get_local_indices(dat_loc,ip)
    return list(compress(sSesList, i))

def getSessionConditionIntervals(ses,sSes):
    """
    Get the time intervals (ephys time) for the following
    - first open field trial
    - first half open field trial
    - second half open field trial
    - light autopy trials
    - dark autopy trials
    
    The intervals are saved in the autopipy Session object as ses.intervalDict
    
    
    Arguments:
    ses: autopipy Session
    sSes: spikeA Session
    """
    #print(ses.name)
    circ80_indices = [i for i, j in enumerate(sSes.desen) if j == 'circ80']
    rest_indices= [i for i, j in enumerate(sSes.desen) if j == 'rest']
    
    
    circ80_inter = sSes.trial_intervals.inter[circ80_indices[:],:] # the two trials
    
    last_rest_inter= sSes.trial_intervals.inter[rest_indices[0:-1],:]
    
    ## first and second circ80 trial
    circ80_inter1 = circ80_inter[0:1,:]
    circ80_inter2 = circ80_inter[1:2,:]
    
    light_inter = ses.trials[ses.trials.light=="light"].iloc[:,7:9].to_numpy()
    dark_inter = ses.trials[ses.trials.light=="dark"].iloc[:,7:9].to_numpy()
    
    
    # create a task, task_1 and task_2 intervals
    task_inter = np.concatenate([light_inter,dark_inter])
    t = task_inter.copy()
    np.random.shuffle(t)
    mid = int(t.shape[0]/2)
    task_1 = t[:mid]
    task_2 = t[mid:]  
    
    
    
    ## get light_1 and light_2
    l = light_inter.copy()
    np.random.shuffle(l)
    mid = int(l.shape[0]/2)
    light_1 = l[:mid]
    light_2 = l[mid:]  
    
    # dark_1 and dark_2
    d = dark_inter.copy()
    np.random.shuffle(d)
    mid = int(d.shape[0]/2)
    dark_1 = d[:mid]
    dark_2 = d[mid:]  
    
    
    
    ses.intervalDict={"circ80":circ80_inter, # we can save our intervals for later use in the autopipy.ses
                      "circ80_1":circ80_inter1,
                      "circ80_2":circ80_inter2,
                      "task":task_inter,
                      "task_1":task_1,
                      "task_2":task_2,
                      "light": light_inter,
                      "light_1":light_1,
                      "light_2":light_2,
                      "dark": dark_inter,
                      "dark_1": dark_1,
                      "dark_2": dark_2}
    #print("{}, number of intervals:{} {} {}".format(ses.name,
    #                                                len(ses.intervalDict["circ80"]),
    #                                                len(ses.intervalDict["light"]),
    #                                                len(ses.intervalDict["dark"]) ))


def getSearchHomingIntervals(ses,sSes):
    """
    Get the time intervals (ephys time)
    This is saved in ses.intervalDict
    It assumes that ses.intervalDict exists (that you runned getSessionConditionIntervals() before)
    
    To keep track of the trialNo associated with each interval, I created a key called "trialNo". 
    You can get a numpy array containing the trialNo for a set of intervals using ses.intervalDict["trialNo"]["searchPath_light"]
    
    Arguments:
    ses: autopipy Session
    sSes: spikeA Session
    
    For the following
    - searchPath_light
    - searchPath_light_1
    - searchPath_light_2
    - searchPath_dark
    - searchPath_dark_1
    - searchPath_dark_2
    - homingPath_light
    - homingPath_light_1
    - homingPath_light_2
    - homingPath_dark
    - homingPath_dark_1
    - homingPath_dark_2 
    - all_light
    - all_light_1
    - all_light_2
    - all_dark_1
    - all_dark_2
    - all_light_noPress
    - all_dark_noPress
    """
    # get navPath data
    fn = ses.path+"/navPathSummary.csv"
    navs = pd.read_csv(fn)
    
    pathTypes = ["searchPath","searchPath","searchToLeverPath","searchToLeverPath","homingPath","homingPath",
                 "homingFromLeavingLever","homingFromLeavingLever", "homingFromLeavingLeverToPeriphery","homingFromLeavingLeverToPeriphery","atLever","atLever","all","all"]
    light = ["light","dark","light","dark","light","dark","light","dark","light","dark","light","dark","light","dark"]
    
    # we want to store the trial number associated with every interval in ses.intervalDict
    ses.intervalDict["trialNo"] = {}
    
    
    for pt,l in zip(pathTypes,light):
        s = navs.startTimeRes[(navs.type==pt) & (navs.light==l) & (navs.nLeverPresses >= 1)]
        e = navs.endTimeRes[(navs.type==pt) & (navs.light==l) & (navs.nLeverPresses >= 1)]
        t = navs.trialNo[(navs.type==pt) & (navs.light==l) & (navs.nLeverPresses >= 1)]
        
        name = "{}_{}".format(pt,l)
        ses.intervalDict[name] = np.vstack([s.to_numpy(),e.to_numpy()]).T
        ses.intervalDict["trialNo"][name] = t.to_numpy()
       
        # half conditions
        m = ses.intervalDict[name].copy()
        np.random.shuffle(m)
        mid = int(m.shape[0]/2)
        ses.intervalDict[name+"_"+"1"] = m[:mid]
        ses.intervalDict[name+"_"+"2"] = m[mid:]  

    # noPress 
    pt = "all"
    for l in ["light", "dark"]:
        s = navs.startTimeRes[(navs.type==pt) & (navs.light==l) & (navs.nLeverPresses < 1)]
        e = navs.endTimeRes[(navs.type==pt) & (navs.light==l) & (navs.nLeverPresses < 1)]
        t = navs.trialNo[(navs.type==pt) & (navs.light==l) & (navs.nLeverPresses < 1)]
        name = "{}_{}_noPress".format(pt,l)
        ses.intervalDict[name] = np.vstack([s.to_numpy(),e.to_numpy()]).T
        ses.intervalDict["trialNo"][name] = t.to_numpy()
    
    
    # create intervals in which we get time on the bridge
    # Interval names: 
    # searchToLeverPathWithBridge_light and searchToLeverPathWithBridge_dark
    # homingFromLeavingLeverWithBridge_light and homingFromLeavingLeverWithBridge_dark
    # we add time before searchToLeverPath and add time after homingFromLeavingLever
    
    # add time before the official beginning to get the bridge time
    addTime = 2 # time in seconds
    pt = "searchToLeverPath"
    for l in ["light","dark"]:
        name = "{}_{}".format(pt,l)
        newName = "{}{}_{}".format(pt,"WithBridge",l)
        intervals = ses.intervalDict[name].copy()
        ses.intervalDict["trialNo"][newName] = ses.intervalDict["trialNo"][name] # the trialNo stays the same
        ses.intervalDict[newName] = np.vstack([intervals[:,0]-addTime,intervals[:,1]]).T
    
    # add time after the official end to get the bridge time
    pt = "homingFromLeavingLever"
    addTime = 2 # time in seconds
    for l in ["light","dark"]:
        name = "{}_{}".format(pt,l)
        newName = "{}{}_{}".format(pt,"WithBridge",l)
        intervals = ses.intervalDict[name].copy()
        ses.intervalDict["trialNo"][newName] = ses.intervalDict["trialNo"][name] # the trialNo stays the same
        ses.intervalDict[newName] = np.vstack([intervals[:,0],intervals[:,1]+addTime]).T
    
    
    
    
    # add allTask, allTask_1 and allTask_2, which combine the all navPaths of both light conditions
    all_task = np.sort(np.vstack([ses.intervalDict["all_light"],ses.intervalDict["all_dark"]]))
    ses.intervalDict["all_task"] = all_task
    m = all_task.copy()
    np.random.shuffle(m)
    mid = int(m.shape[0]/2)
    ses.intervalDict["all_task"+"_"+"1"] = m[:mid]
    ses.intervalDict["all_task"+"_"+"2"] = m[mid:]  
    
def getShortSearchDf(df,ses):
    """
    df: dataframe returned by ses.getTrialVariablesDataFrame()
    ses: autopipy session object
    """
    # Only keep the trials that are in ses.trials
    df = df [df.trialNo.isin(ses.trials.trialNo)]

    # get the median error in darkness
    
    medianLength = np.nanmedian(df.searchLength[df.light=="dark"])

    df["shortSearch"] = df.searchLength < medianLength
    df["shortSearch"][np.isnan(df.searchLength)] = np.nan
    return df
    
    
def getShortSearchIntervals(ses):
    """
    Get intervals for dark trials with short search and long search path
    Short search trials are trials with search length smaller than the median search length of the dark trials
    Using the median ensures that there is a similar number of dark short and long search trials
    """

    df = ses.getTrialVariablesDataFrame()

    df = getShortSearchDf(df,ses)
    
    # we need only the navPath associated with the error, not the whole trial
    fn = ses.path+"/navPathSummary.csv"
    navs = pd.read_csv(fn)
    # only keep trials that are in ses.trials
    navs = navs[navs.trialNo.isin(ses.trials.trialNo)]
    # get the last navPath of each trial
    pathNames = []
    for t in df.trialNo : 
        res = navs[(navs.trialNo==t)&(navs.type=="all")]
        if(len(res)>0):
            pathNames.append(res.iloc[-1,0])
        else:
            pathNames.append(np.nan)
            
    # add the name of the last navPath in the data frame with accuracy
    df["navPathName"]=pathNames


    short_indices = (df.light=="dark") & (df.shortSearch==True)
    long_indices = (df.light=="dark") & (df.shortSearch==False)

    ses.intervalDict["dark_shortSearch"] = np.vstack([navs.startTimeRes[navs.name.isin(df.navPathName[short_indices])].to_numpy(),navs.endTimeRes[navs.name.isin(df.navPathName[short_indices])].to_numpy()]).T
    ses.intervalDict["dark_longSearch"] = np.vstack([navs.startTimeRes[navs.name.isin(df.navPathName[long_indices])].to_numpy(),navs.endTimeRes[navs.name.isin(df.navPathName[long_indices])].to_numpy()]).T   
        
def getAccurateHomingDf(df,ses):
    """
    df: dataframe retruned by ses.getTrialVariablesDataFrame()
    ses: autopipy sesssion object
    """

    # Only keep the trials that are in ses.trials
    df = df [df.trialNo.isin(ses.trials.trialNo)]

    # get the median error in darkness
    df["homingErrorAtPeripheryAbs"] = np.abs(df.homingErrorAtPeriphery)
    medianError = np.nanmedian(df.homingErrorAtPeripheryAbs[df.light=="dark"])

    df["accurateHoming"] = df.homingErrorAtPeripheryAbs < medianError
    df["accurateHoming"][np.isnan(df.homingErrorAtPeripheryAbs)] = np.nan
    return df
    

def getAccurateHomingIntervals(ses):
    """
    Get intervals for dark trials with accurate and inaccurate homing
    Accurate are trials with homing error smaller than the median homing error of the dark trials
    Using the median ensures that there is a similar number of dark accurate and dark innacurate trials
    """

    df = ses.getTrialVariablesDataFrame()

    df = getAccurateHomingDf(df,ses)
    
    # we need only the navPath associated with the error, not the whole trial
    fn = ses.path+"/navPathSummary.csv"
    navs = pd.read_csv(fn)
    # only keep trials that are in ses.trials
    navs = navs[navs.trialNo.isin(ses.trials.trialNo)]
    # get the last navPath of each trial
    pathNames = []
    for t in df.trialNo : 
        res = navs[(navs.trialNo==t)&(navs.type=="all")]
        if(len(res)>0):
            pathNames.append(res.iloc[-1,0])
        else:
            pathNames.append(np.nan)
            
    # add the name of the last navPath in the data frame with accuracy
    df["navPathName"]=pathNames


    accurate_indices = (df.light=="dark") & (df.accurateHoming==True)
    inaccurate_indices = (df.light=="dark") & (df.accurateHoming==False)

    ses.intervalDict["dark_accurateHoming"] = np.vstack([navs.startTimeRes[navs.name.isin(df.navPathName[accurate_indices])].to_numpy(),navs.endTimeRes[navs.name.isin(df.navPathName[accurate_indices])].to_numpy()]).T
    ses.intervalDict["dark_inaccurateHoming"] = np.vstack([navs.startTimeRes[navs.name.isin(df.navPathName[inaccurate_indices])].to_numpy(),navs.endTimeRes[navs.name.isin(df.navPathName[inaccurate_indices])].to_numpy()]).T
     
       
def RosToResTime(rosTimes, resRosTime):
    """
    Function to go from ROS time to res time
    
    Argument
    rosTimes: 1D array with the ROS times for which you want the res time
    resRosTime: 2D array (:,2) that is used to build the interpolation model (first column is res time, second column is ROS time)
    
    Return
    1D array the same size of rosTime input. It contains the res times for the ros times given.
    
    Example to transfrom some intervals (2D arrays):
    RosToResTime(journey_intervals.flatten(),np.stack([sSes.ap.pose[:,0],sSes.ap.pose[:,7]],axis=1)).reshape((-1,2))
    
    """
    fx = interp1d(resRosTime[:,1],resRosTime[:,0], bounds_error=False) # x we will start at 0 until the end of the file
    return fx(rosTimes)

def resToRosTime(resTimes, rosResTime):
    """
    Function to go from res time to ROS time
    
    Argument
    resTimes: 1D array with the res times for which you want the ROS time
    rosResTime: 2D array (:,2) that is used to build the interpolation model (first column is the ros time, second column is the res time)
    
    Return
    1D array the same size of resTime input. It contains the ros times for the res times given.
    
    Example to transform some intervals (2D arrays):
    resToRosTime(res.flatten(),np.stack([sSes.ap.pose[:,7],sSes.ap.pose[:,0]],axis=1)).reshape((-1,2))
    """
    fx = interp1d(rosResTime[:,1],rosResTime[:,0], bounds_error=False) # x we will start at 0 until the end of the file
    return fx(resTimes)

def isin_tolerance( A, B, tol):
        """
        Are elements of A in B, with some tolerance
        
        We need this to compare float64 that can differ by tiny amount

        """
        A = np.asarray(A)
        B = np.asarray(B)

        if B.size<2:
            print("B should have a size of at least 2")

        Bs = np.sort(B) # skip if already sorted
        idx = np.searchsorted(Bs, A)

        linvalid_mask = idx==len(B)
        idx[linvalid_mask] = len(B)-1
        lval = Bs[idx] - A
        lval[linvalid_mask] *=-1

        rinvalid_mask = idx==0
        idx1 = idx-1
        idx1[rinvalid_mask] = 0
        rval = A - Bs[idx1]
        rval[rinvalid_mask] *=-1
        return np.minimum(lval, rval) <= tol


def load_ifr_behavior(ses,verbose=False):
    """
    Load data files for a session that we need to correlate ifr and navpath variables
    
    ifr_autopi.pkl: the instantaneous firing rate of all neurons as a function of time. 
    navPathSummary.csv: description of navPath that can be analyzed
    navPathInstan.csv: contains variables that changes within a single navPath (e.g., distance run and speed)
    
    """
    fn = ses.path+"/ifr_autopi.pkl"
    if verbose:
        print("loading",fn)
    with open(fn, 'rb') as intp:
        ifr = pickle.load(intp)
    
    fn = ses.path+"/navPathSummary.csv"
    if verbose:
        print("loading",fn)
    navPathSummary = pd.read_csv(fn)
    navPathSummary
    
    fn = ses.path+"/navPathInstan.csv"
    if verbose:
        print("loading",fn)
    navPathInstan = pd.read_csv(fn)
    navPathInstan
    
    
    # all the timeRes values in our navPathInstan should be in IFR time array
    intol = isin_tolerance(navPathInstan.timeRes,ifr[1],tol=0.00001) # function from setup_project.py
    if np.sum(intol) != navPathInstan.shape[0]:
        raise ValueError("Not all values of the navPathInstan are in the ifr time")
    
    return ifr, navPathSummary, navPathInstan

def loadMyProjectWithTrials(myProject):
    """
    Get the myProject object with the trial extraction already done
    All the navPaths are already there.
    """
    fn=myProject.dataPath+"/results/myProjectWithTrials.pickle"
    print("Loading:",fn)
    with open(fn, 'rb') as handle:
        myProject = pickle.load(handle)
    return myProject


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
    
def vectorAngle(v,rv=np.array([[1,0]]),degrees=False,quadrant=False) :
    """

    Calculate the angles between an array of vectors relative to a reference vector
    Argument:
        v: Array of vectors, one vector per row
        rv: Reference vector
        degrees: Boolean indicating whether to return the value as radians (False) or degrees (True)
        quadrant: Adjust the angle for 3 and 4 quadrants, assume rv is (1,0) and the dimension of v is 2.
    Return:
        Array of angles, between -pi/2 and p/2
    """
    # length of vector
    if v.shape[1]!=rv.shape[1]:
        print("v and rv should have the same number of column")
        return
    vLen = np.sqrt(np.sum(v*v,axis=1))
    vLen[vLen==0] = np.NAN
    rvLen = np.sqrt(np.sum(rv*rv,axis=1))

    # get unitary vectors
    uv = v/vLen[:,None]
    urv = rv/rvLen[:,None]

    # get the angle, dot product, then acos
    theta = np.arccos(np.clip(np.sum(uv*urv,axis=1),  -1.0, 1.0))

    if quadrant:
        # deal with the 3 and 4 quadrant
        theta[v[:,-1] < 0] = 2*np.pi - theta[v[:,-1]<0] 

    if degrees :
        theta = theta * 360 / (2*np.pi)

    return theta
    
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
    


def binCentersFromEdges(bins):
    """
    returns bin centers from bin edges, 
    The array returned is -1 in size compared to bins
    
    Arguments:
    bins: 1D arrays of bin edges with equal spacing. returned by histo functions
    """
    binSize=np.diff(bins)[0]
    binCenters=bins[:-1]+binSize/2
    return binCenters


def vectorLengthFromHisto(rates,rad):
    """
    Calculate the mean vector length of a polar histogram
    
    rates: firing rate in a bin
    rad: angle of the bin in radian
    """
    x = np.sum(np.cos(rad)*rates)
    y = np.sum(np.sin(rad)*rates)
    sr = np.sum(rates)
    x = x/sr
    y = y/sr
    return np.sqrt(x**2+y**2)


###
### Functions to calculate trial matrix correlation and its shuffling.
###

@jit()
def rollARow(a):
    """
    Function to roll the firing rate value of a single trial (one row)

    Only rolls the bins from first to last valid entries in the array
    
    The amount of shift is random
    """
    
    if np.sum(~np.isnan(a)) == 0:
        return a
    lastVal = np.where(~np.isnan(a))[0].max()
    firstVal = np.where(~np.isnan(a))[0].min()
    myRange = lastVal-firstVal
    A = a.copy()
    if myRange > 2:
        shift = np.random.randint(low = 0, high=myRange, size=1)[0]
        b = a[firstVal:(lastVal+1)]
        b = np.roll(b,shift)
        A[firstVal:lastVal+1] = b 
    return A


def shuffledTrialMap(m):
    return np.apply_along_axis(rollARow,axis=1,arr=m)


@jit()
def myOwnCorr(x,y,valx,valy):
    """
    Perform pearson correlation on 2 1D arrays, 
    It removes observations with np.nan
    """

    # keep only observations without np.nan
    indices= valx&valy
    #print("valid:",np.sum(indices))
    xx = x[indices]
    yy = y[indices]
    
    if np.sum(indices) < 2:
        return np.nan
    if np.all(xx==0):
        return np.nan
    if np.all(yy==0):
        return np.nan   
    else :
        return np.corrcoef(xx,yy)[0,1]
    


@jit()
def trialMatrixInternalCorrelation(m):
    """
    Calculate the mean of the correlation matrix (one half) of a matrix m, excluding the diagonal
    
    Argument:
    m: is a 2D matrix containing the firing rate on different trials. Each row is a trial.
    
    """
    # check which values are valid only once and not in the loops
    val = ~np.isnan(m)
    
    dim = m.shape[0]
    res = np.zeros(int(dim*(dim-1)/2)) # number of unique pairs of trials
    count = 0
    for i in range(dim):
        for j in range(i+1,dim):
            res[count] = myOwnCorr(m[i],m[j],val[i],val[j])
            count+=1
    return np.nanmean(res)


def trialMatrixInternalCorrelationShuffle(m,nShuf=10):
    """
    Calculate the mean of the correlation matrix (one half) of matrix m with matrix n, excluding the diagonal
    
    n is generated by shifting the rate values within each trial, keeping sequence but moving it forwards or backwards
    """
    resShuf = np.empty(nShuf) # to store results

    # check which values are valid only once and not in the loops
    valm = ~np.isnan(m)
    dim = m.shape[0]
    res = np.zeros(int(dim*(dim-1)/2)) # for all pairs of trials

    for x in range(nShuf):
        n = shuffledTrialMap(m)
        valn = ~np.isnan(n)
        
        count = 0
        for i in range(dim):
            for j in range(i+1,dim):
                res[count] = myOwnCorr(m[i],n[j],valm[i],valn[j])
                count+=1
        resShuf[x] = np.nanmean(res)
            
    return resShuf


############################################
### Function to standardize our figures  ###
############################################
def darkLightColors(lighter=False):
    """
    Create a palette that can be used to plot dark and light trials
    
    Use to keep color constant across figures.
    First color is for dark, second is for light
    """
    if lighter:
        rgb = [(50, 50, 120),(245, 212, 66)]  # took the values from Maryam's work in Inkscape
        colors = [tuple(t / 255 * 1.2 for t in x) for x in rgb]
    else:
        rgb = [(50, 50, 120),(245, 212, 66)]  # took the values from Maryam's work in Inkscape
        colors = [tuple(t / 255 for t in x) for x in rgb]
        
    return colors


def twoColorDictionary():
    """
    return a dictionary with 2 colors that can be used to plot
    
    """
    
    myDict= {1:(170/255,80/255,80/255),
            2: (80/255,170/255,80/255)}
    return myDict


def figurePanelDefaultSize():
    """
    Use to keep the size of panels similar across figures
    """
    return (1.8,1.8)

def setFigureDefaultParameters():
    plt.rc('axes', labelsize=10) #fontsize of the title
    plt.rc('axes', titlesize=10) #fontsize of the title
    plt.rc('axes', linewidth=1.25) 
    plt.rc("xtick.major", width=1.25)
    plt.rc("ytick.major", width=1.25)



def formatPValue(p,withE=True):
    """
    Format the p value to print on a plot
    It will format use latex to format P = 1.2 x 10-20 with -20 as supercript and P in italic.
    You can pass the returned value to ax.text().
    
    Argument:
    p: p-value
    withE: boolean, indicate whether to print in the 1.23e-10 format
    
    Example
    ax.text(1,1, formatPValue(p))
    
    
    """
    if p > 0.001:
        pString = "$P$ = {:.2}".format(p)
    else:
        if withE:
            pString = "$P$ = {:.2e}".format(p)
        else:
            sciNoti = "{:.1e}".format(p)
            base = sciNoti.split("e")[0]
            expo = sciNoti.split("e")[1]
            pString = "$P$ = {} x $10^{{{}}}$".format(base,expo)
    return pString

def set_ap_and_cg_intervals(ses, environment, trialNo,verbose=True):
    """
    Set the intervalse in ap and cg objects so that they are ready to make firing rate maps. 
    """
    trialCode = "{}_{}".format(environment,trialNo)
    sesTrials = ses.session_environment_trial_data_frame()
    envInter = np.array([[sesTrials[sesTrials.trialCode==trialCode].startTime.item(),sesTrials[sesTrials.trialCode==trialCode].endTime.item()]])
    ses.ap.set_intervals(envInter)
    for n in ses.cg.neuron_list:
        n.spike_train.set_intervals(envInter)
        
    if verbose:
        print("{} {} {} start: {:.3f}, duration: {:.3f}".format(ses.name, environment,trialNo,envInter[0,0], envInter[0,1]-envInter[0,0] ))
def unset_ap_and_cg_intervals(ses):
    ses.ap.unset_intervals()
    for n in ses.cg.neuron_list:
        n.spike_train.unset_intervals()

