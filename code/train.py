#!/usr/bin/env python
# coding: utf-8


import torch; import glob; import numpy as np; import pandas as pd
from monai.data import DataLoader, Dataset, CacheDataset
from monai import transforms as monai_trans
import monai
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import errno
import nibabel as nib
from datetime import datetime
import argparse
import sys

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import kde
from scipy.stats import gaussian_kde
import seaborn as sns

notebook = False
debug=False
local=True
render_model_figures=False

if notebook:
    print("Running notebook mode")
    sys.argv = ['']
    from tqdm.notebook import tqdm 
    
if local:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,0"
    torch.multiprocessing.set_sharing_strategy('file_system')
    
if not notebook:
    print("Running command line mode")
    from tqdm import tqdm
#     from tqdm.auto import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default='sex', help='the variable to be predicted, default = "sex"')
parser.add_argument("--inputs", nargs="+", default=["flair", "t1", "dwi", "metadata", "connectivity"], help='the data to be trained with, default = ["flair", "t1", "dwi", "metadata", "connectivity"]')
parser.add_argument("--metadata_blocks", nargs="+", default=["constitutional","serology","psychology","disease"], help='the metadata blocks to be passed with, default = ["constitutional","serology","psychology","disease"]')
parser.add_argument("--device", type=str, default='cuda:0', help='gpu card access, default = "cuda:0"')
parser.add_argument("--inpath", type=str, default='/data/BIOBANK/', help='the data path which contains the imaging, metadata and connectivity. It should contain the subdirectories of TEST and TRAIN, with subsequent directory cascades of T1, FLAIR, DWI, etc., default = "/data/BIOBANK/"')
parser.add_argument("--outpath", type=str, default='/data/BIOBANK/results_out/', help='the data path for writing results to, default = "/data/BIOBANK/results_out/"')
parser.add_argument("--epochs", type=int, default=999999, help='the number of epochs to train for, default = 999999')
parser.add_argument("--restore", type=str, default='no', help='whether you are continuing a halted training loop, default=False')
parser.add_argument("--batch_size", type=int, default=64, help='model batch size, default=64')
parser.add_argument("--num_workers", type=int, default=10, help='number of multiprocessing workers, default=18')
parser.add_argument("--resize", type=str, default='yes', help='Whether to resize the imaging, if passed as True shall work in 64 cubed, rather than the default 128, default=yes')
parser.add_argument("--veto_transformations", type=str, default='yes', help='Whether to prevent use of image transformations on the fly, default=yes since we have pre-computed them for speedup')
parser.add_argument("--augs_probability", type=int, default=0.05, help='probability of augmentation application, default=0.05')
parser.add_argument("--diagnostic", type=str, default='no', help='Whether to enable diagnostic mode, which re-encodes all input data as 1 and 0, for debugging purposes, default=no')
parser.add_argument("--data_size", type=int, default=999999999, help='Whether to reduce the size of the data for training, for debugging purposes, default = 999999999')
parser.add_argument("--multi_gpu", type=str, default='yes', help='Whether to enable,default=no')
parser.add_argument("--n_gpus", type=int, default=4, help='Number of GPUs to permit,default=4')
parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate,defauly=0.0001')
parser.add_argument("--patience", type=int, default=50, help='Model patience,default=30')
parser.add_argument("--cache_val_set", type=str, default='yes', help='whether to cache load the val dataset, default=yes')
parser.add_argument("--hello_world",help="hello world introduction",action="store_true",default=False)

args = parser.parse_args()





print("")

if debug:
    print("Running debugger")
    args.data_size=9999999999999999
    args.epochs=100
    args.inputs=['t1','flair','dwi']
    args.target='body_fat' 
    args.batch_size = 64
    args.num_workers= 32
    args.n_gpus=2
    args.multi_gpu='yes'
    args.cache_val_set='no'
    args.resize = 'yes'

if local:
#     args.inpath = '/home/jruffle/Desktop/brc2/james/BIOBANK/'
    args.inpath = '/mnt/wwn-0x5002538f4132e495/BIOBANK/'
    args.outpath = '/home/jruffle/Desktop/BIOBANK_MEGAMODELLER/'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,0"
    args.n_gpus=2
    args.multi_gpu='yes'
    args.cache_val_set='no'
    args.resize = 'yes'
    args.metadata_blocks =''
    
pombo = False
if pombo:
    ###can replace with your path to the brc2/james/BIOBANK/ - which is where the data is organised
    args.inpath = '/home/jruffle/Desktop/brc2/james/BIOBANK/'
    args.outpath = '/home/jruffle/Desktop/brc2/james/BIOBANK/'
    
print("")





all_imaging_keys = ["flair", "t1", "dwi"]
all_non_imaging_keys = ["connectivity", "metadata"]
all_possible_keys = all_imaging_keys + all_non_imaging_keys





"""
Configurable initialising parameters
"""
target = args.target
inputs = args.inputs

path = args.inpath
device = args.device
outpath = args.outpath

metadata_test_mlp = pd.read_csv(path+'TEST/metadata_test.csv',index_col=0).sort_values(by='biobank_id', ascending=True).reset_index(drop=True,inplace=False)

variables_to_drop = ['neuroticism','haematocrit','bmi']

constitutional_block = ['sex','age','weight','handedness']
disease_block = ['smoking','hypertension','atopy','asthma','body_fat']
serology_block = ['Hb','HDL','hba1c','LDL']
psychology_block = ['mood_swings', 'miserableness',
       'irritability', 'sensitivity', 'fed_up', 'nervous', 'anxious', 'tense',
       'worry', 'lonely', 'guilty','reaction_time']

metadata_test_mlp.drop(variables_to_drop,axis=1,inplace=True)





if args.hello_world:
    print("Welcome to the UK Biobank Megamodeller v3.0 - May 2022 Build")
    print("")
    print("This code functions to provide relatively easy automated modelling of large scale imaging data with a variety of customisable inputs and parameters, across a range of possible targets")
    print("")
    print("Pass code.py -h will provide a breakdown of the customisable argments")
    print("")
    print("There are a range of possible targets, these are as follows: "+str(metadata_test_mlp.columns[1:].values))
    print("")
    print("There are a range of possible data modelling inputs, these are as follows: "+str(all_possible_keys))
    print("")
    print("All queries to j.ruffle@ucl.ac.uk")
    sys.exit()





"""
Model name primer...
"""

inputs = sorted(inputs)
model_name = target

for i in inputs:
    model_name+='_'
    model_name+=str(i)
    
for i in args.metadata_blocks:
    model_name+='_'
    model_name+=str(i)

print("")
print("Model to be trained as follows: "+str(model_name))
print("")
print("Target is as follows: "+str(target))
print("Inputs are as follows: "+str(inputs))
print("")
print("")





# Seed
import random
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True





"""
Model hyperparameters
"""
try:
    os.mkdir(outpath+model_name)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir(outpath+model_name+'/checkpoints/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir(outpath+model_name+'/best_model/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir(outpath+model_name+'/results/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

checkpoint_folder = outpath+model_name+'/checkpoints/'
best_model_folder = outpath+model_name+'/best_model/'
results_folder = outpath+model_name+'/results/'

restore_model = args.restore

checkpoint_file_path = checkpoint_folder + 'state_dictionary_checkpoint.pt'
best_model_file_path = best_model_folder + 'best_model_state_dictionary.pt'
max_epochs = args.epochs
checkpoint_frequency = 1

batch_size = args.batch_size
num_workers = args.num_workers
patience = args.patience
patience_iterator=0
cnn_dropout_rate=0.1
dense_dropout_rate=0.1
connectivity_hidden_dropout_rate=0.5
learning_rate=args.lr
use_half_precision = False
apply_batch_norm = True
apply_batch_norm_cnn = True
apply_batch_norm_metadata = True
apply_batch_norm_connectivity=True
###changed to on 5/5/22
resize_input = args.resize
new_side_length = 64

cnn_channels_hidden = [32, 64, 128, 256, 256]
cnn_output_dims = 128 

if args.resize !='yes': ###only fo 128 cube
    cnn_channels_hidden = [32, 64, 128, 256, 256, 256]
    cnn_output_dims=256

#8 million params
mlp_connectivity_layer_widths = [64620, 128, 128]

mlp_metadata_layer_widths = [metadata_test_mlp.shape[1]-2, 128, 64, 32] ## try this

concatenated_width = cnn_output_dims + mlp_connectivity_layer_widths[-1] + mlp_metadata_layer_widths[-1]
mlp_logits_layer_widths = [concatenated_width, 256, 256, 2]

diagnostic = args.diagnostic
number_of_examples_to_use = args.data_size

performance_metrics_columns = ['mode','target','inputs','accuracy','precision','recall','f1','auc','training_time_mins','loss','r2','r']
performance_metrics = pd.DataFrame(np.zeros(shape=(1,len(performance_metrics_columns))),columns=performance_metrics_columns)
columns = ['list_of_training_accuracies_per_epoch','list_of_validation_accuracies_per_epoch','list_of_training_losses_per_epoch','list_of_validation_losses_per_epoch','list_of_training_times_per_epoch','list_of_validation_times_per_epoch']

fig_length=5
fig_height=5
nbins=300

print("")
print("Path information")
print("Data path is: "+str(path))
print("Using device: "+str(device))
print("Outpath is: "+str(outpath))
print("Checkpoints shall be saved at: "+str(checkpoint_folder))
print("Checkpoints shall be every "+str(checkpoint_frequency)+" epochs")
print("Best performing model shall be saved at: "+str(best_model_file_path))
print("Results shall be saved at: "+str(results_folder))

print("")
print("Hyperparamter information")
print("Batch size is: "+str(batch_size))
print("Number of workers is: "+str(num_workers))
print("Max number of epochs is: "+str(max_epochs))
print("Training patience is: "+str(patience))

print("")
print("Model information...")
print("cnn_channels_hidden"+str(cnn_channels_hidden))
print("mlp_connectivity_layer_widths"+str(mlp_connectivity_layer_widths))
print("mlp_metadata_layer_widths"+str(mlp_metadata_layer_widths))
print("concatenated_width"+str(concatenated_width))
print("mlp_logits_layer_widths"+str(mlp_logits_layer_widths))
print("")

###override the resize where no CNN, to speed things up:
if 't1' not in inputs and 'flair' not in inputs and 'dwi' not in inputs:
    print("Enabling resizing for faster learning with non CNN component")
    resize_input='yes'

if resize_input=='yes':
    print("Imaging side length shall be as follows: "+str(new_side_length))
if resize_input!='yes':
    print("Imaging side length shall be as follows: 128")

print("")
if diagnostic=='yes':
    print("***WARNING - training model in diagnostic mode***")





def clamp(train,test):
    perc01 = np.percentile(train, 0.5, keepdims=True)
    perc99 = np.percentile(train, 99.5, keepdims=True)
    clamped_train = np.clip(train, a_min=perc01, a_max=perc99)
    clamped_test = np.clip(test, a_min=perc01, a_max=perc99)
    return clamped_train, clamped_test

def zscore(train,test):
    m = np.mean(train)
    sd = np.std(train)
    zscored_train = (train-m)/sd
    zscored_test = (test-m)/sd
    return zscored_train, zscored_test

def normalise(train,test):
    normalised_train = 2*(train - np.min(train)) / (np.max(train) - np.min(train))-1
    normalised_test = 2*(test - np.min(train)) / (np.max(train) - np.min(train))-1
    return normalised_train, normalised_test

###implement undersampling to deal with class imbalances...
def undersampler(sample_definition,y,num_to_undersample):
    for i, val in enumerate(sample_definition):
        itemindex = np.where(y[:,i]==1)[0]
        sampler_def = np.random.choice(itemindex, num_to_undersample,replace=False)
        
        if i==0:
            ids = sampler_def.copy()
        if i>0:
            ids = np.insert(ids, np.arange(len(sampler_def)), sampler_def)

    return ids





class Error(Exception):
    """Base class for other exceptions"""
    pass





if render_model_figures:
    imaging_fmri_train=np.load(path+'TRAIN/imaging_fmri_train.npy')
    imaging_fmri_test=np.load(path+'TEST/imaging_fmri_test.npy')
    mean_connectivity = np.zeros(shape=(imaging_fmri_train.shape[1],imaging_fmri_train.shape[2]))
    for i in range(imaging_fmri_train.shape[0]):
        mean_connectivity=mean_connectivity+imaging_fmri_train[i,...]
    for i in range(imaging_fmri_test.shape[0]):
        mean_connectivity=mean_connectivity+imaging_fmri_test[i,...]
    mean_connectivity=mean_connectivity/(imaging_fmri_train.shape[0]+imaging_fmri_test.shape[0])
    sns.clustermap(mean_connectivity)
    plt.savefig('mean_connectivity.png', dpi=150,bbox_inches='tight')





"""
Setup the initial paths
"""
paths_t1_train = sorted(glob.glob(path+"TRAIN/T1/*"))[:number_of_examples_to_use]
paths_t1_test = sorted(glob.glob(path+"TEST/T1/*"))[:number_of_examples_to_use]

paths_fl_train = sorted(glob.glob(path+"TRAIN/FLAIR/*"))[:number_of_examples_to_use]
paths_fl_test = sorted(glob.glob(path+"TEST/FLAIR/*"))[:number_of_examples_to_use]

paths_dwi_train = sorted(glob.glob(path+"TRAIN/DWI/*"))[:number_of_examples_to_use]
paths_dwi_test = sorted(glob.glob(path+"TEST/DWI/*"))[:number_of_examples_to_use]

with open(path+'TRAIN/imaging_fmri_train.txt') as f:
    fmri_subs_tr = f.readlines()
fmri_subs_tr = list(map(lambda x:x.strip(),fmri_subs_tr))[:number_of_examples_to_use]

with open(path+'TEST/imaging_fmri_test.txt') as f:
    fmri_subs_ts = f.readlines()
fmri_subs_ts = list(map(lambda x:x.strip(),fmri_subs_ts))[:number_of_examples_to_use]

if 'connectivity' in inputs:
    imaging_fmri_train=np.load(path+'TRAIN/imaging_fmri_train.npy')[:number_of_examples_to_use]
    imaging_fmri_test=np.load(path+'TEST/imaging_fmri_test.npy')[:number_of_examples_to_use]
    # Print(imaging_fmri_train.shape)
    # Extract strict upper triangular part of the connectivitiy:
    imaging_fmri_train_upper_tri_ind = np.triu_indices(imaging_fmri_train.shape[-1], 1)
    imaging_fmri_test_upper_tri_ind = np.triu_indices(imaging_fmri_test.shape[-1], 1)

    imaging_fmri_train = imaging_fmri_train[:,
                                            imaging_fmri_train_upper_tri_ind[0],
                                            imaging_fmri_train_upper_tri_ind[1]]
    imaging_fmri_test = imaging_fmri_test[:,
                                            imaging_fmri_test_upper_tri_ind[0],
                                            imaging_fmri_test_upper_tri_ind[1]]

    #print(imaging_fmri_train.shape)
    
else: 
    dim_fmri_tr = np.load(path+'TRAIN/imaging_fmri_train.npy',mmap_mode='r')[:number_of_examples_to_use].shape[0]
    imaging_fmri_train = np.zeros(shape=(dim_fmri_tr, 64620))
    dim_fmri_ts = np.load(path+'TEST/imaging_fmri_test.npy',mmap_mode='r')[:number_of_examples_to_use].shape[0]
    imaging_fmri_test = np.zeros(shape=(dim_fmri_ts, 64620))

metadata_train_mlp = pd.read_csv(path+'TRAIN/metadata_train.csv',index_col=0).sort_values(by='biobank_id', ascending=True).reset_index(drop=True,inplace=False).iloc[:number_of_examples_to_use,:].astype(np.float32)
metadata_test_mlp = pd.read_csv(path+'TEST/metadata_test.csv',index_col=0).sort_values(by='biobank_id', ascending=True).reset_index(drop=True,inplace=False).iloc[:number_of_examples_to_use,:].astype(np.float32)

metadata_train_mlp.drop(variables_to_drop,axis=1,inplace=True)
metadata_test_mlp.drop(variables_to_drop,axis=1,inplace=True)

train_ids = metadata_train_mlp['biobank_id']
test_ids = metadata_test_mlp['biobank_id']

y_train = metadata_train_mlp[target].values.copy()
y_test = metadata_test_mlp[target].values.copy()

if target in constitutional_block:
    print("Zeroing the same domain block: "+str('constitutional'))
    metadata_train_mlp[constitutional_block]=0
    metadata_test_mlp[constitutional_block]=0
    
if target in disease_block:
    print("Zeroing the same domain block: "+str('disease'))
    metadata_train_mlp[disease_block]=0
    metadata_test_mlp[disease_block]=0
    
if target in serology_block:
    print("Zeroing the same domain block: "+str('serology'))
    metadata_train_mlp[serology_block]=0
    metadata_test_mlp[serology_block]=0
    
if target in psychology_block:
    print("Zeroing the same domain block: "+str('psychology_block'))
    metadata_train_mlp[psychology_block]=0
    metadata_test_mlp[psychology_block]=0
    
if 'constitutional' not in args.metadata_blocks:
    print("Zeroing the following block..."+str('constitutional'))
    metadata_train_mlp[constitutional_block]=0
    metadata_test_mlp[constitutional_block]=0
    
if 'disease' not in args.metadata_blocks:
    print("Zeroing the following block..."+str('disease'))
    metadata_train_mlp[disease_block]=0
    metadata_test_mlp[disease_block]=0
    
if 'serology' not in args.metadata_blocks:
    print("Zeroing the following block..."+str('serology'))
    metadata_train_mlp[serology_block]=0
    metadata_test_mlp[serology_block]=0
    
if 'psychology' not in args.metadata_blocks:
    print("Zeroing the following block..."+str('psychology'))
    metadata_train_mlp[psychology_block]=0
    metadata_test_mlp[psychology_block]=0

metadata_train_mlp.drop(['biobank_id',target],axis=1,inplace=True)
metadata_test_mlp.drop(['biobank_id',target],axis=1,inplace=True)

print("After dropping the target, the remaining metadata variables that can be used in training [if 'metadata' is passed] are as follows: "+str(metadata_train_mlp.columns))
print("")

metadata_train_mlp=metadata_train_mlp.values
metadata_test_mlp=metadata_test_mlp.values

print("number of unique y instances")
print(len(np.unique(y_train)))
print(y_train)

if len(np.unique(y_train)) > 2:
    print("")
    print("Continuous input, clamping, zscoring and normalising...")
    print("")
    continuous_target=True
    y_train,y_test = clamp(y_train,y_test)
    y_train,y_test = zscore(y_train,y_test)
    y_train,y_test = normalise(y_train,y_test)
    med = np.median(y_train)
    
    y_train = torch.tensor(y_train.astype(float))
    y_test = torch.tensor(y_test.astype(float))
    mlp_logits_layer_widths[-1]=1
    print("Revised mlp_logits_layer_widths: "+str(mlp_logits_layer_widths))
    print("")
    
    
else:
    print("Categorical input, casting to one hot encode...")
    continuous_target=False
    n_bins = len(np.unique(y_train))
    y_train = torch.tensor(y_train.astype(int))
    y_test = torch.tensor(y_test.astype(int)) 
    y_train = nn.functional.one_hot(y_train, num_classes=n_bins).to(torch.float32)
    y_test = nn.functional.one_hot(y_test, num_classes=n_bins).to(torch.float32)

print("")
print("Clamping, z-scoring and normalising metadata now...")





def check_order_train():
    t1_subs_tr = [i.split('_t1.nii.gz', 1)[0] for i in paths_t1_train] 
    t1_subs_tr = [i.split(args.inpath+'TRAIN/T1/', 1)[1] for i in t1_subs_tr] 

    fl_subs_tr = [i.split('_flair.nii.gz', 1)[0] for i in paths_fl_train] 
    fl_subs_tr = [i.split(args.inpath+'TRAIN/FLAIR/', 1)[1] for i in fl_subs_tr] 

    dwi_subs_tr = [i.split('_dwi.nii.gz', 1)[0] for i in paths_dwi_train] 
    dwi_subs_tr = [i.split(args.inpath+'TRAIN/DWI/', 1)[1] for i in dwi_subs_tr] 

    if dwi_subs_tr == t1_subs_tr == fl_subs_tr == list(train_ids.astype(int).astype(str)) == fmri_subs_tr and len(dwi_subs_tr)==imaging_fmri_train.shape[0]:
        print("Matched training inputs to targets stored, progressing")

    else:
        raise Error('Unmatched inputs, exiting...')





def check_order_remainder():
    t1_subs_ts = [i.split('_t1.nii.gz', 1)[0] for i in paths_t1_test] 
    t1_subs_ts = [i.split(args.inpath+'TEST/T1/', 1)[1] for i in t1_subs_ts] 

    fl_subs_ts = [i.split('_flair.nii.gz', 1)[0] for i in paths_fl_test] 
    fl_subs_ts = [i.split(args.inpath+'TEST/FLAIR/', 1)[1] for i in fl_subs_ts] 

    dwi_subs_ts = [i.split('_dwi.nii.gz', 1)[0] for i in paths_dwi_test] 
    dwi_subs_ts = [i.split(args.inpath+'TEST/DWI/', 1)[1] for i in dwi_subs_ts] 

    if dwi_subs_ts == t1_subs_ts == fl_subs_ts == list(test_ids.astype(int).astype(str)) == fmri_subs_ts and len(dwi_subs_ts)==imaging_fmri_test.shape[0]:
        print("Matched testing inputs to targets stored, progressing")

    else:
        raise Error('Unmatched inputs, exiting...')





print("")
check_order_train()
print("")





print("")
check_order_remainder()
print("")





#initialised here, needed again at epoch level

train_files = [{"connectivity":u,'flair': v, "t1": w, "dwi":x, "metadata":y, "y": z} for u, v, w, x, y, z in zip(imaging_fmri_train,paths_fl_train,paths_t1_train,paths_dwi_train,metadata_train_mlp, y_train)]
val_files = [{"connectivity":u,'flair': v, "t1": w, "dwi":x, "metadata":y, "y": z} for u, v, w, x, y, z in zip(imaging_fmri_test, paths_fl_test,paths_t1_test,paths_dwi_test, metadata_test_mlp, y_test)]





###zero the ununsed inputs...

empty_metadata = np.zeros(train_files[0]['metadata'].shape,dtype=np.float32)
empty_connectivity = np.zeros(train_files[0]['connectivity'].shape,dtype=np.float32)

if resize_input=='yes':
    empty_imaging = np.zeros([1,new_side_length,new_side_length,new_side_length],dtype=np.float32)
else:
    empty_imaging = np.zeros(nib.load(train_files[0]['t1']).get_fdata().shape,dtype=np.float32)
    empty_imaging = empty_imaging[np.newaxis,:]

zero_list = sorted(list(set(all_possible_keys) - set(inputs)))

for i in zero_list:
    print("Zeroing the following unused input: "+str(i))
    
    if i == 'metadata':
        update = {i:empty_metadata}
        
    if i == 'connectivity':
        update = {i:empty_connectivity}
        
    if i == 'dwi' or i=='flair' or i=='t1':
        update = {i:empty_imaging}
    
    train_files = [{**d,**update} for d in train_files]
    val_files = [{**d,**update} for d in val_files ]





# https://stackoverflow.com/questions/23130300/split-list-into-2-lists-corresponding-to-every-other-element
test_files = val_files[::2]
val_files = val_files[1::2]





print("")
print("Sample information")
print("Raw training sample size: "+str(len(train_files)))
print("Raw validation sample size: "+str(len(val_files)))
print("Raw test sample size: "+str(len(test_files)))





"""
Setup the input pipeline
"""
imaging_keys=list(set(inputs).intersection(all_imaging_keys))
non_imaging_keys = list(set(inputs).intersection(all_non_imaging_keys))
all_keys = imaging_keys + non_imaging_keys

print("imaging_keys: "+str(imaging_keys))

# Define input pipeline
from monai import transforms as monai_trans
from torch_wrapper import TorchIOWrapper
from torchio import transforms as torchio_trans
from clamp_by_percentile_augmentation import ClampByPercentile as ClampByPercentile

# Augmentation parameters
veto_transformations = args.veto_transformations
prob_of_application = args.augs_probability
histogram_shift_control_points = (3, 6)  # This might cause a more subtle change...
three_d_deform_magnitudes = (3, 5)  # Reduce if warping too much
three_d_deform_sigmas = (1, 3)
scale_range = (0.01, 0.01, 0.01)
translate_range = (2, 2, 2)
shear_angle_in_rads = 2 * (2 * 3.14159 / 360)  # the number on the far left is in degrees!
rot_angle_in_rads = 2 * (2 * 3.14159 / 360)
nii_target_shape=[128,128,128]
min_small_crop_size = [int(0.99 * d) for d in nii_target_shape]
max_inflation_factor = 0.05

if len(imaging_keys)>0:
    print("Cascading down 3D generator")

    # Load the images!
    train_transforms = [monai_trans.LoadImaged(keys=imaging_keys)]
    train_transforms += [monai_trans.AddChanneld(keys=imaging_keys)]  # PyTorch wants a (trailing) image channel

    if resize_input=='yes':
        train_transforms += [monai_trans.Resized(keys=imaging_keys,
                                                 spatial_size=[new_side_length] * 3,
                                                 allow_missing_keys=True)]

    if veto_transformations=='yes':
        print("Vetoing training augmentations!")
    else:
         print("Training augmentations skipped as pre-generated!")

        # The net should always see z-scored images rescaled into the same interval
    train_transforms += [monai_trans.NormalizeIntensityd(keys=imaging_keys)]
    train_transforms += [monai_trans.ScaleIntensityd(keys=imaging_keys, minv=-1.0, maxv=1.0)]

    # Finally, turn everything back into a numpy
    # train_transforms += [monai_trans.SqueezeDimd(keys=imaging_keys, dim=0)]  # Remove the singleton dimension that PyTorch needed
    # train_transforms += [monai_trans.ToNumpyd(keys=keys)]
    train_transforms += [monai_trans.ToTensord(keys=all_keys)]
    train_transforms = monai_trans.Compose(train_transforms)
    val_transforms = train_transforms
    
if len(imaging_keys)==0:
    print("Cascading down non-3D generator")
    train_transforms = [monai_trans.ToTensord(keys=all_keys)]
    train_transforms = monai_trans.Compose(train_transforms)
    val_transforms = train_transforms





#primed here, but overwritten later
#train set
dataset_train = Dataset(data=train_files, transform=train_transforms)

#val set
if args.cache_val_set=='yes':
    dataset_val = CacheDataset(data=val_files, transform=val_transforms,num_workers=num_workers)
else:
    dataset_val = Dataset(data=val_files, transform=train_transforms)
    
#test set
dataset_test = Dataset(data=test_files, transform=train_transforms)

loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True,
                        num_workers=num_workers,pin_memory=False)
loader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size, drop_last=False,
                        num_workers=num_workers,pin_memory=False)
loader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size, drop_last=False,
                        num_workers=num_workers,pin_memory=False)





class ConvLayers(nn.Module):
    """
    Applies multiple `convolution blocks', comprising a pair of convolutions with a skip conv, terminated with
    a single dense layer
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        self.incoming_side_length = kwargs['incoming_side_length']  # E.g. 128, for 128^3 images
        self.channels_input = kwargs['channels_input']  # E.g. 1 for one modality, or 3 for three
        self.channels_hidden = kwargs['channels_hidden']  # E.g. [8, 16, 32, 64, 128]
        self.kernel_size = 3  # Must be 1, 2 or 3 right now... (see below)
        self.dropout_rate = kwargs['dropout_rate']
        self.batch_norm = kwargs['batch_norm']
        self.output_dims = kwargs['output_dims']
        
        self.all_conv_channels = [self.channels_input] + self.channels_hidden
        self.hidden_activation = nn.GELU()
        self.output_activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.pool = nn.MaxPool3d(2, stride=2)

        if self.kernel_size == 1:
            self.pad = torch.nn.Identity()
        elif self.kernel_size == 2:
            self.pad = nn.ConstantPad2d((1, 0, 1, 0, 1, 0), 0)
        elif self.kernel_size == 3:
            self.pad = nn.ConstantPad2d((1, 1, 1, 1, 1, 1), 0)
        else:
            print("Kernels must be 1x or 2x or 3x. Quitting.")
            quit()
        
        """
        Determine the dimensions of the data emerging from the convolutional layers
        """
        self.final_side_length = self.incoming_side_length / (2 ** len(self.channels_hidden))
        self.final_conv_dims = int(self.channels_hidden[-1] * (self.final_side_length ** 3))
                
        self.dense_layer = nn.Linear(in_features=self.final_conv_dims,
                                     out_features=self.output_dims, bias=True)
        
        self.final_batch_norm = nn.BatchNorm1d(num_features=self.output_dims,
                                                 affine=True, track_running_stats=True)
        
        """
        Initialise convolution operations
        """
        self.conv_layers = nn.ModuleList()
        self.skip_conv_layers = nn.ModuleList()
        for k in range(len(self.all_conv_channels)-1):
            self.skip_conv_layers.append(nn.Conv3d(in_channels=self.all_conv_channels[k],
                                                   out_channels=self.all_conv_channels[k+1],
                                                   kernel_size=1, stride=1, padding=0, bias=True))
            self.conv_layers.append(nn.Conv3d(in_channels=self.all_conv_channels[k],
                                              out_channels=self.all_conv_channels[k+1],
                                              kernel_size=self.kernel_size, stride=1,
                                              padding=0, bias=True))

            self.conv_layers.append(nn.Conv3d(in_channels=self.all_conv_channels[k+1],
                                              out_channels=self.all_conv_channels[k+1],
                                              kernel_size=self.kernel_size, stride=1,
                                              padding=0, bias=True))
            
        """
        Initialise batch norm layers
        """
        self.bn_layers = nn.ModuleList()
        for channel in self.channels_hidden:
            if self.batch_norm:
                self.bn_layers.append(nn.BatchNorm3d(num_features=channel,
                                                     affine=True, track_running_stats=True))
            else:
                self.bn_layers.append(torch.nn.Identity())
        
        
    def forward(self, x):
        """
        Convolutional layers
        """
        for k in range(len(self.channels_hidden)):
            
            """
            Skip connection
            """
            x_skip = self.skip_conv_layers[k](x)
            
            """
            3x3x3 convolutions (with padding to stop the feature maps changing size)
            """
            x = self.conv_layers[2*k](self.pad(x))
            x = self.hidden_activation(x)
            x = self.conv_layers[2*k + 1](self.pad(x))
            x += x_skip
            x = self.bn_layers[k](x)
            x = self.hidden_activation(x)
            x = self.pool(x)
            x = self.dropout(x)
                
        """
        Dense layer
        """
        x = torch.flatten(x, start_dim=1)        
        x = self.dense_layer(x)
        x = self.final_batch_norm(x)
        x = self.output_activation(x)
        
        return x





class DenseLayers(nn.Module):
    """

    """
    def __init__(self, **kwargs):
        super().__init__()

#         self.input_shape = kwargs['input_shape']
#         self.output_shape = kwargs['output_shape']
#         self.dims = [self.input_shape, 512, 256, 128, self.output_shape]
        
        self.dims = kwargs['layer_widths']
        self.apply_hidden_dropout = kwargs['apply_hidden_dropout']
        self.apply_output_dropout = kwargs['apply_output_dropout']
        self.dropout_rate = kwargs['dropout_rate']
        self.hidden_activation = nn.GELU()
        self.output_activation = kwargs['output_activation']
        self.hidden_batch_norm = kwargs['hidden_batch_norm']
        self.output_batch_norm = kwargs['output_batch_norm']
        
        """
        Initialise dense layers
        """
        self.dense_layers = nn.ModuleList()
        for k in range(len(self.dims)-1):
            self.dense_layers.append(nn.Linear(in_features=self.dims[k],
                                               out_features=self.dims[k+1], bias=True))
    
        """
        Initialise batch norm layers
        """
        self.bn_layers = nn.ModuleList()
        for k in range(len(self.dims)-2):
            if self.hidden_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(num_features=self.dims[k+1],
                                                     affine=True, track_running_stats=True))
            else:
                self.bn_layers.append(torch.nn.Identity())

        if self.output_batch_norm:
            # When producing logits we must skip this final batch norm layer!
            self.bn_layers.append(nn.BatchNorm1d(num_features=self.dims[-1],
                                                 affine=True, track_running_stats=True))
            
        self.drop = nn.Dropout(self.dropout_rate)
        
        
    def forward(self, x):
        """
        This is where we pass the data through the layers
        """
        x = torch.flatten(x, start_dim=1)
        
        for k in range(len(self.dense_layers)):
            """
            Apply affine transformations and batch normalisation
            """
            x = self.dense_layers[k](x)
            
            if not(k == len(self.dense_layers) - 1 and not self.output_batch_norm):
                # Skip batch norm on the final layer only if we have said output_batch_norm=False
                x = self.bn_layers[k](x)
            
            """
            Apply activations
            """
            if k < len(self.dense_layers) - 1:
                x = self.hidden_activation(x)
            elif self.output_activation is not None:
                x = self.output_activation(x)
            
            """
            Apply dropout
            """
            if k < len(self.dense_layers) - 1:
                if self.apply_hidden_dropout:
                    x = self.drop(x)
            else:
                if self.apply_output_dropout:
                    x = self.drop(x)
            
        
        return x





"""
Initialise the models
"""

if resize_input=='yes':
    incoming_side_length = new_side_length
else:
    incoming_side_length = 128
    
print("Incoming side length is: "+str(incoming_side_length))    

cnn = ConvLayers(incoming_side_length=incoming_side_length,
                 channels_input=3, 
                 channels_hidden=cnn_channels_hidden,
                 dropout_rate=cnn_dropout_rate, 
                 batch_norm=apply_batch_norm_cnn,
                 output_dims=cnn_output_dims).to(device)

mlp_connectivity = DenseLayers(layer_widths=mlp_connectivity_layer_widths,
                               apply_hidden_dropout=connectivity_hidden_dropout_rate>0,
                               apply_output_dropout=dense_dropout_rate>0,
                               dropout_rate=connectivity_hidden_dropout_rate,
                               output_activation=torch.nn.GELU(),
                               hidden_batch_norm=apply_batch_norm_connectivity,
                               output_batch_norm=True).to(device)

mlp_metadata = DenseLayers(layer_widths=mlp_metadata_layer_widths,
                           apply_hidden_dropout=dense_dropout_rate>0,
                           apply_output_dropout=dense_dropout_rate>0,
                           dropout_rate=dense_dropout_rate,
                           output_activation=torch.nn.GELU(),
                           hidden_batch_norm=apply_batch_norm_metadata,
                           output_batch_norm=True).to(device)


mlp_logits = DenseLayers(layer_widths=mlp_logits_layer_widths,
                         apply_hidden_dropout=dense_dropout_rate>0,
                         apply_output_dropout=False,
                         dropout_rate=dense_dropout_rate,
                         output_activation=None,
                         hidden_batch_norm=apply_batch_norm,
                         output_batch_norm=False).to(device)





if args.multi_gpu=='yes':
    cnn = torch.nn.DataParallel(cnn, device_ids=list(range(args.n_gpus)),output_device=None)
    mlp_connectivity = torch.nn.DataParallel(mlp_connectivity, device_ids=list(range(args.n_gpus)),output_device=None)
    mlp_metadata = torch.nn.DataParallel(mlp_metadata, device_ids=list(range(args.n_gpus)),output_device=None)
    mlp_logits = torch.nn.DataParallel(mlp_logits, device_ids=list(range(args.n_gpus)),output_device=None)





"""
Setup the optimiser
"""

params = []

if 'metadata' in inputs:
    print("Adding metadata params")
    params += list(mlp_metadata.parameters())

if 'connectivity' in inputs:
    print("Adding connectivity params")
    params += list(mlp_connectivity.parameters())

if 'dwi' or 'flair' or 't1' in inputs:
    print("Adding CNN params")
    params += list(cnn.parameters())
    
params += list(mlp_logits.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

scaler = torch.cuda.amp.GradScaler(enabled=use_half_precision)





# Function to tally the regularisation terms


def sum_non_bias_l2_norms(parameters, multiplier=1e-4, stronger_regularisation_exception=[[128, 64620], 1e-3]):
    """
    Given parameters=model.parameters() where model is a PyTorch model, this iterates through the list and tallies
    the L2 norms of all the non-bias tensors.
    
    We have added 'stronger_regularisation_exception' to enable the specification of a single exceptional parameter
    identified by it's 2D shape, stronger_regularisation_exception[0], whose regularisation coefficient is given 
    by stronger_regularisation_exception[1].
    """
    l2_reg = 0
    for param in parameters:
        
        current_shape = list(param.size())
        
        if len(current_shape) > 1:
            if stronger_regularisation_exception is not None and stronger_regularisation_exception[0][0] == current_shape[0] and stronger_regularisation_exception[0][1] == current_shape[1]:
                l2_reg += stronger_regularisation_exception[1] * torch.mean(torch.square(param))
            else:
                l2_reg += multiplier * torch.mean(torch.square(param))

    return l2_reg


# Function to count the params in a model
def count_unique_parameters(parameters):
    # Only counts unique params
    count = 0
    list_of_names = []
    for p in parameters:
        name = p[0]
        param = p[1]
        if name not in list_of_names:
            list_of_names.append(name)
            count += np.prod(param.size())
    return count





def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    if sum_stats:
        accuracy  = np.trace(cf) / float(np.sum(cf))

        if len(cf)==2:
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if figsize==None:
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        categories=False

    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)





"""
Right before the training loop, with everything initialised, we can restore model parameters from a checkpoint
"""
if restore_model=='yes':
    print("Restoring model!")
    checkpoint = torch.load(checkpoint_file_path,map_location="cpu")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    starting_epoch = checkpoint['epoch'] + 1
    
    list_of_training_losses_per_epoch = checkpoint['list_of_training_losses_per_epoch']
    list_of_training_accuracies_per_epoch = checkpoint['list_of_training_accuracies_per_epoch']
    list_of_training_times_per_epoch = checkpoint['list_of_training_times_per_epoch']
    
    list_of_validation_losses_per_epoch = checkpoint['list_of_validation_losses_per_epoch']
    list_of_validation_accuracies_per_epoch = checkpoint['list_of_validation_accuracies_per_epoch']
    list_of_validation_times_per_epoch = checkpoint['list_of_validation_times_per_epoch']
    
    mlp_connectivity.load_state_dict(checkpoint['mlp_connectivity_state_dict'])
    mlp_metadata.load_state_dict(checkpoint['mlp_metadata_state_dict'])
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    mlp_logits.load_state_dict(checkpoint['mlp_logits_state_dict'])
    
else:
    starting_epoch = 0
    list_of_training_losses_per_epoch = []
    list_of_training_accuracies_per_epoch = []
    list_of_training_times_per_epoch = []
    list_of_validation_losses_per_epoch = []
    list_of_validation_accuracies_per_epoch = []
    list_of_validation_times_per_epoch=[]





print("Number of parameters in the cnn: " + str(count_unique_parameters(cnn.named_parameters())))
print("Number of parameters in the mlp_connectivity: " + str(count_unique_parameters(mlp_connectivity.named_parameters())))
print("Number of parameters in the mlp_metadata: " + str(count_unique_parameters(mlp_metadata.named_parameters())))
print("Number of parameters in the mlp_logits: " + str(count_unique_parameters(mlp_logits.named_parameters())))

total_param_count = count_unique_parameters(cnn.named_parameters())
total_param_count += count_unique_parameters(mlp_connectivity.named_parameters())
total_param_count += count_unique_parameters(mlp_metadata.named_parameters())
total_param_count += count_unique_parameters(mlp_logits.named_parameters())

print("Total number of parameters in the model: " + str(total_param_count))





"""
Initialise Tensorboard

"""
from torch.utils.tensorboard import SummaryWriter
tensorboard_dir = checkpoint_folder
writer = SummaryWriter(log_dir=tensorboard_dir)
print("To start Tensorboard execute the following in the terminal: tensorboard --logdir " + tensorboard_dir + " --port=8008")






if render_model_figures:
    from torchviz import make_dot

    counter=0
    for batch in loader_val:
        print(batch)
        counter+=1
        if counter >0:
            break

    flair = batch['flair'].to(device, non_blocking=True)
    t1 = batch['t1'].to(device, non_blocking=True)
    dwi = batch['dwi'].to(device, non_blocking=True)
    connectivity = batch['connectivity'].float().to(device, non_blocking=True)
    metadata = batch['metadata'].to(device, non_blocking=True)
    y = batch['y'].to(device, non_blocking=True)
    all_imaging = torch.cat((flair, t1, dwi), dim=1)
    labels = y.to(device, non_blocking=True)

    if continuous_target:
        labels = labels[:,np.newaxis]

    """
    Selective Forward pass
    """

    if 'metadata' in zero_list:
        output_metadata = torch.zeros_like(torch.empty(len(y),mlp_metadata_layer_widths[-1])).to(device, non_blocking=True)

    else:
        output_metadata = mlp_metadata(metadata.float())

    if 'connectivity' in zero_list:
        output_connectivity = torch.zeros_like(torch.empty(len(y),mlp_connectivity_layer_widths[-1])).to(device, non_blocking=True)

    else:
        output_connectivity = mlp_connectivity(connectivity)

    if 'dwi' in zero_list and 'flair' in zero_list and 't1' in zero_list:
        output_cnn = torch.zeros_like(torch.empty(len(y),cnn_output_dims)).to(device, non_blocking=True)

    else:
        output_cnn = cnn(all_imaging)


    data_concatenated = torch.cat((output_connectivity, output_metadata, output_cnn), dim=1)
    logits = mlp_logits(data_concatenated)

    dot = make_dot(output_metadata.mean(), params=dict(mlp_metadata.named_parameters()))
    dot.format = 'png'
    dot.render('metadata_architecture')

    dot = make_dot(output_cnn.mean(), params=dict(cnn.named_parameters()))
    dot.format = 'png'
    dot.render('cnn_architecture')

    dot = make_dot(output_connectivity.mean(), params=dict(mlp_connectivity.named_parameters()))
    dot.format = 'png'
    dot.render('connectivity_architecture')

    dot = make_dot(logits.mean(), params=dict(mlp_logits.named_parameters()))
    dot.format = 'png'
    dot.render('logits_architecture')





###initialise what directory to pull data from at the epoch level
epoch_paths = np.tile(range(len(glob.glob(path+'AUGMENTATIONS/TRAIN/*'))), 9999)[:max_epochs]





"""
Define what happens in one epoch of training/testing
"""
def one_epoch(input_dictionary):
    start = datetime.now()
    
    epoch = input_dictionary['epoch']
    optimizer = input_dictionary['optimizer']
    writer = input_dictionary['writer']
    mlp_connectivity = input_dictionary['mlp_connectivity']
    mlp_metadata = input_dictionary['mlp_metadata']
    cnn = input_dictionary['cnn']
    mlp_logits = input_dictionary['mlp_logits']
    params = input_dictionary['params']
    training = input_dictionary['training']
    mode_label = input_dictionary['mode_label']
    
    loss_over_whole_epoch = 0
    accuracy_over_whole_epoch = 0
    epoch_time = 0
    predictions = np.empty([0, 2])
    predictions_continuous = np.empty([0,1])
    ground_truth = np.empty([0, 2])
    ground_truth_continuous = np.empty([0,1])
    
    if training:
        print("")
        print("Training")
        print("")
        epoch_path=epoch_paths[epoch]
        print("Pulling in data from epoch_path: "+str(epoch_path))
        paths_t1_train = sorted(glob.glob(path+'AUGMENTATIONS/TRAIN/'+str(epoch_path)+'/T1/*'))[:number_of_examples_to_use]
        paths_fl_train = sorted(glob.glob(path+'AUGMENTATIONS/TRAIN/'+str(epoch_path)+'/FLAIR/*'))[:number_of_examples_to_use]
        paths_dwi_train = sorted(glob.glob(path+'AUGMENTATIONS/TRAIN/'+str(epoch_path)+'/DWI/*'))[:number_of_examples_to_use]
        print("")
        check_order_train()
        print("")
        train_files = [{"connectivity":u,'flair': v, "t1": w, "dwi":x, "metadata":y, "y": z} for u, v, w, x, y, z in zip(imaging_fmri_train,paths_fl_train,paths_t1_train,paths_dwi_train,metadata_train_mlp, y_train)]
        
        empty_metadata = np.zeros(train_files[0]['metadata'].shape,dtype=np.float32)
        empty_connectivity = np.zeros(train_files[0]['connectivity'].shape,dtype=np.float32)
        
        if resize_input=='yes':
            empty_imaging = np.zeros([1,new_side_length,new_side_length,new_side_length],dtype=np.float32)
        else:
            empty_imaging = np.zeros(nib.load(train_files[0]['t1']).get_fdata().shape,dtype=np.float32)
            empty_imaging = empty_imaging[np.newaxis,:]
            
        zero_list = sorted(list(set(all_possible_keys) - set(inputs)))
        for i in zero_list:
            print("Zeroing the following unused input: "+str(i))

            if i == 'metadata':
                update = {i:empty_metadata}

            if i == 'connectivity':
                update = {i:empty_connectivity}

            if i == 'dwi' or i=='flair' or i=='t1':
                update = {i:empty_imaging}

            train_files = [{**d,**update} for d in train_files]

        if len(np.unique(y_train)) == 2:
            samples = np.sum(y_train.cpu().detach().numpy(),axis=0,dtype=int)
            classes = samples.shape[0]
            majority_class = np.where(samples==np.amax(samples))[0][0]
            minority_class = np.where(samples==np.amin(samples))[0][0]
            num_to_undersample = int(y_train[:,minority_class].sum())
            model_resampler = undersampler(samples,y_train,num_to_undersample)
            train_files = np.array(train_files)[model_resampler].tolist()

        dataset_train = Dataset(data=train_files, transform=train_transforms)
        loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True,
                            num_workers=num_workers,pin_memory=False)
            
        mlp_connectivity.train()
        mlp_metadata.train()
        cnn.train()
        mlp_logits.train()

    else:
        print("")
        print(mode_label)
        print("")
        zero_list = list(set(all_possible_keys) - set(inputs))
        if mode_label=='Validation':
            loader = loader_val
        
        if mode_label=='Testing':
            loader = loader_test
            
        mlp_connectivity.eval()
        mlp_metadata.eval()
        cnn.eval()
        mlp_logits.eval()

    #####    
    progress_bar = tqdm(loader, mode_label + ") Epoch " + str(epoch))
    
    for batch in progress_bar:

        if training:
            optimizer.zero_grad()
            
        progress_bar_dict = {}

        """
        Grab data from the batch and send to the gpu
        """
        flair = batch['flair'].to(device, non_blocking=True)
        t1 = batch['t1'].to(device, non_blocking=True)
        dwi = batch['dwi'].to(device, non_blocking=True)
        connectivity = batch['connectivity'].float().to(device, non_blocking=True)
        metadata = batch['metadata'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        all_imaging = torch.cat((flair, t1, dwi), dim=1)
        labels = y.to(device, non_blocking=True)
        
#         if debug:
#             print("raw label shape")
#             print(labels.shape)
        
        if continuous_target:
            labels = labels[:,np.newaxis]

        """
        Diagnostic
        """
        if diagnostic=='yes' and continuous_target==False:
            ##This modifies the inputs and outputs to both be simply 1s and 0s to check model is learning...
            for k in range(len(labels)):
                if labels[k, 0] == 1:
                    flair[k, ...] = 0
                    t1[k, ...] = 0
                    dwi[k, ...] = 0
                    all_imaging[k, ...] = 0
                    connectivity[k, ...] = 0
                    metadata[k, ...] = 0

                elif labels[k, 1] == 1:
                    flair[k, ...] = 1
                    t1[k, ...] = 1
                    dwi[k, ...] = 1
                    all_imaging[k, ...] = 1
                    connectivity[k, ...] = 1
                    metadata[k, ...] = 1

                else:
                    quit()
                    
        if diagnostic=='yes' and continuous_target==True:
            ##This modifies the inputs and outputs to both be simply 0s to check model is learning...
            labels=torch.zeros_like(labels)
            labels = 50 * torch.ones_like(labels)
            metadata = torch.zeros_like(metadata)
            all_imaging = torch.zeros_like(all_imaging)
            connectivity = torch.zeros_like(connectivity)
            
        """
        Selective Forward pass
        """
        
        if 'metadata' in zero_list:
            output_metadata = torch.zeros_like(torch.empty(len(y),mlp_metadata_layer_widths[-1])).to(device, non_blocking=True)
            
        else:
            output_metadata = mlp_metadata(metadata.float())

        if 'connectivity' in zero_list:
            output_connectivity = torch.zeros_like(torch.empty(len(y),mlp_connectivity_layer_widths[-1])).to(device, non_blocking=True)
            
        else:
            output_connectivity = mlp_connectivity(connectivity)
            
        if 'dwi' in zero_list and 'flair' in zero_list and 't1' in zero_list:
            output_cnn = torch.zeros_like(torch.empty(len(y),cnn_output_dims)).to(device, non_blocking=True)
            
        else:
            output_cnn = cnn(all_imaging)

     
        data_concatenated = torch.cat((output_connectivity, output_metadata, output_cnn), dim=1)
        logits = mlp_logits(data_concatenated)
        
#         if debug:
#             print("post model shapes")
#             print(logits.shape)
#             print(labels.shape)
        
        if logits.shape != labels.shape:
            raise Error('Logits shape != target, exiting...')
    
    
        """
        Evaluate the loss
        """
        if continuous_target:
            probabilities = logits
            loss = torch.nn.functional.mse_loss(input=probabilities.float(), target=labels.float())
            
        else:
            probabilities = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy(input=probabilities, target=labels)
        
        """
        Regularise the non-bias params
        """
        loss += sum_non_bias_l2_norms(params)
        
        """
        Backwards pass, then update the weights
        """
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            """
            Predictions needed for confusion matrices
            """
            if continuous_target:
                probabilities = probabilities.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                
                ####create one-hot representation of continuous target just so other plotting features still persist
                predictions_continuous_to_one_hot = np.zeros([len(probabilities), 2])
                labels_continuous_to_one_hot = np.zeros([len(labels), 2])

                predictions_continuous_to_one_hot[np.argwhere(probabilities<=med),0]=1
                predictions_continuous_to_one_hot[np.argwhere(probabilities>med),1]=1
                
                probabilities_continuous = probabilities.copy() ###save the continuous target under revised name
                probabilities= torch.tensor(predictions_continuous_to_one_hot) ###replace the continuous target with the revised one-hot representation

                labels_continuous = labels.copy()
                labels_continuous_to_one_hot[np.argwhere(labels<=med),0]=1
                labels_continuous_to_one_hot[np.argwhere(labels>med),1]=1
                labels=torch.tensor(labels_continuous_to_one_hot)

            if not continuous_target: ###i.e. is binary target
                probabilities_continuous = np.empty(shape=(len(probabilities),1))##since we don't need this
                labels_continuous = np.empty(shape=(len(probabilities),1))##since we don't need this
                labels_continuous_to_one_hot = labels.cpu().detach().numpy()

            predictions=np.append(predictions,probabilities.cpu().detach().numpy(),axis=0)
            predictions_continuous=np.append(predictions_continuous,probabilities_continuous,axis=0)
            ground_truth = np.append(ground_truth,labels.cpu().detach().numpy(),axis=0)

            ground_truth_continuous = np.append(ground_truth_continuous,labels_continuous,axis=0)
   
            
            """
            Compute the accuracy
            """
            predicted_classes = probabilities > 0.5
            matches = torch.eq(predicted_classes, labels).float()
            accuracy = 100 * torch.mean(matches)

            """
            Manually create a tally of the losses
            """
            loss_over_whole_epoch += loss.cpu().detach().numpy()
            accuracy_over_whole_epoch += accuracy.cpu().detach().numpy()

            """
            Send extra info to the progress bar
            """
            progress_bar_dict['Loss'] = loss.cpu().detach().numpy()
            progress_bar_dict['Accuracy'] = accuracy.cpu().detach().numpy()
            progress_bar.set_postfix(progress_bar_dict)
            
            """
            Send info to Tensorboard
            """
            writer.add_scalar(mode_label + "Loss", loss.cpu().detach().numpy(), epoch)
            writer.add_scalar(mode_label + "Accuracy", accuracy.cpu().detach().numpy(), epoch)  
                                
    with torch.no_grad():
        """
        Update our manually-created tallies
        """
        loss_over_whole_epoch /= len(loader)
        accuracy_over_whole_epoch /= len(loader)
    
    end = datetime.now()
    difference = end - start
    seconds_in_day = 24 * 60 * 60
    difference = divmod(difference.days * seconds_in_day + difference.seconds, 60)
    epoch_time= difference[0]+(difference[1]/60) ###in minutes
    
    output_dictionary = {'loss_over_whole_epoch': loss_over_whole_epoch,
                        'accuracy_over_whole_epoch': accuracy_over_whole_epoch,
                        'epoch_time':epoch_time,
                        'predictions':predictions,
                        'predictions_continuous':predictions_continuous,
                         'labels':labels,
                         'probabilities':probabilities,
                         'ground_truth':ground_truth,
                         'ground_truth_continuous':ground_truth_continuous,
                         'labels_continuous':labels_continuous,
                        'labels_continuous_to_one_hot':labels_continuous_to_one_hot,
                        }
    
    return output_dictionary
        





"""
Start training/validating
"""
for epoch in range(starting_epoch, max_epochs):
    """
    Training

    """
    input_dictionary = {'mode_label': 'Training',
                        'training': True,
                        'epoch': epoch,
                        'optimizer': optimizer,
                        'writer': writer,
                        'mlp_connectivity': mlp_connectivity,
                        'mlp_metadata': mlp_metadata,
                        'cnn': cnn,
                        'mlp_logits': mlp_logits,
                        'params': params}
    
    output_dictionary_train = one_epoch(input_dictionary)
    
    list_of_training_losses_per_epoch.append(output_dictionary_train['loss_over_whole_epoch'])
    list_of_training_accuracies_per_epoch.append(output_dictionary_train['accuracy_over_whole_epoch'])
    list_of_training_times_per_epoch.append(output_dictionary_train['epoch_time'])

    print("Average training loss over epoch " + str(epoch) + ": " + str(list_of_training_losses_per_epoch[-1]))
    print("Average training accuracy over epoch " + str(epoch) + ": " + str(list_of_training_accuracies_per_epoch[-1]))
    print("Epoch training time (mins): " + str(round(list_of_training_times_per_epoch[-1],3)))
        
    """
    Validating

    """
    with torch.no_grad():
        input_dictionary = {'mode_label': 'Validation',
                            'training': False,
                            'epoch': epoch,
                            'optimizer': optimizer,
                            'writer': writer,
                            'mlp_connectivity': mlp_connectivity,
                            'mlp_metadata': mlp_metadata,
                            'cnn': cnn,
                            'mlp_logits': mlp_logits,
                            'params': params}

        output_dictionary = one_epoch(input_dictionary)

        list_of_validation_losses_per_epoch.append(output_dictionary['loss_over_whole_epoch'])
        list_of_validation_accuracies_per_epoch.append(output_dictionary['accuracy_over_whole_epoch'])
        list_of_validation_times_per_epoch.append(output_dictionary['epoch_time'])
        val_predictions = output_dictionary['predictions']
        val_predictions_continuous=output_dictionary['predictions_continuous']
        ground_truth = output_dictionary['ground_truth']
        ground_truth_continuous = output_dictionary['ground_truth_continuous']

        print("Average validation loss over epoch " + str(epoch) + ": " + str(list_of_validation_losses_per_epoch[-1]))
        print("Average validation accuracy over epoch " + str(epoch) + ": " + str(list_of_validation_accuracies_per_epoch[-1]))
        print("Epoch validation time (mins): " + str(round(list_of_validation_times_per_epoch[-1],3)))
        

        """
        Code for saving checkpoints
        """
        if epoch % checkpoint_frequency == 0:
            print("Saving checkpoint")
            checkpoint_dict = {'epoch': epoch,
                               'mlp_connectivity_state_dict': mlp_connectivity.state_dict(),
                               'mlp_metadata_state_dict': mlp_metadata.state_dict(),
                               'cnn_state_dict': cnn.state_dict(),
                               'mlp_logits_state_dict': mlp_logits.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'scaler_state_dict': scaler.state_dict(),
                               'list_of_training_losses_per_epoch': list_of_training_losses_per_epoch,
                               'list_of_training_accuracies_per_epoch': list_of_training_accuracies_per_epoch,
                               'list_of_training_times_per_epoch':list_of_training_times_per_epoch,
                               'list_of_validation_losses_per_epoch': list_of_validation_losses_per_epoch,
                               'list_of_validation_accuracies_per_epoch': list_of_validation_accuracies_per_epoch,
                               'list_of_validation_times_per_epoch':list_of_validation_times_per_epoch}

            torch.save(checkpoint_dict, checkpoint_file_path)
            patience_iterator+=1
            
            training_progress = pd.DataFrame(np.zeros(shape=((checkpoint_dict['epoch']+1),len(columns))),columns=columns)

            for i in training_progress.columns:
                training_progress[i] = checkpoint_dict[i]

            training_progress.reset_index(inplace=True)
            training_progress.rename(columns={"index": "epoch"},inplace=True)
            training_progress.to_csv(results_folder+'training_progress.csv')

            plt.figure(figsize=(fig_length,fig_height))
            plt.plot(training_progress['epoch'].values, training_progress['list_of_training_accuracies_per_epoch'].values, label = "Training accuracy %")
            plt.plot(training_progress['epoch'].values, training_progress['list_of_validation_accuracies_per_epoch'].values, label = "Validation accuracy %")
            plt.xlabel('Epochs')
            plt.ylabel('% Accuracy')
            plt.title("Input(s): "+str(inputs) +"; Target: "+str(target))
            plt.legend(loc="lower right")
            plt.savefig(results_folder+'accuracy_plot.png',dpi=150,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(fig_length,fig_height))
            plt.plot(training_progress['epoch'].values, training_progress['list_of_training_losses_per_epoch'].values, label = "Training loss")
            plt.plot(training_progress['epoch'].values, training_progress['list_of_validation_losses_per_epoch'].values, label = "Validation loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title("Input(s): "+str(inputs) +"; Target: "+str(target))
            plt.legend(loc="upper right")
            plt.savefig(results_folder+'loss_plot.png',dpi=150,bbox_inches='tight')
            plt.close()

            val_predictions_argmax = np.argmax(val_predictions,1)
            
            if continuous_target:
                r2 = np.round(r2_score(ground_truth_continuous, val_predictions_continuous),3)
                r_corr = np.round(np.corrcoef(ground_truth_continuous.flatten(), val_predictions_continuous.flatten())[0,1],3)
                xi, yi = np.mgrid[ground_truth_continuous.flatten().min():ground_truth_continuous.flatten().max():nbins*1j, val_predictions_continuous.flatten().min():val_predictions_continuous.flatten().max():nbins*1j]
                plt.figure(figsize=(fig_length,fig_height))
                
                try:
                    k = gaussian_kde([ground_truth_continuous.flatten(),val_predictions_continuous.flatten()])
                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                    
                except Exception:
                    pass
                
                plt.scatter(ground_truth_continuous, val_predictions_continuous,color="black",s=0.1)
                m, b = np.polyfit(ground_truth_continuous.flatten(), val_predictions_continuous.flatten(), 1)
                plt.plot(ground_truth_continuous, m*np.array(ground_truth_continuous) + b,color="red",label = "R2: "+str(r2)+"| r: "+str(r_corr))
                plt.xlabel('Ground Truth')
                plt.ylabel('Model Predictions')
                plt.title("Validation Set | Input(s): "+str(inputs) +"; Target: "+str(target))
                plt.legend(loc="upper right")
                plt.savefig(results_folder+'regression_target_plot_val.png',dpi=150,bbox_inches='tight')
                plt.close()
          
            y_test_argmax = np.argmax(ground_truth,1)
            confusion = confusion_matrix(y_test_argmax, val_predictions_argmax)
            make_confusion_matrix(confusion, figsize=(fig_length, fig_height), cbar=False, title="Validation Set | Input(s): "+str(inputs) +"; Target: "+str(target))
            plt.savefig(results_folder+'confusion_matrix_val.png', dpi=150,bbox_inches='tight')
            plt.close()
            
        """
        Save the best performing model...
        """
        if min(list_of_validation_losses_per_epoch)==list_of_validation_losses_per_epoch[-1]:
            print("Saving new best model")
            torch.save(checkpoint_dict, best_model_file_path)
            best_epoch = training_progress['epoch'].values[-1]
            patience_iterator=0     
            best_val_predictions=val_predictions
            best_val_predictions_continuous=val_predictions_continuous
            best_loss = training_progress['list_of_validation_losses_per_epoch'].values[-1]
            
        if patience_iterator==patience:
            print("Patience stopping criterion reached")
            break
            
        print("")





"""
Post-training results reviewer
"""

print("")
print("Training complete, now compiling final results")

training_progress = pd.DataFrame(np.zeros(shape=((checkpoint_dict['epoch']+1),len(columns))),columns=columns)

for i in training_progress.columns:
    training_progress[i] = checkpoint_dict[i]

training_progress.reset_index(inplace=True)
training_progress.rename(columns={"index": "epoch"},inplace=True)
training_progress.to_csv(results_folder+'training_progress.csv')
print("Saved training progress csv...")

plt.figure(figsize=(fig_length,fig_height))
plt.plot(training_progress['epoch'].values, training_progress['list_of_training_accuracies_per_epoch'].values, label = "Training accuracy %")
plt.plot(training_progress['epoch'].values, training_progress['list_of_validation_accuracies_per_epoch'].values, label = "Validation accuracy %")
plt.vlines(x=best_epoch, ymin=min(training_progress['list_of_validation_accuracies_per_epoch']), ymax=max(training_progress['list_of_validation_accuracies_per_epoch']), colors='purple', ls='--', lw=2, label='Best model at epoch '+str(best_epoch))
plt.xlabel('Epochs')
plt.ylabel('% Accuracy')
plt.title("Input(s): "+str(inputs) +"; Target: "+str(target))
plt.legend(loc="lower right")
plt.savefig(results_folder+'accuracy_plot.png',dpi=600,bbox_inches='tight')
plt.close()
print("Saved accuracy plot...")

plt.figure(figsize=(fig_length,fig_height))
plt.plot(training_progress['epoch'].values, training_progress['list_of_training_losses_per_epoch'].values, label = "Training loss")
plt.plot(training_progress['epoch'].values, training_progress['list_of_validation_losses_per_epoch'].values, label = "Validation loss")
plt.vlines(x=best_epoch, ymin=min(training_progress['list_of_validation_losses_per_epoch']), ymax=max(training_progress['list_of_validation_losses_per_epoch']), colors='purple', ls='--', lw=2, label='Best model at epoch '+str(best_epoch))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Input(s): "+str(inputs) +"; Target: "+str(target))
plt.legend(loc="upper right")
plt.savefig(results_folder+'loss_plot.png',dpi=600,bbox_inches='tight')
plt.close()
print("Saved loss plot...")

best_val_predictions_argmax = np.argmax(best_val_predictions,1)

if continuous_target:
    y_test = ground_truth_continuous
    r2 = np.round(r2_score(y_test, best_val_predictions_continuous),3)
    r_corr = np.round(np.corrcoef(y_test.flatten(), best_val_predictions_continuous.flatten())[0,1],3)
    k = gaussian_kde([y_test.flatten(),best_val_predictions_continuous.flatten()])
    xi, yi = np.mgrid[y_test.flatten().min():y_test.flatten().max():nbins*1j, best_val_predictions_continuous.flatten().min():best_val_predictions_continuous.flatten().max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.figure(figsize=(fig_length,fig_height))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
    plt.scatter(y_test.flatten(), best_val_predictions_continuous.flatten(),color="black",s=0.1)
    m, b = np.polyfit(y_test.flatten(), best_val_predictions_continuous.flatten(), 1)
    plt.plot(y_test, m*np.array(y_test) + b,color="red",label = "R2: "+str(r2)+"| r: "+str(r_corr))
    plt.xlabel('Ground Truth')
    plt.ylabel('Model Predictions')
    plt.title("Validation Set | Input(s): "+str(inputs) +"; Target: "+str(target))
    plt.legend(loc="upper right")
    plt.savefig(results_folder+'regression_target_plot_val.png',dpi=150,bbox_inches='tight')
    plt.close()

y_test_argmax = np.argmax(ground_truth,1)
confusion = confusion_matrix(y_test_argmax, best_val_predictions_argmax)
make_confusion_matrix(confusion, figsize=(fig_length, fig_height), cbar=False, title="Validation Set | Input(s): "+str(inputs) +"; Target: "+str(target))
plt.savefig(results_folder+'confusion_matrix_val.png', dpi=600,bbox_inches='tight')
plt.close()
print("Saved confusion matrix...")

performance_metrics['mode']='Validation'
performance_metrics['target']=target
performance_metrics['inputs']=model_name[len(target)+1:]
performance_metrics['accuracy']=balanced_accuracy_score(y_test_argmax, best_val_predictions_argmax)
performance_metrics['precision'], performance_metrics['recall'], performance_metrics['f1'], support = precision_recall_fscore_support(y_test_argmax, best_val_predictions_argmax, average='macro')
fpr, tpr, _ = roc_curve(ground_truth.ravel(), best_val_predictions.ravel())
roc_auc = auc(fpr, tpr)
performance_metrics['auc']=auc(fpr, tpr)
performance_metrics['training_time_mins']=sum(checkpoint_dict['list_of_validation_times_per_epoch'])+sum(checkpoint_dict['list_of_training_times_per_epoch'])
performance_metrics['loss']=best_loss

if continuous_target:
    performance_metrics['r2']=r2
    performance_metrics['r']=r_corr

plt.figure(figsize=(fig_length, fig_height))
plt.plot(fpr, tpr, color='darkorange',label='ROC (Microaverage AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Validation Set | Input(s): "+str(inputs) +"; Target: "+str(target))
plt.legend(loc="lower right")
plt.savefig(results_folder+'microaverage_roc_val.png',dpi=600,bbox_inches='tight')
plt.close()
print("Saved roc...")

print("")
print("Training and validation complete!")
print("")





"""
Start testing
"""

print("Testing on: " +str(best_model_file_path))
checkpoint_dict = torch.load(best_model_file_path)

mlp_connectivity.load_state_dict(checkpoint_dict['mlp_connectivity_state_dict'])
mlp_metadata.load_state_dict(checkpoint_dict['mlp_metadata_state_dict'])
cnn.load_state_dict(checkpoint_dict['cnn_state_dict'])
mlp_logits.load_state_dict(checkpoint_dict['mlp_logits_state_dict'])

with torch.no_grad():
    input_dictionary = {'mode_label': 'Testing',
                        'training': False,
                        'epoch': epoch,
                        'optimizer': optimizer,
                        'writer': writer,
                        'mlp_connectivity': mlp_connectivity,
                        'mlp_metadata': mlp_metadata,
                        'cnn': cnn,
                        'mlp_logits': mlp_logits,
                        'params': params}

    output_dictionary_test = one_epoch(input_dictionary)

print("Average test loss over epoch " + str(epoch) + ": " + str(output_dictionary_test['loss_over_whole_epoch']))
print("Average test accuracy over epoch " + str(epoch) + ": " + str(output_dictionary_test['accuracy_over_whole_epoch']))
print("Epoch test time (mins): " + str(round(output_dictionary_test['epoch_time'],3)))

output_dictionary_test['epoch']= checkpoint_dict['epoch']
torch.save(output_dictionary_test, results_folder+'model_testing.pt')

test_predictions = output_dictionary_test['predictions']
test_predictions_continuous=output_dictionary_test['predictions_continuous']
ground_truth = output_dictionary_test['ground_truth']
ground_truth_continuous = output_dictionary_test['ground_truth_continuous']

test_predictions_argmax = np.argmax(test_predictions,1)

if continuous_target:
    r2 = np.round(r2_score(ground_truth_continuous, test_predictions_continuous),3)
    r_corr = np.round(np.corrcoef(ground_truth_continuous.flatten(), test_predictions_continuous.flatten())[0,1],3)
    xi, yi = np.mgrid[ground_truth_continuous.flatten().min():ground_truth_continuous.flatten().max():nbins*1j, test_predictions_continuous.flatten().min():test_predictions_continuous.flatten().max():nbins*1j]
    plt.figure(figsize=(fig_length,fig_height))

    try:
        k = gaussian_kde([ground_truth_continuous.flatten(),test_predictions_continuous.flatten()])
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

    except Exception:
        pass

    plt.scatter(ground_truth_continuous, test_predictions_continuous,color="black",s=0.1)
    m, b = np.polyfit(ground_truth_continuous.flatten(), test_predictions_continuous.flatten(), 1)
    plt.plot(ground_truth_continuous, m*np.array(ground_truth_continuous) + b,color="red",label = "R2: "+str(r2)+"| r: "+str(r_corr))
    plt.xlabel('Ground Truth')
    plt.ylabel('Model Predictions')
    plt.title("Test Set | Input(s): "+str(inputs) +"; Target: "+str(target))
    plt.legend(loc="upper right")
    plt.savefig(results_folder+'regression_target_plot_test.png',dpi=150,bbox_inches='tight')
    plt.close()

y_test_argmax = np.argmax(ground_truth,1)
confusion = confusion_matrix(y_test_argmax, test_predictions_argmax)
make_confusion_matrix(confusion, figsize=(fig_length, fig_height), cbar=False, title="Test Set | Input(s): "+str(inputs) +"; Target: "+str(target))
plt.savefig(results_folder+'confusion_matrix_test.png', dpi=150,bbox_inches='tight')
plt.close()

acc = balanced_accuracy_score(y_test_argmax, test_predictions_argmax)
prec, rec, f1, sup = precision_recall_fscore_support(y_test_argmax, test_predictions_argmax, average='macro')
fpr, tpr, _ = roc_curve(ground_truth.ravel(), test_predictions.ravel())
roc_auc = auc(fpr, tpr)
time = output_dictionary_test['epoch_time']
test_loss = output_dictionary_test['loss_over_whole_epoch']

plt.figure(figsize=(fig_length, fig_height))
plt.plot(fpr, tpr, color='darkorange',label='ROC (Microaverage AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Test Set | Input(s): "+str(inputs) +"; Target: "+str(target))
plt.legend(loc="lower right")
plt.savefig(results_folder+'microaverage_roc_test.png',dpi=600,bbox_inches='tight')
plt.close()
print("Saved roc...")


test_results = {'mode': 'Test', 'target': target, 'inputs': model_name[len(target)+1:],
                            'accuracy':acc, 'precision':prec,'recall':rec,'f1':f1,'auc':roc_auc,
                             'training_time_mins':time,'loss':test_loss,'r2':0,'r':0}

if continuous_target:
    test_results['r2']=r2
    test_results['r']=r_corr
    
test_results = pd.DataFrame({k:[v] for k,v in test_results.items()})

performance_metrics = pd.concat([performance_metrics,test_results], axis=0)
performance_metrics.to_csv(results_folder+'performance_metrics_'+str(model_name)+'.csv')
print("Saved metric csv...")
    
print("")

