#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Demo script to extract ground truth + calcium traces and process them with CASCADE


"""



"""

Import python packages

"""


import os, sys, glob
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import silence_tensorflow.auto

import shutil, copy

os.chdir(r'C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade')

if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current directory: {}'.format( os.getcwd() ))

import keras
from copy import deepcopy
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml

# perform checks to catch most likly import errors
from cascade2p import checks    # TODO: put all of this in one function
print('\nChecks for packages:')
checks.check_packages()

from cascade2p import cascade
from cascade2p import config
from cascade2p import utils
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth

from importlib import reload

reload(cascade)


"""

Get list of available models

"""

cascade.download_model( 'update_models',verbose = 1)

yaml_file = open('Pretrained_models/available_models.yaml')
X = yaml.load(yaml_file, Loader=yaml.Loader)
list_of_models = list(X.keys())

# for model in list_of_models:
#   print(model)


model_name = 'GC8s_EXC_30Hz_smoothing50ms_high_noise'

cascade.download_model(model_name)



"""

Extract ground truth for testing with resampling

"""

all_datasets = ['DS30-GCaMP8f-m-V1',
 'DS31-GCaMP8m-m-V1',
 'DS32-GCaMP8s-m-V1']

framerate = 30
noise_level = 1.5 # 0.5 is very low noise level
before_frac = 0.5
windowsize = 64
smoothing = 0.05

for dataset in all_datasets:
    
    dataset_of_interest = [os.path.join('Ground_truth', dataset) ]    
    neuron_files = glob.glob(os.path.join('Ground_truth',dataset,'CAttached*mini.mat'))

    
    for jkl,neuron_file in enumerate(neuron_files):
        
        omission_list = np.arange(len(neuron_files))
        
        omission_list = np.delete(omission_list,np.where(omission_list == jkl))

        # test model with the one remaining test_dataset
        calcium, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
                                    ground_truth_folders = dataset_of_interest,
                                    before_frac = before_frac,
                                    windowsize = windowsize,
                                    after_frac = 1 - before_frac,
                                    noise_level = noise_level,
                                    sampling_rate = framerate,
                                    smoothing = smoothing * framerate,
                                    omission_list = omission_list,
                                    permute = 0,
                                    verbose = 0,
                                    replicas = 0)
        
        
        window_position = int(windowsize/2-0)

        calcium = calcium[:,window_position,]
        ground_truth = ground_truth[:,]
        
        if len(calcium) > 0:
            print(dataset+'. Neuron #'+str(jkl+1)+'. Number of timepoints extracted for this neuron: '+str(len(calcium)))
       
        
            spike_rates_GC8 = cascade.predict( model_name, calcium.T, verbosity=0 )
            spike_rates_GC8 = np.squeeze(spike_rates_GC8)
            
            
            if not os.path.isdir('Results_for_autocalibration\\'+os.path.basename(dataset_of_interest[0])):
                os.makedirs('Results_for_autocalibration\\'+os.path.basename(dataset_of_interest[0]))
                
            filename = 'Results_for_autocalibration\\'+os.path.basename(dataset_of_interest[0])+'\\'+os.path.basename(neuron_files[jkl])
        
            sio.savemat(filename,{'spike_rates_GC8':spike_rates_GC8,'calcium':calcium,'ground_truth':ground_truth})
                        
        else:
             print(dataset+'. Neuron #'+str(jkl+1)+'. No timepoints extracted for this neuron at this noise level.')
                

