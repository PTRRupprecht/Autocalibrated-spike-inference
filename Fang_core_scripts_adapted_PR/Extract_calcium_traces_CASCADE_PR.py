#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to extract ground truth + calcium traces and process them with CASCADE
"""


import os
import sys
from pathlib import Path
import glob
import silence_tensorflow.auto
import numpy as np
import scipy.io as sio
from ruamel.yaml import YAML


systematic_bias_divisions = np.round(np.logspace(np.log10(0.2),np.log10(5),20)*100)/100

for systematic_bias_division in systematic_bias_divisions:

    # define paths (change acrrording to your own path)
    CASCADE_DIR = Path(r'C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade')
    MODELS_DIR = CASCADE_DIR / 'Pretrained_models'
    DATA_DIR = Path(r'C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Autocalibration\Autocalibrated-spike-inference\GT_autocalibration')
    temp_string = 'Results_for_autocalibration_d'+str(systematic_bias_division)
    RESULTS_DIR = CASCADE_DIR / temp_string
    
    # set up CASCADE environment
    os.environ['CASCADE_PATH'] = str(CASCADE_DIR)
    os.environ['CASCADE_MODELS_PATH'] = str(MODELS_DIR)
    
    # create result directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # add CASCADE to Python path
    if str(CASCADE_DIR) not in sys.path:
        sys.path.append(str(CASCADE_DIR))
    
    # import CASCADE modules
    from cascade2p import checks, cascade, config, utils
    from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
    
    # Perform package checks
    print('\nChecks for packages:')
    checks.check_packages()
    
    
    """
    Get list of available models
    """
    
    # point to the correct models directory
    yaml = YAML(typ='rt')
    
    # download the model list
    os.chdir(CASCADE_DIR)
    cascade.download_model('update_models', verbose=1)
    
    # download the specific model
    
    
    
    
    """
    Extract ground truth for testing with resampling
    """
    
    all_datasets = ['DS30-GCaMP8f-m-V1','DS31-GCaMP8m-m-V1','DS32-GCaMP8s-m-V1']
    
    model_name = ['GC8f_EXC_30Hz_smoothing50ms_high_noise','GC8m_EXC_30Hz_smoothing50ms_high_noise','GC8s_EXC_30Hz_smoothing50ms_high_noise']
    
    # analysis parameters (apply low noise level may cannot extract traces, apply high noise level may make spikes too smooth)
    framerate = 30
    noise_level = 1.5
    before_frac = 0.5
    windowsize = 64
    smoothing = 0.05
    
    
    # process datasets
    for ix,dataset in enumerate(all_datasets):
        
        os.chdir(CASCADE_DIR)
        model_path = MODELS_DIR / model_name[ix]
        cascade.download_model(model_name[ix])
        
        
        dataset_path = DATA_DIR / dataset
        print(f"\nLooking for dataset in: {dataset_path.absolute()}")
        
        if not dataset_path.exists():
            print(f"WARNING: Dataset directory not found: {dataset_path}")
            continue
        
        dataset_of_interest = [str(dataset_path)]
        neuron_files = glob.glob(str(dataset_path / 'CAttached*mini.mat'))
        print(f"Found {len(neuron_files)} neuron files")
        
        for jkl, neuron_file in enumerate(neuron_files):
            print(f"\nProcessing neuron file {jkl + 1}/{len(neuron_files)}: {Path(neuron_file).name}")
            omission_list = np.arange(len(neuron_files))
            omission_list = np.delete(omission_list, np.where(omission_list == jkl))
    
            try:
                calcium, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
                    ground_truth_folders=dataset_of_interest,
                    before_frac=before_frac,
                    windowsize=windowsize,
                    after_frac=1 - before_frac,
                    noise_level=noise_level,
                    sampling_rate=framerate,
                    smoothing=smoothing * framerate,
                    omission_list=omission_list,
                    permute=0,
                    verbose=1,
                    replicas=0)
                
                window_position = int(windowsize/2-0)
                calcium = calcium[:,window_position,]/systematic_bias_division
                ground_truth = ground_truth[:,]
                
                if len(calcium) > 0:
                    print(f"{dataset}. Neuron #{jkl+1}. Number of timepoints extracted for this neuron: {len(calcium)}")
                    
                    # Use full path for model prediction
                    spike_rates_GC8 = cascade.predict(
                        str(model_path),
                        calcium.T,
                        verbosity=1
                    )
                    spike_rates_GC8 = np.squeeze(spike_rates_GC8)
                    
                    # Save results
                    results_dataset_dir = RESULTS_DIR / dataset
                    results_dataset_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = results_dataset_dir / Path(neuron_file).name
                    print(f"Saving results to: {filename}")
                    sio.savemat(str(filename), {
                        'spike_rates_GC8': spike_rates_GC8,
                        'calcium': calcium,
                        'ground_truth': ground_truth
                    })
                    print(f"Successfully saved results for neuron {jkl+1}")
                else:
                    print(f"{dataset}. Neuron #{jkl+1}. No timepoints extracted for this neuron at this noise level.")
                    
            except Exception as e:
                print(f"Error processing {neuron_file}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
    
    print("\nProcessing complete!")



