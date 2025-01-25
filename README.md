# Autocalibrated spike inference
 Development towards an algorithm for spike inference with autocalibration.
 
 Our hybrid GMM can accurately detect single spikes in datasets recorded using the recently published GCaMP8s/m calcium indicators.
 Subsequently, it can improve single spike inference from calcium imaging data using CASCADE.

 This repository consists of:

 - Calcium imaging data with simultaneous electrophysiological recordings from 264 neurons, copied from the CASCADE repository https://github.com/HelmchenLabSoftware/Cascade
 - The python script named 'Spike_inference_CASCADE' can be used to perform spike inference after downloading CASCADE from the CASCADE repository https://github.com/HelmchenLabSoftware/Cascade
 - Deconvolution scripts for calculating amplitudes, rise times, decay times based on extracting kernels
 - Gradient descent modeling scripts for simulating the calcium ΔF/F data and extracting amplitudes
 - The 'Hybrid_GMM' script for spike detection and auto-calibration


 This repository was created in the context of Xusheng Fang's (Felix-bangbang) master's thesis, under the supervision of Dr. Peter Rupprecht (PTRRupprecht). 
 
 Thesis Title: Auto-calibration Machine Learning Algorithms for Accurate Spike Detection from Calcium Imaging Data.
