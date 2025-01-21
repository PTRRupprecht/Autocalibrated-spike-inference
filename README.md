# Autocalibrated spike inference
 Development towards an algorithm for spike inference with autocalibration.
 
 Our hybrid GMM can accurately detect single spikes in datasets recorded using the recently published GCaMP8s/m calcium indicators.
 Subsequently, it can improve single spike inference from calcium imaging data using CASCADE.

 This repository consists of:

 - Calcium imaging data with simultaneously electrophysiological recording from 264 neurons
 - The python script named 'Spike_inference_CASCADE' can be used to perform spike inference after downloading CASCADE from https://github.com/HelmchenLabSoftware/Cascade?tab=readme-ov-file
 - Deconvolution scripts for calculating amplitudes, rise times, decay times based on extracting kernels
 - Gradient descent modeling scripts for simulating the calcium ΔF/F data and extracting amplitudes
 - The 'Hybrid_GMM' script for spike detection and auto-calibration
   
