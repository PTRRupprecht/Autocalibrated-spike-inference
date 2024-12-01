%% Deconvolution


cd('Autocalibrated-spike-inference/GT_autocalibration')


% preparations
dataset_folder = 'DS31-GCaMP8m-m-V1';
dataset_name = 'GCaMP8m';
cd(dataset_folder)

% calculate dt for the dataset
neuron_files = dir('CAttached*.mat');
dtX = [];

% go through all neurons to calculate dt
for neuron_index = 1:numel(neuron_files)
    load(neuron_files(neuron_index).name)
    
    for index = 1:numel(CAttached)
        fluo_time = CAttached{index}.fluo_time;
        dt = nanmedian(diff(fluo_time));
        dtX = [dtX; dt];
    end
end

% median dt across all neurons
dt0 = nanmedian(dtX);


% Extract average spike-evoked kernels and deconv_amp

% initialize arrays
deconv_amp = [];
neuron_deconv_amp = [];
rise_times = [];
decay_times = [];
kernel_averaged = [];

% go through all neurons
for neuron_index = 1:numel(neuron_files)
    load(neuron_files(neuron_index).name)
    kernels_all = [];
    
    % go through all recordings from the currently analyzed neuron
    for index = 1:numel(CAttached)

        % standardize vector orientation
        if size(CAttached{index}.fluo_time,2) > 1
            CAttached{index}.fluo_time = CAttached{index}.fluo_time';
        end

        if size(CAttached{index}.fluo_mean,2) > 1
            CAttached{index}.fluo_mean = CAttached{index}.fluo_mean';
        end
        
        fluo_time = CAttached{index}.fluo_time;
        fluo_trace = CAttached{index}.fluo_mean;
        AP_times = CAttached{index}.events_AP / 1e4;
        
        % find non-NaN values
        good_indices = find(~isnan(fluo_time).*~isnan(fluo_trace));
        fluo_time = fluo_time(good_indices);
        fluo_trace = fluo_trace(good_indices);
        
        % compute dt for current recording
        dt = nanmedian(diff(fluo_time));
        
        % resample if necessary to match target dt0
        if abs(dt - dt0)/dt0 > 0.05
            fluo_mean_resampled = resample(double(fluo_trace), round(1/dt0*100), round(1/dt*100));
            fluo_time_resampled = (dt0:dt0:dt0*numel(fluo_mean_resampled));
        else
            fluo_mean_resampled = double(fluo_trace);
            fluo_time_resampled = fluo_time + dt0 - fluo_time(1);
        end
        
        % prepare spike density and fluorescence for deconvolution
        spike_density = hist(AP_times(AP_times<(max(fluo_time_resampled))), fluo_time_resampled);
        fluorescence = fluo_mean_resampled;
        
        good_indices = find(~isnan(fluorescence));
        spike_density = spike_density(good_indices);
        fluorescence = fluorescence(good_indices);
        
        % perform deconvolution
        try
            kernel_lucy = deconvreg(fluorescence, spike_density);
        catch
            kernel_lucy = deconvreg(fluorescence, spike_density');
        end
        
        % extract central part of kernel
        center = round(numel(kernel_lucy)/2);
        window_extent = 4; % in seconds
        relevant_kernel_excerpt = kernel_lucy(round(center-1/dt0*window_extent):round(center+1/dt0*window_extent));
        
        % calculate metrics for this recording
        baseline = nanmedian(relevant_kernel_excerpt);
        amplitude = max(relevant_kernel_excerpt) - baseline;
        
        % calculate rise and decay times
        [~, peak_index] = max(relevant_kernel_excerpt);
        half_max = (max(relevant_kernel_excerpt) + baseline) / 2;
        
        try
            rise_time = find(relevant_kernel_excerpt(1:peak_index) > half_max, 1) * dt0;
            decay_time = find(relevant_kernel_excerpt(peak_index:end) < half_max, 1) * dt0;
        catch
            rise_time = NaN;
            decay_time = NaN;
        end
        
        % store values for individual recordings
        deconv_amp = [deconv_amp; amplitude];
        rise_times = [rise_times; rise_time];
        decay_times = [decay_times; decay_time];
        
        % store kernel for averaging
        kernels_all = [kernels_all, relevant_kernel_excerpt];
    end
    
    % average kernels for this neuron
    if size(kernels_all,1) > 1 && size(kernels_all,2) > 1
        avg_kernel = squeeze(nanmean(kernels_all,2));
    else
        avg_kernel = squeeze(kernels_all);
    end

    kernel_averaged = [kernel_averaged, avg_kernel];
    
    % calculate neuron deconv_amp from averaged kernel
    neuron_baseline = nanmedian(avg_kernel);
    neuron_amplitude = max(avg_kernel) - neuron_baseline;
    neuron_deconv_amp = [neuron_deconv_amp, neuron_amplitude];
    
end



%% Visualizations and summary statistics


% Histogram
figure;
histogram(deconv_amp(deconv_amp < 5), 'BinWidth', 0.05, ...
    'Normalization', 'probability', 'FaceColor', 'b');
title([dataset_name, ' Distribution of Neuron deconv_amp']);
xlabel('Amplitude (dF/F)');
ylabel('Probability');


% Box plot
figure;
boxplot(deconv_amp(deconv_amp <= 5), 'Labels', {dataset_name});
ylabel('Amplitude (dF/F)');
title([dataset_name, ' Amplitude Distribution']);



% summary statistics

% individual recordings
mean_amplitude = mean(deconv_amp);
median_amplitude = median(deconv_amp);
std_amplitude = std(deconv_amp);
tau_rise = nanmedian(rise_times);
tau_decay = nanmedian(decay_times);

% display results
fprintf('\nResults for %s:\n', dataset_name);
fprintf('Mean Amplitude: %.2f\n', mean_amplitude);
fprintf('Median Amplitude: %.2f\n', median_amplitude);
fprintf('Standard Deviation: %.2f\n', std_amplitude);
fprintf('Risetime: %.2f\n', tau_rise);
fprintf('Decaytime: %.2f\n', tau_decay);

