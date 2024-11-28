%% Deconvolution for a single neuron


cd('Autocalibrated-spike-inference/GT_autocalibration')

% set up the dataset to be analyzed
dataset_folder = 'DS32-GCaMP8s-m-V1';
dataset_name = 'GCaMP8s';

cd(dataset_folder)

% load neuron files
neuron_files = dir('CAttached*.mat');

% select which neuron to analyze
selected_neuron = 10;

% calculate dt for the selected neuron
load(neuron_files(selected_neuron).name)
dtX = [];

for index = 1:numel(CAttached)
    fluo_time = CAttached{index}.fluo_time;
    dt = nanmedian(diff(fluo_time));
    dtX = [dtX; dt];
end

% median dt for this neuron
dt0 = nanmedian(dtX);

% initialize arrays for analysis
deconv_amp = [];
kernels_all = [];
rise_times = [];
decay_times = [];

% analyze all recordings from the selected neuron
load(neuron_files(selected_neuron).name)

% go through all recordings for this neuron
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
    
    % store values
    deconv_amp = [deconv_amp; amplitude];
    rise_times = [rise_times; rise_time];
    decay_times = [decay_times; decay_time];
    kernels_all = [kernels_all, relevant_kernel_excerpt];
end

% calculate average kernel for this neuron
if size(kernels_all,1) > 1 && size(kernels_all,2) > 1
    avg_kernel = squeeze(nanmean(kernels_all,2));
else
    avg_kernel = squeeze(kernels_all);
end

% calculate metrics from averaged kernel
neuron_baseline = nanmedian(avg_kernel);
neuron_amplitude = max(avg_kernel) - neuron_baseline;
snr = neuron_amplitude / std(neuron_baseline);

%% Visualizations and summary statistics

% plot fluorescence trace and spike times for first recording
figure;
subplot(2,1,1);
plot(CAttached{1}.fluo_time, CAttached{1}.fluo_mean, 'b');
hold on;
AP_times = CAttached{1}.events_AP / 1e4;
ylims = ylim;
plot([AP_times AP_times]', repmat(ylims', 1, length(AP_times)), 'r--');
title(['Neuron ' num2str(selected_neuron) ' - Recording 1']);
xlabel('Time (s)');
ylabel('Fluorescence (dF/F)');
legend('Fluorescence', 'Action Potentials');

% Box plot
subplot(2,1,2);
boxplot(deconv_amp);
title(sprintf('Amplitude Box Plot for Neuron %d', selected_neuron));
ylabel('Amplitude (dF/F)');

% Display results
fprintf('\nResults for Neuron %d:\n', selected_neuron);
fprintf('Mean Amplitude: %.2f\n', mean(deconv_amp));
fprintf('Median Amplitude: %.2f\n', median(deconv_amp));
fprintf('Standard Deviation: %.2f\n', std(deconv_amp));
fprintf('SNR: %.2f\n', snr);
fprintf('Average Rise Time: %.2f\n', nanmean(rise_times));
fprintf('Average Decay Time: %.2f\n', nanmean(decay_times));
fprintf('Number of recordings analyzed: %d\n', numel(CAttached));

