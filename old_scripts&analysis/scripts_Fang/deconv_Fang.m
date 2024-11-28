%% Calculate dt and spike rate across datasets

cd('Autocalibrated-spike-inference/GT_autocalibration')

GT_folders = {'DS09-GCaMP6f-m-V1',...
    'DS10-GCaMP6f-m-V1-neuropil-corrected',...
    'DS11-GCaMP6f-m-V1-neuropil-corrected',...
    'DS13-GCaMP6s-m-V1-neuropil-corrected',...
    'DS14-GCaMP6s-m-V1',...
    'DS29-GCaMP7f-m-V1','DS30-GCaMP8f-m-V1','DS31-GCaMP8m-m-V1','DS32-GCaMP8s-m-V1',...
    'DS06-GCaMP6f-zf-aDp','DS07-GCaMP6f-zf-dD','DS08-GCaMP6f-zf-OB'};

% dt = delta_t = 1/(frame rate)
dt_all = zeros(numel(GT_folders),1);

clear spike_rate_GT_all

% Go through all GT datasets
for folder_index =  1:numel(GT_folders)
    
    cd(GT_folders{folder_index})
    
    % Find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    dtX = [];
    spike_rate_GTX = [];

    for neuron_index = 1:numel(neuron_files)

        load(neuron_files(neuron_index).name)

        for index = 1:numel(CAttached)
            
            fluo_time = CAttached{index}.fluo_time; % Time points of the fluorescence trace
            fluo_trace = CAttached{index}.fluo_mean; % Fluorescence trace (in dF/F)
            AP_times = CAttached{index}.events_AP / 1e4; % Action potential times (in seconds)
            
            % compute spike rate for the current neuron
            spike_rate_GT = numel(AP_times)/(max(fluo_time)-min(fluo_time));
            % concatenate current neuron's spike rate to variable "spike_rate_GTX"
            spike_rate_GTX = [spike_rate_GTX;numel(AP_times)/(max(fluo_time)-min(fluo_time))];

            % compute dt from the variable fluo_time (median difference between time points)
            dt = nanmedian(diff(fluo_time));
            dtX = [dtX;dt];
        end
    end

    % Average across neurons for each GT dataset 
    dt_all(folder_index) = nanmedian(dtX);
    spike_rate_GT_all(folder_index) = nanmedian(spike_rate_GTX);

    % Plot sorted distribution of spike rates
    %  %figure(411), plot((1:numel(spike_rate_GTX))/numel(spike_rate_GTX),sort(spike_rate_GTX)); hold on;
    
    % Print summary statistics for the current GT dataset
    disp(['For dataset ',GT_folders{folder_index},', mean spike rate: ',num2str(spike_rate_GT_all(folder_index)),...
        '; average framerate: ',num2str(1/dt_all(folder_index))])

    cd ..


end


%% Extract average spike-evoked kernels for all datasets and neurons

% Clear the variable that will lateron contain all kernels
clear kernel_averaged_all

% Go through all GT datasets
for folder_index = 1:numel(GT_folders)
    
    cd(GT_folders{folder_index})
    
    % Find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    % Initialize matrix that will lateron contain all kernels for this dataset
    kernel_averaged = [];

    % Go through all neurons of this GT dataset
    for neuron_index = 1:numel(neuron_files)

        load(neuron_files(neuron_index).name)

        kernelX_all = [];

        % Go through all recordings done from the currently analyzed neuron
        for index = 1:numel(CAttached)
            
            % This part is necessary because "fluo_time" is sometimes a
            % column vector and sometimes a row vector, depending on the dataset
            if size(CAttached{index}.fluo_time,2) > 1
                CAttached{index}.fluo_time = CAttached{index}.fluo_time';
            end
            if size(CAttached{index}.fluo_mean,2) > 1
                CAttached{index}.fluo_mean = CAttached{index}.fluo_mean';
            end
            
            fluo_time = CAttached{index}.fluo_time; % Time points of the fluorescence trace
            fluo_trace = CAttached{index}.fluo_mean; % Fluorescence trace (in dF/F)
            AP_times = CAttached{index}.events_AP / 1e4; % Action potential times (in seconds)
            
            % find non-NaN values
            good_indices = find(~isnan(CAttached{index}.fluo_time).*~isnan(CAttached{index}.fluo_mean));
            
            fluo_time = fluo_time(good_indices);
            fluo_trace = fluo_trace(good_indices);
            CAttached{index}.fluo_time = CAttached{index}.fluo_time(good_indices);
            CAttached{index}.fluo_mean = CAttached{index}.fluo_mean(good_indices);

            spikes = AP_times;
            
            % compute dt (time between two imaging frames)
            dt = nanmedian(diff(fluo_time));
            % get (from the script above) the target dt0 for this dataset
            dt0 = dt_all(folder_index);

            % resample the recording to the target frame rate 1/dt0 if
            % necessary; this is important if frame rates vary between
            % recordings of a single dataset
           if abs(dt - dt0)/dt0 > 0.05
                CAttached{index}.fluo_mean_resampled = resample(double(fluo_trace),round(1/dt0*100),round(1/dt*100));
                CAttached{index}.fluo_time_resampled = (dt0:dt0:dt0*numel(CAttached{index}.fluo_mean_resampled)) ;
           else
                CAttached{index}.fluo_mean_resampled = double(fluo_trace);
                CAttached{index}.fluo_time_resampled = CAttached{index}.fluo_time+dt0-CAttached{index}.fluo_time(1);
            end
            
            % The next steps prepare the variables "spike_density" and
            % "fluorescence"; deconvolution is used to obtain the response
            % kernel from those two signals

            % allocate all spikes into bins; the bins are indicated by the
            % frame times, "CAttached{index}.fluo_time_resampled"; 
            spike_density = hist(spikes(spikes<(max(CAttached{index}.fluo_time_resampled))),CAttached{index}.fluo_time_resampled);
            fluorescence = CAttached{index}.fluo_mean_resampled;
            
            good_indices = find(~isnan(fluorescence));
            spike_density = spike_density(good_indices);
            fluorescence = fluorescence(good_indices);
            
            % perform deconvolution to obtain response kernel
            try
%                 kernel = fftshift( ifft(fft(DF)./fft(spike_density)));
                kernel_lucy = deconvreg(fluorescence,spike_density);
            catch
%                 kernel = fftshift( ifft(fft(DF)./fft(HD')));
                kernel_lucy = deconvreg(fluorescence,spike_density');
            end
            
            % get center point of the resulting kernel
            center = round(numel(kernel_lucy)/2);
            
            % extract only the central part of the kernel
            window_extent = 4; % in seconds
            relevant_kernel_excerpt = kernel_lucy(round(center-1/dt0*window_extent):round(center+1/dt0*window_extent));
            
            % concatenate all extracted kernels
            kernelX_all = [kernelX_all, relevant_kernel_excerpt];
        end

        % Treat some special cases ...

        if 1%  (folder_index > 1 && folder_index < 5)

            if size(kernelX_all,1) > 1 && size(kernelX_all,2) > 1
                kernel_averaged = [kernel_averaged,squeeze(nanmean(kernelX_all,2))];
            else
                kernel_averaged = [kernel_averaged,squeeze(kernelX_all)];
            end
        else

            if size(kernelX_all,1) > 1 && size(kernelX_all,2) > 1
                kernel_averaged = [kernel_averaged,squeeze(nanmean(kernelX_all,1))'];
            else
                kernel_averaged = [kernel_averaged,squeeze(kernelX_all)'];
            end

        end



    end
    
    % write the extracted kernels for each dataset into the pooling
    % variable "kernel_averaged_all"
    kernel_averaged_all{folder_index} = kernel_averaged;


    cd ..

end


%% Calculate amplitudes for each neuron across all datasets

% initialize cell array to store amplitudes and baselines
neuron_amplitudes = cell(1, numel(GT_folders));
neuron_baselines = cell(1, numel(GT_folders)); 

% go through all GT datasets
for folder_index = 1:numel(GT_folders)
    
    % get kernels for current dataset
    dataset_kernels = kernel_averaged_all{folder_index};
    
    % initialize array to store amplitudes and baselines
    amplitudes = zeros(1, size(dataset_kernels, 2));
    baselines = zeros(1, size(dataset_kernels, 2)); 

    % go through all neurons of current dataset
    for neuron_index = 1:size(dataset_kernels, 2)

        % get the kernel for current neuron
        neuron_kernel = dataset_kernels(:, neuron_index);
        
        % calculate baseline from kernels
        baseline = nanmedian(neuron_kernel);
        baselines(neuron_index) = baseline; 
        
        % calculate amplitude
        amplitude = max(neuron_kernel) - baseline;
        amplitudes(neuron_index) = amplitude;

    end

    % store amplitudes for current dataset
    neuron_amplitudes{folder_index} = amplitudes;
    % store baselines for current dataset
    neuron_baselines{folder_index} = baselines; 
end


%% Visualizations


% simplified dataset names
datasets = {'GC6f','GC6f_c','GC6f_c','GC6s_c','GC6s','GC7f','GC8f','GC8m','GC8s','GC6f_zf','GC6f_zf','GC6f_zf'};

% define colors
colors = {'k','k','k','c','c','m','r','b','g','k','k','k'}; 


% Line graph
figure;
hold on;
for i = 1:numel(neuron_amplitudes)
    amplitudes_to_plot = sort(neuron_amplitudes{i}(neuron_amplitudes{i} <= 5)); 
    plot(1:numel(amplitudes_to_plot), amplitudes_to_plot, 'DisplayName', datasets{i}, 'Color', colors{i});
end
hold off;
title('Neuron Amplitudes Across Datasets');
xlabel('Neuron Index');
ylabel('Amplitude (dF/F)');
legend('show', 'Location', 'eastoutside');


% Histogram
figure;
hold on;
for i = 1:numel(neuron_amplitudes)
    histogram(neuron_amplitudes{i}(neuron_amplitudes{i} < 5), 'DisplayName', datasets{i}, 'BinWidth', 0.05,...
        'Normalization', 'probability', 'FaceColor', colors{i});
end
hold off;
title('Distribution of Neuron Amplitudes');
xlabel('Amplitude (dF/F)');
ylabel('Probability');
legend('show', 'Location', 'eastoutside');


% Box plot
figure;
hold on;
for i = 1:numel(neuron_amplitudes)
    amplitudes_to_plot = neuron_amplitudes{i}(neuron_amplitudes{i} <= 5);
    boxplot(amplitudes_to_plot, 'positions', i, 'colors', colors{i}); 
end
hold off;
set(gca, 'xtick', 1:numel(datasets), 'xticklabel', datasets);
xtickangle(45);


%% Find optimal amplitudes for each calcium indicator


% dataset names
datasets = {'GC6f','GC6f_c','GC6f_c','GC6s_c','GC6s','GC7f','GC8f','GC8m','GC8s','GC6f_zf','GC6f_zf','GC6f_zf'};

% initialize arrays to store relevant metrics
mean_amplitudes = zeros(1, numel(GT_folders));
median_amplitudes = zeros(1, numel(GT_folders));
snr = zeros(1, numel(GT_folders));
rise_times = zeros(1, numel(GT_folders));
decay_times = zeros(1, numel(GT_folders));

% calculate metrics for each dataset
for folder_index = 1:numel(GT_folders)
    amplitudes = neuron_amplitudes{folder_index};
    baselines = neuron_baselines{folder_index};
    kernels = kernel_averaged_all{folder_index};
    
    mean_amplitudes(folder_index) = mean(amplitudes);
    median_amplitudes(folder_index) = median(amplitudes);
    
    % calculate SNR
    snr(folder_index) = mean(amplitudes) / std(baselines);
    
    % calculate average rise and decay times
    avg_kernel = mean(kernels, 2);
    [~, peak_index] = max(avg_kernel);
    half_max = max(avg_kernel) / 2;
    
    rise_time = find(avg_kernel(1:peak_index) > half_max, 1) * dt_all(folder_index);
    decay_time = find(avg_kernel(peak_index:end) < half_max, 1) * dt_all(folder_index);
    
    rise_times(folder_index) = rise_time;
    decay_times(folder_index) = decay_time;
end


% create a table with the results
results_table = table(datasets', mean_amplitudes', median_amplitudes', snr', rise_times', decay_times', ...
    'VariableNames', {'Dataset', 'MeanAmplitude', 'MedianAmplitude', 'SNR', 'RiseTime', 'DecayTime'});
