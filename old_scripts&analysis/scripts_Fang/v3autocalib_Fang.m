%% Autocalibration for all datasets


cd('Autocalibrated-spike-inference/GT_autocalibration')


% list of folders
GT_folders = {'DS09-GCaMP6f-m-V1', 'DS10-GCaMP6f-m-V1-neuropil-corrected',...
'DS11-GCaMP6f-m-V1-neuropil-corrected', 'DS13-GCaMP6s-m-V1-neuropil-corrected',...
'DS14-GCaMP6s-m-V1', 'DS29-GCaMP7f-m-V1',...
'DS30-GCaMP8f-m-V1', 'DS31-GCaMP8m-m-V1', 'DS32-GCaMP8s-m-V1',...
'DS06-GCaMP6f-zf-aDp','DS07-GCaMP6f-zf-dD', 'DS08-GCaMP6f-zf-OB'};


% simplify names
datasets = {'GC6f','GC6f_c','GC6f_c','GC6s_c','GC6s','GC7f','GC8f','GC8m','GC8s','GC6f_zf','GC6f_zf','GC6f_zf'};


% Parameters (arbitrary)
threshold = 0.05;
smoothing_value = 5;
duration_threshold = 3;
offset_time = 3;


% initialize result arrays for all datasets
median_amplitude_changes = cell(1, numel(datasets));
amplitude_changes = cell(1, numel(datasets));
amplitudes = cell(1, numel(datasets));
baselines = cell(1, numel(datasets));
delta_f_f0 = cell(1, numel(datasets));


% create a large figure to hold all subplots
figure;
total_plots = length(datasets) * 4;  % 4 subplots per dataset
plot_idx = 1;


% loop through all datasets
for folder_index = 1:length(GT_folders)
    folder_name = GT_folders{folder_index};
    dataset_name = datasets{folder_index};
    
    % change to the dataset directory
    cd(GT_folders{folder_index});
    
    % find all GT neurons
    neuron_files = dir('CAttached*.mat');
    num_neurons = length(neuron_files);
    
    % initialize arrays for the current dataset
    dataset_amplitude_changes = [];
    dataset_amplitudes = [];
    dataset_baselines = [];
    dataset_delta_f_f0 = [];
    dataset_median_amplitude_changes = [];
    
    % loop through all neurons in the current dataset
    for neuron_index = 1:num_neurons

        % load the data
        load(neuron_files(neuron_index).name);

        amplitude_changes_neuron = [];
        amplitudes_neuron = [];
        baselines_neuron = [];
        delta_f_f0_neuron = [];
        median_amplitude_changes_neuron = [];
    
        
        % loop through all recordings in the current neuron
        for recording_idx = 1:length(CAttached)
            cell_data = CAttached{recording_idx};
            
            % load calcium trace
            measured_trace = cell_data.fluo_mean;
            
            % calculate baseline
            baseline = nanmedian(measured_trace);

            % detect transients
            transients = diff(smooth(measured_trace, smoothing_value)) > threshold;
            
            % detect connected transients
            transient_labels = bwlabel(transients);
            detected_components = regionprops(transient_labels);
            
            % allocate matrix for isolated events
            detected_events = zeros(size(measured_trace));
            
            % process event candidates
            for i = 1:numel(detected_components)
                if detected_components(i).Area < duration_threshold
                    centroid = round(detected_components(i).Centroid);
                    detected_events(centroid) = 1;
                end
            end
            
            % find times of detected small events
            all_event_times = find(detected_events);
            
            % allocate matrices for results
            amplitude_changes_temp = zeros(size(all_event_times));
            amplitudes_temp = zeros(size(all_event_times));
            delta_f_f0_temp = zeros(size(all_event_times));
            
            % calculate amplitude changes and exact amplitudes
            for k = 1:numel(all_event_times)
                if all_event_times(k) > offset_time && all_event_times(k) + offset_time <= length(measured_trace)
                    amplitude_changes_temp(k) = measured_trace(all_event_times(k) + offset_time) - measured_trace(all_event_times(k) - offset_time);
                    amplitudes_temp(k) = measured_trace(all_event_times(k));
                    delta_f_f0_temp(k) = (amplitudes_temp(k) - baseline) / baseline;
                end
            end 

            % store results for current recording
            median_amplitude_changes_neuron = [median_amplitude_changes_neuron; amplitude_changes_temp(:)];
            amplitude_changes_neuron = [amplitude_changes_neuron; amplitude_changes_temp(:)];
            amplitudes_neuron = [amplitudes_neuron; amplitudes_temp(:)];
            baselines_neuron = [baselines_neuron; baseline];
            delta_f_f0_neuron = [delta_f_f0_neuron; delta_f_f0_temp(:)];
        end

        % stroe results for current neuron
        dataset_median_amplitude_changes = [dataset_median_amplitude_changes; median(median_amplitude_changes_neuron)];
        dataset_amplitude_changes = [dataset_amplitude_changes; median(amplitude_changes_neuron)];
        dataset_amplitudes = [dataset_amplitudes; median(amplitudes_neuron)];
        dataset_baselines = [dataset_baselines; median(baselines_neuron)];
        dataset_delta_f_f0 = [dataset_delta_f_f0; median(delta_f_f0_neuron)];
    end

    
    % store results for the current dataset
    median_amplitude_changes{folder_index} = dataset_median_amplitude_changes;
    amplitude_changes{folder_index} = dataset_amplitude_changes;
    amplitudes{folder_index} = dataset_amplitudes;
    baselines{folder_index} = dataset_baselines;
    delta_f_f0{folder_index} = dataset_delta_f_f0;
    
    % plot histograms for the current dataset
    subplot(ceil(total_plots / 4), 4, plot_idx);
    histogram(dataset_amplitude_changes, 20);
    title(dataset_name);
    xlabel('Amplitude Change');
    plot_idx = plot_idx + 1;
    
    subplot(ceil(total_plots / 4), 4, plot_idx);
    histogram(dataset_amplitudes, 20);
    title(dataset_name);
    xlabel('Fluorescence');
    plot_idx = plot_idx + 1;
    
    subplot(ceil(total_plots / 4), 4, plot_idx);
    histogram(dataset_delta_f_f0, 20);
    title(dataset_name);
    xlabel('ΔF/F0');
    plot_idx = plot_idx + 1;
    
    subplot(ceil(total_plots / 4), 4, plot_idx);
    histogram(dataset_baselines, 20);
    title(dataset_name);
    xlabel('Baseline Fluorescence');
    plot_idx = plot_idx + 1;


    cd ..
end

%% Amplitudes Visualization

% colors for each dataset
colors = {'k','k','k','c','c','m','r','b','g','k','k','k'};


% Line graph
figure;
hold on;
for i = 1:numel(amplitudes)
    amplitudes_to_plot = sort(amplitudes{i}); 
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
for i = 1:numel(amplitudes)
    histogram(amplitudes{i}, 'DisplayName', datasets{i}, 'BinWidth', 0.05,...
        'Normalization', 'probability', 'FaceColor', colors{i}, 'FaceAlpha', 0.5);
end
hold off;
title('Distribution of Neuron Amplitudes');
xlabel('Amplitude (dF/F)');
ylabel('Probability');
legend('show', 'Location', 'eastoutside');


% Box plot
figure;
hold on;
for i = 1:numel(amplitudes)
    amplitudes_to_plot = amplitudes{i};
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

% calculate metrics for each dataset
for folder_index = 1:numel(GT_folders)

    mean_amplitudes(folder_index) = mean(amplitudes{folder_index});
    median_amplitudes(folder_index) = median(amplitudes{folder_index});
    snr(folder_index) = mean(amplitudes{folder_index}) / std(baselines{folder_index});

end


% create a table with the results
results_table = table(datasets', mean_amplitudes', median_amplitudes', snr', ...
    'VariableNames', {'Dataset', 'MeanAmplitude', 'MedianAmplitude', 'SNR'});
