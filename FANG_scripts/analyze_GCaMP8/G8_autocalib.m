%% Autocalibration for GCaMP8s


cd('Autocalibrated-spike-inference/GT_autocalibration')


% set up the dataset to be analyzed
dataset_folder = 'DS32-GCaMP8s-m-V1';
dataset_name = 'GCaMP8s';

cd(dataset_folder);


% Parameters
threshold = 0.06;
smoothing_value = 6;
duration_threshold = 4;
offset_time = 4;

% find all GT neurons
neuron_files = dir('CAttached*.mat');
num_neurons = length(neuron_files);

% initialize arrays for the current dataset
amplitude_changes = [];
amplitudes = [];
baselines = [];
delta_f_f0 = [];

% initialize arrays for storing one value per recording
per_recording_amplitudes = [];
per_recording_changes = [];
per_recording_baseline = [];
per_recording_dff = [];

% loop through all neurons in the current dataset
for neuron_index = 1:num_neurons
    
    load(neuron_files(neuron_index).name);
    
    % arrays for all values from current neuron
    all_changes = [];
    all_amplitudes = [];
    all_baselines = [];
    all_dff = [];

    recording_values = [];  % one median amplitude per recording
    
    % loop through all recordings for each neuron
    for index = 1:numel(CAttached)
        
        % load calcium trace
        measured_trace = CAttached{index}.fluo_mean;
        
        % calculate baseline
        baseline = nanmedian(measured_trace);
        
        % calculate deltaF/F0
        delta_f_f0_value = (measured_trace - baseline) / baseline;
        
        % detect transients
        transients = diff(smooth(measured_trace, smoothing_value)) > threshold;
        
        % detect connected transients
        transient_labels = bwlabel(transients);
        detected_components = regionprops(transient_labels);
        
        % initialize matrix for isolated events
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
        

        % store all values
        all_changes = [all_changes; amplitude_changes_temp(:)];
        all_amplitudes = [all_amplitudes; amplitudes_temp(:)];
        all_baselines = [all_baselines; baseline];
        all_dff = [all_dff; delta_f_f0_temp(:)];
        
        % store median value for current recording
        recording_values = [recording_values; median(amplitudes_temp)];
    end
    
    % store values for all recordings
    per_recording_amplitudes = [per_recording_amplitudes; recording_values];

    % store results for current neuron
    amplitude_changes = [amplitude_changes; median(all_changes)];
    amplitudes = [amplitudes; median(all_amplitudes)];
    baselines = [baselines; median(all_baselines)];
    delta_f_f0 = [delta_f_f0; median(all_dff)];
    
end


% Histogram
figure;
histogram(per_recording_amplitudes, 'BinWidth', 0.05, ...
    'Normalization', 'probability', 'FaceColor', 'b');
title([dataset_name, ' Distribution of Neuron deconv_amp']);
xlabel('Amplitude (dF/F)');
ylabel('Probability');


% initialize arrays to store relevant matrics
mean_amplitudes = mean(amplitudes);
median_amplitudes = median(amplitudes);
snr = mean(amplitudes) / std(baselines);

% create a table with the results
results_table = table({dataset_name}', mean_amplitudes', median_amplitudes', snr', ...
    'VariableNames', {'Dataset', 'MeanAmplitude', 'MedianAmplitude', 'SNR'});

% display the results table
disp(results_table);

