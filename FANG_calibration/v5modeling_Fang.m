%% GD Modeling across GCaMP8 datasets_2

cd('Autocalibrated-spike-inference/GT_autocalibration')


% list of folders
GT_folders = {'DS30-GCaMP8f-m-V1', 'DS31-GCaMP8m-m-V1', 'DS32-GCaMP8s-m-V1'};


datasets = {'GC8f','GC8m','GC8s'};


% number of neurons to plot per dataset
neurons_plot = 8;

% plot for each folder
for folder_index = 1:length(GT_folders)
    folder_name = GT_folders{folder_index};
    dataset_name = datasets{folder_index};
    cd(GT_folders{folder_index})
    
    % find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    % select 10 random neurons
    num_neurons = min(neurons_plot, numel(neuron_files));
    neuron_indices = randperm(numel(neuron_files), num_neurons); 
    
    % create a figure with subplots for the selected neurons
    subplot_rows = ceil(sqrt(num_neurons));
    subplot_cols = ceil(num_neurons / subplot_rows);
    figure('Name', sprintf('Calcium Traces - %s', dataset_name), 'Position', [100, 100, 1200, 800]);
    
    for i = 1:num_neurons
        neuron_index = neuron_indices(i); 
        load(neuron_files(neuron_index).name);
        
        % select a random recording for each neuron
        recording_index = randi(numel(CAttached));
        
        % extract data
        fluo_time = CAttached{recording_index}.fluo_time;
        fluo_trace = CAttached{recording_index}.fluo_mean;
        AP_times = CAttached{recording_index}.events_AP / 1e4;
    
        % generate simulated trace
        [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, ~] = ...
            Gradient_Descent(fluo_time, AP_times, fluo_trace);
        
        simulated_trace = zeros(size(fluo_time));
        for j = 1:length(AP_times)
            t_since_ap = fluo_time - AP_times(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace = simulated_trace + optimized_amplitude * ...
                (exp(-t_since_ap / optimized_tau_decay) .* (1 - exp(-t_since_ap / optimized_tau_rise)));
        end
        
        % plot results in subplot
        subplot(subplot_rows, subplot_cols, i);
        plot(fluo_time, fluo_trace, 'b', fluo_time, simulated_trace, 'r');
        title(sprintf('Neuron %d', neuron_index)); 
        %if i == num_neurons  % only add legend to the last subplot
        legend('Measured', 'Simulated', 'Location', 'southoutside');
        
        xlabel('Time (s)');
        ylabel('Fluorescence');
    end
    
    % title for the figure
    sgtitle(sprintf('Calcium Traces - %s', dataset_name));
    
    cd ..
end


%% Amplitude Extraction and Visualization


cd('Autocalibrated-spike-inference/GT_autocalibration')


% list of folders
GT_folders = {'DS30-GCaMP8f-m-V1', 'DS31-GCaMP8m-m-V1', 'DS32-GCaMP8s-m-V1'};


datasets = {'GC8f','GC8m','GC8s'};
colors = {'r' 'b' 'g'};


% initialize a cell array to store amplitudes for each dataset
all_amplitudes = cell(length(GT_folders), 1);


% parallel processing
%{
parfor folder_index = 1:length(GT_folders)
    folder_name = GT_folders{folder_index};
    dataset_name = datasets{folder_index};
%}

for folder_index = 1:length(GT_folders)
    folder_name = GT_folders{folder_index};
    dataset_name = datasets{folder_index};
    cd(GT_folders{folder_index})
    
    % Find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    % Initialize array to store amplitudes for this dataset
    dataset_amplitudes = [];

    for neuron_index = 1:numel(neuron_files)
        data = load(fullfile(folder_name, neuron_files(neuron_index).name));

        % process all recordings for this neuron
        for recording_index = 1:numel(data.CAttached)

            % extract data
            fluo_time = data.CAttached{recording_index}.fluo_time;
            fluo_trace = data.CAttached{recording_index}.fluo_mean;
            AP_times = data.CAttached{recording_index}.events_AP / 1e4;

            %{
            % adjust vectors
            if size(fluo_time, 2) > 1
                fluo_time = fluo_time';
            end

            if size(fluo_trace, 2) > 1
                fluo_trace = fluo_trace';
            end
            
            % remove NaN values
            good_indices = ~isnan(fluo_time) & ~isnan(fluo_trace);
            fluo_time = fluo_time(good_indices);
            fluo_trace = fluo_trace(good_indices);
            %}
            
            % generate simulated trace and extract amplitude
            [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, ~] = ...
                Gradient_Descent(fluo_time, AP_times, fluo_trace);

            % Store the amplitude
            dataset_amplitudes = [dataset_amplitudes; optimized_amplitude];
        end
    end
    
    % Store amplitudes for this dataset
    all_amplitudes{folder_index} = dataset_amplitudes;
    
    cd ..
end



% Visualization
% Line graph
figure;
hold on;
for i = 1:numel(all_amplitudes)
    amplitudes_to_plot = sort(all_amplitudes{i}(all_amplitudes{i} <= 5)); 
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
for i = 1:numel(all_amplitudes)
    histogram(all_amplitudes{i}(all_amplitudes{i} < 5), 'DisplayName', datasets{i}, 'BinWidth', 0.05,...
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
for i = 1:numel(all_amplitudes)
    amplitudes_to_plot = all_amplitudes{i}(all_amplitudes{i} <= 5);
    boxplot(amplitudes_to_plot, 'positions', i, 'colors', colors{i}); 
end
hold off;
set(gca, 'xtick', 1:numel(datasets), 'xticklabel', datasets);
xtickangle(45);




% Find optimal amplitudes for each calcium indicator

% initialize arrays to store relevant metrics
mean_amplitudes = zeros(1, numel(GT_folders));
median_amplitudes = zeros(1, numel(GT_folders));


% calculate metrics for each dataset
for folder_index = 1:numel(GT_folders)
    amplitudes = all_amplitudes{folder_index};

    mean_amplitudes(folder_index) = mean(amplitudes);
    median_amplitudes(folder_index) = median(amplitudes);
    
end
    

% create a table with the results
results_table = table(datasets', mean_amplitudes', median_amplitudes', ...
    'VariableNames', {'Dataset', 'MeanAmplitude', 'MedianAmplitude'});





% Gradient Descent Function

function [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] ...
    = Gradient_Descent(time, ap_events, measured_trace)
    
    % initial parameter values
    amplitude = 1;
    tau_rise = 0.05;
    tau_decay = 0.5;
    baseline = nanmedian(measured_trace);

    % Gradient Descent parameters
    learning_rate = 0.01;
    max_iterations = 1000;  % maximum number of iterations
    convergence_threshold = 1e-5;  % threshold for change in error
    
    % initialize variables for adaptive iterations
    prev_error = Inf;
    error_change = Inf;
    iteration = 0;

    while error_change > convergence_threshold && iteration < max_iterations
        iteration = iteration + 1;
        
        simulated_trace = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* ...
                (1 - exp(-t_since_ap / (tau_rise))));
        end
        
        % add baseline fluctuation
        baseline_fluctuation = baseline * (1 + 0.1 * sin(2 * pi * time / max(time)));
        %simulated_trace = simulated_trace + baseline_fluctuation;

        % add noise
        noise_level = 0.1;
        noise_std = noise_level * baseline;
        sigma = noise_std * randn(size(simulated_trace));
        %simulated_trace = simulated_trace + sigma;

        % compute error (MSE)
        error = mean((measured_trace - simulated_trace).^2);

        % calculate error change
        error_change = abs(prev_error - error);
        prev_error = error;

        % calculate gradients
        grad_amplitude = 0;
        grad_tau_rise = 0;
        grad_tau_decay = 0;
        
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            
            grad_amplitude = grad_amplitude - 2 * mean((measured_trace - simulated_trace) .* ...
                (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise)))));
            
            grad_tau_rise = grad_tau_rise +  2 * amplitude * mean((measured_trace - simulated_trace) .* ...
                (t_since_ap / (tau_rise^2)) .* exp(-t_since_ap / (tau_rise)));
            
            grad_tau_decay = grad_tau_decay - 2 * amplitude * mean((measured_trace - simulated_trace) .* ...
                (t_since_ap / (tau_decay^2)) .* exp(-t_since_ap / (tau_decay)));
        end

        % update parameters
        amplitude = amplitude - learning_rate * grad_amplitude;
        tau_rise = tau_rise - learning_rate * grad_tau_rise;
        tau_decay = tau_decay - learning_rate * grad_tau_decay;

        % ensure parameters stay positive
        amplitude = max(amplitude, 0.01);
        tau_rise = max(tau_rise, 0.01);
        tau_decay = max(tau_decay, 0.01);
        
        % print progress every 100 iterations
        if mod(iteration, 50) == 0
            fprintf('Iteration %d: Error = %f, Change in Error = %f\n', iteration, error, error_change);
        end
    end

        % drift
        drift = movquant(measured_trace', 0.10, 4000, 1,'omitnan','zeropad');
        simulated_trace_drift = simulated_trace' + drift;
        simulated_trace = simulated_trace_drift';

        % compute error (MSE)
        error = mean((measured_trace - simulated_trace).^2);
    

    optimized_amplitude = amplitude;
    optimized_tau_rise = tau_rise;
    optimized_tau_decay = tau_decay;
    final_error = error;
end
