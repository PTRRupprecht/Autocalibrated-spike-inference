%% Modeling for a single neuron


cd('Autocalibrated-spike-inference/GT_autocalibration')

% set up the data set to be analyzed
dataset_folder = 'DS32-GCaMP8s-m-V1';
dataset_name = 'GCaMP8s';

cd(dataset_folder)

% find all GT neurons
neuron_files = dir('CAttached*.mat');
num_neurons = numel(neuron_files);

% choose one neuron to analyze
selected_neuron = 10;

% create a figure
figure('Name', sprintf('GCaMP8s Neuron %d Analysis', selected_neuron), 'Position', [100, 100, 1200, 800]);

% load neuron data
load(neuron_files(selected_neuron).name);

% initialize arrays to store results
recording_amplitudes = [];
all_traces = {};
all_simulated = {};

% process all recordings for this neuron
num_recordings = numel(CAttached);

% create subplots based on number of recordings
num_rows = ceil(sqrt(num_recordings));
num_cols = ceil(num_recordings/num_rows);

for recording_idx = 1:num_recordings
    fprintf('Processing recording %d of %d\n', recording_idx, num_recordings);

    % extract data
    fluo_time = CAttached{recording_idx}.fluo_time;
    fluo_trace = CAttached{recording_idx}.fluo_mean;
    AP_times = CAttached{recording_idx}.events_AP / 1e4;
    
    % generate simulated trace and extract parameters
    [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] = ...
        Gradient_Descent(fluo_time, AP_times, fluo_trace);
    
    % store the amplitude for this recording
    recording_amplitudes = [recording_amplitudes; optimized_amplitude];
    
    % generate simulated trace for plotting
    simulated_trace = zeros(size(fluo_time));
    
    for j = 1:length(AP_times)
        t_since_ap = fluo_time - AP_times(j);
        t_since_ap(t_since_ap < 0) = 1e12;
        
        simulated_trace = simulated_trace + optimized_amplitude * ...
            (exp(-t_since_ap / optimized_tau_decay) .* (1 - exp(-t_since_ap / optimized_tau_rise)));
    end
    
    % store traces for later use
    all_traces{recording_idx} = fluo_trace;
    all_simulated{recording_idx} = simulated_trace;
    
    % plot each recording
    subplot(num_rows, num_cols, recording_idx);
    plot(fluo_time, fluo_trace, 'b', fluo_time, simulated_trace, 'r');
    title(sprintf('Recording %d (A: %.2f)', recording_idx, optimized_amplitude));
    xlabel('Time (s)');
    ylabel('Fluorescence');
    if recording_idx == 1  % Only show legend for first subplot to save space
        legend('Measured', 'Simulated', 'Location', 'best');
    end
end

% add overall title
sgtitle(sprintf('Neuron %d - All Recordings Analysis', selected_neuron));

% create a new figure for amplitude analysis
figure('Name', sprintf('GCaMP8s Neuron %d Amplitude Analysis', selected_neuron));

% Histogram
subplot(2,1,1);
histogram(recording_amplitudes, 'BinWidth', 0.05, ...
    'Normalization', 'probability', 'FaceColor', 'g');
title(sprintf('Distribution of Amplitudes for Neuron %d', selected_neuron));
xlabel('Amplitude (dF/F)');
ylabel('Probability');

% Box plot
subplot(2,1,2);
boxplot(recording_amplitudes);
title(sprintf('Amplitude Box Plot for Neuron %d', selected_neuron));
ylabel('Amplitude (dF/F)');

% Calculate and display statistics
mean_amplitude = mean(recording_amplitudes);
median_amplitude = median(recording_amplitudes);
std_amplitude = std(recording_amplitudes);

fprintf('\nResults for Neuron %d:\n', selected_neuron);
fprintf('Mean Amplitude: %.2f\n', mean_amplitude);
fprintf('Median Amplitude: %.2f\n', median_amplitude);
fprintf('Standard Deviation: %.2f\n', std_amplitude);


%% Gradient Descent Function remains the same as in original script
function [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] ...
    = Gradient_Descent(time, ap_events, measured_trace)
    
    % calculate drift
    drift = movquant(measured_trace', 0.10, 4000, 1, 'omitnan', 'zeropad');
    
    % initial parameters
    amplitude = 1;
    tau_rise = 0.05;
    tau_decay = 0.5;

    % Gradient Descent parameters
    learning_rate = 0.01;
    max_iterations = 2000;
    convergence_threshold = 1e-6;
    
    % initialize variables for adaptive iterations
    prev_error = Inf;
    error_change = Inf;
    iteration = 0;

    while error_change > convergence_threshold && iteration < max_iterations
        iteration = iteration + 1;
        
        % generate basic simulated trace (double exponential template)
        simulated_trace = zeros(size(time));

        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;

            simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* ...
                (1 - exp(-t_since_ap / (tau_rise))));
        end

        % add drift to simulated trace
        simulated_trace_with_drift = simulated_trace + drift';

        % compute error (MSE)
        error = mean((measured_trace - simulated_trace_with_drift).^2);

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
            
            grad_amplitude = grad_amplitude - 2 * mean((measured_trace - simulated_trace_with_drift) .* ...
                (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise)))));
            
            grad_tau_rise = grad_tau_rise + 2 * amplitude * mean((measured_trace - simulated_trace_with_drift) .* ...
                (t_since_ap / (tau_rise^2)) .* exp(-t_since_ap / (tau_rise)));
            
            grad_tau_decay = grad_tau_decay - 2 * amplitude * mean((measured_trace - simulated_trace_with_drift) .* ...
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
        if mod(iteration, 100) == 0
            fprintf('Iteration %d: Error = %f, Change in Error = %f\n', iteration, error, error_change);
        end
    end

    % final error and parameters
    final_error = error;
    optimized_amplitude = amplitude;
    optimized_tau_rise = tau_rise;
    optimized_tau_decay = tau_decay;
    
end

