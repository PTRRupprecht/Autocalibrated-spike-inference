%% Quickly Test Gradient Descent Function by a single neuron

% load the data
data = load('CAttached_jGCaMP8f_478349_3_mini.mat');

% access the CAttached field
CAttached = data.CAttached;

% initializa array to store amplitudes
all_amplitudes = [];
all_error = [];
all_amplitude_error = [];

% loop through each cell in CAttached
for i = 1:length(CAttached)

    % extract data
    fluo_time = CAttached{i}.fluo_time;
    fluo_trace = CAttached{i}.fluo_mean;
    AP_times = CAttached{i}.events_AP / 1e4;
    
    % generate simulated trace and extract parameters
    [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] = ...
        Gradient_Descent(fluo_time, AP_times, fluo_trace);
    
    % store the amplitude
    all_amplitudes = [all_amplitudes; optimized_amplitude];

    % generate final simulated trace for plotting
    simulated_trace = zeros(size(fluo_time));
    for j = 1:length(AP_times)
        t_since_ap = fluo_time - AP_times(j);
        t_since_ap(t_since_ap < 0) = 1e12;
        simulated_trace = simulated_trace + optimized_amplitude * ...
            (exp(-t_since_ap / optimized_tau_decay) .* (1 - exp(-t_since_ap / optimized_tau_rise)));
    end

    all_error = [all_error; final_error];
    fprintf('Recording %d Final Error: %.4f\n', i, final_error);

    % find peaks in ground truth and simulated traces
    [real_peaks] = findpeaks(fluo_trace, 'MinPeakProminence', 0.1);  % Adjust 'MinPeakProminence' based on data
    [sim_peaks] = findpeaks(simulated_trace, 'MinPeakProminence', 0.1);

    % align the number of peaks (if different)
    min_len = min(length(real_peaks), length(sim_peaks));
    real_peaks = real_peaks(1:min_len);
    sim_peaks = sim_peaks(1:min_len);

    % calculate amplitude error (absolute and percentage)
    amplitude_error = abs(real_peaks - sim_peaks);

    % store amplitude error for each recording
    all_amplitude_error = [all_amplitude_error; mean(amplitude_error)];  % Mean absolute error

    % print results
    fprintf('Recording %d Amplitude Error (Mean Absolute): %.4f\n', i, mean(amplitude_error));

end

%% Gradient Descent Function
function [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] ...
    = Gradient_Descent(time, ap_events, measured_trace)
    
    % Calculate drift once at the beginning
    drift = movquant(measured_trace', 0.10, 4000, 1, 'omitnan', 'zeropad');
    
    % initial parameter values
    amplitude = 1;
    tau_rise = 0.05;
    tau_decay = 0.5;
    baseline = nanmedian(measured_trace);

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
        
        % Generate basic simulated trace
        simulated_trace = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* ...
                (1 - exp(-t_since_ap / (tau_rise))));
        end

        % add noise
        %noise_level = 0.5;
        %noise_std = noise_level * baseline;
        %sigma = noise_std * randn(size(simulated_trace));
        %simulated_trace = simulated_trace + sigma;

        % add baseline fluctuation
        baseline_fluctuation = baseline * (1 + 0.1 * sin(2 * pi * time / max(time)));
        simulated_trace = simulated_trace + baseline_fluctuation;

        % Add drift to simulated trace
        simulated_trace_with_drift = simulated_trace + drift';

        % compute error (MSE) with drift included
        error = mean((measured_trace - simulated_trace_with_drift).^2);

        % calculate error change
        error_change = abs(prev_error - error);
        prev_error = error;

        % calculate gradients comparing full signals (with drift)
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

    % Final error is already calculated with drift
    final_error = error;
    optimized_amplitude = amplitude;
    optimized_tau_rise = tau_rise;
    optimized_tau_decay = tau_decay;
end
