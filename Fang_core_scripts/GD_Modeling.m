%% Gradient Descent Modeling


cd('Autocalibrated-spike-inference/GT_autocalibration')


% preparations
dataset_folder = 'DS32-GCaMP8s-m-V1';
dataset_name = 'GCaMP8s';
cd(dataset_folder)

neuron_files = dir('CAttached*.mat');
num_neurons = numel(neuron_files);

modeling_amp = [];
neuron_amplitudes = [];

% randomly select two neurons to plot
selected_neurons = randperm(num_neurons, 2); %change if u wanna more plots

figure('Name','GCaMP8 Selected Neuron Traces', 'Position', [100, 100, 1000, 1000]);

% process each neuron
for neuron_idx = 1:num_neurons

    fprintf('Processing neuron %d of %d\n', neuron_idx, num_neurons);

    load(neuron_files(neuron_idx).name);

    num_recordings = numel(CAttached);
    
    % choose a random recording from selected neurons
    selected = find(selected_neurons == neuron_idx);
    if ~isempty(selected)
        plot_recording_idx = randi(num_recordings);
    end
    
    % process each recording
    for recording_idx = 1:num_recordings
        fprintf('Processing recording %d of %d\n', recording_idx, num_recordings);

        % extract data
        fluo_time = CAttached{recording_idx}.fluo_time;
        fluo_trace = CAttached{recording_idx}.fluo_mean;
        AP_times = CAttached{recording_idx}.events_AP / 1e4;
        
        % generate simulated trace and extract parameters
        [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] = ...
            Gradient_Descent(fluo_time, AP_times, fluo_trace);
        
        % store all amplitudes from each recording as our modeling amplitudes
        modeling_amp = [modeling_amp; optimized_amplitude];
        
        % plot only the selected random neuron and recording
        if ~isempty(selected) && recording_idx == plot_recording_idx

            % generate simulated trace for plotting
            simulated_trace = zeros(size(fluo_time));

            for j = 1:length(AP_times)
                t_since_ap = fluo_time - AP_times(j);
                t_since_ap(t_since_ap < 0) = 1e12;

                simulated_trace = simulated_trace + optimized_amplitude * ...
                    (exp(-t_since_ap / optimized_tau_decay) .* (1 - exp(-t_since_ap / optimized_tau_rise)));
            end
            
            subplot(1, 2, selected);
            plot(fluo_time, fluo_trace, 'b', fluo_time, simulated_trace, 'r');
            title(sprintf('Neuron %d Recording %d Amp: %.2f', neuron_idx, plot_recording_idx, modeling_amp(plot_recording_idx)));
            xlabel('Time (s)');
            ylabel('Fluorescence');
            legend('Measured', 'Simulated','Location', 'best');

        end
    end

    % store also median amplitude of each neuron
    neuron_median_amplitude = median(modeling_amp);
    neuron_amplitudes = [neuron_amplitudes; neuron_median_amplitude];

end

sgtitle('Measured vs Simulated Calcium Trace - Two Random Examples');



%% Visualization of amplitudes

figure('Name', 'GCaMP8s Amplitude Distribution');

% Histogram
subplot(2,1,1);
histogram(modeling_amp(modeling_amp <= 5), 'BinWidth', 0.05, ...
    'Normalization', 'probability', 'FaceColor', 'g');
title('Distribution of GCaMP8s Neuron Amplitudes');
xlabel('Amplitude (dF/F)');
ylabel('Probability');

% Box plot
subplot(2,1,2);
boxplot(modeling_amp(modeling_amp <= 5));
title('GCaMP8s Amplitude Box Plot');
ylabel('Amplitude (dF/F)');


% statistics
mean_amplitude = mean(modeling_amp);
median_amplitude = median(modeling_amp);
std_amplitude = std(modeling_amp);

fprintf('\nResults for %s:\n', dataset_name);
fprintf('Mean Amplitude: %.2f\n', mean_amplitude);
fprintf('Median Amplitude: %.2f\n', median_amplitude);
fprintf('Standard Deviation: %.2f\n', std_amplitude);


%% Gradient Descent Function


function [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] ...
    = Gradient_Descent(time, ap_events, measured_trace)
    
    % calculate drift
    drift = movquant(measured_trace', 0.10, 4000, 1, 'omitnan', 'zeropad');
    
    % initial parameters
    amplitude = 1.68;
    tau_rise = 0.05;
    tau_decay = 0.5;
    %baseline = nanmedian(measured_trace);

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

        % add baseline fluctuation
        %baseline_fluctuation = baseline * (1 + 0.1 * sin(2 * pi * time / max(time)));
        %simulated_trace = simulated_trace + baseline_fluctuation;

        % add noise
        %noise_level = 0.1;
        %noise_std = noise_level * baseline;
        %sigma = noise_std * randn(size(simulated_trace));
        %simulated_trace = simulated_trace + sigma;

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
