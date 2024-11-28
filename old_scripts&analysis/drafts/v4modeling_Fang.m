%% GD Modeling across all datasets

cd('Autocalibrated-spike-inference/GT_autocalibration')

% list of folders
GT_folders = {'DS09-GCaMP6f-m-V1', 'DS10-GCaMP6f-m-V1-neuropil-corrected',...
              'DS11-GCaMP6f-m-V1-neuropil-corrected', 'DS13-GCaMP6s-m-V1-neuropil-corrected',...
              'DS14-GCaMP6s-m-V1', 'DS29-GCaMP7f-m-V1',...
              'DS30-GCaMP8f-m-V1', 'DS31-GCaMP8m-m-V1', 'DS32-GCaMP8s-m-V1',...
              'DS06-GCaMP6f-zf-aDp','DS07-GCaMP6f-zf-dD', 'DS08-GCaMP6f-zf-OB'};

datasets = {'GC6f','GC6f_c','GC6f_c','GC6s_c','GC6s','GC7f','GC8f','GC8m','GC8s','GC6f_zf','GC6f_zf','GC6f_zf'};

% calculate dt_all
dt_all = zeros(numel(GT_folders), 1);

for folder_index = 1:numel(GT_folders)
    cd(GT_folders{folder_index})

    % find all GT neurons
    neuron_files = dir('CAttached*.mat');
    dtX = [];
    
    for neuron_index = 1:numel(neuron_files)
        load(neuron_files(neuron_index).name);
        for index = 1:numel(CAttached)
            fluo_time = CAttached{index}.fluo_time;
            dt = nanmedian(diff(fluo_time));
            dtX = [dtX; dt];
        end
    end
    
    % average across neurons for each GT dataset
    dt_all(folder_index) = nanmedian(dtX);
    
    cd ..
end

% plot for each folder

for folder_index = 1:length(GT_folders)
    folder_name = GT_folders{folder_index};
    dataset_name = datasets{folder_index};
    cd(GT_folders{folder_index})
    
    % find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    % create a figure with subplots for all neurons
    num_neurons = numel(neuron_files);
    subplot_rows = ceil(sqrt(num_neurons));
    subplot_cols = ceil(num_neurons / subplot_rows);
    figure('Name', sprintf('Calcium Traces - %s', dataset_name), 'Position', [100, 100, 1200, 800]);
    
    for neuron_index = 1:num_neurons
        load(neuron_files(neuron_index).name);

        %{
        % combine all recordings for this neuron
        combined_fluo_time = [];
        combined_fluo_trace = [];
        combined_AP_times = [];
        total_time = 0;
        
        for recording_index = 1:numel(CAttached)

            % extract data
            fluo_time = CAttached{recording_index}.fluo_time;
            fluo_trace = CAttached{recording_index}.fluo_mean;
            AP_times = CAttached{recording_index}.events_AP / 1e4;
            
            % ensure vectors are column vectors
            if size(CAttached{index}.fluo_time,2) > 1
                CAttached{index}.fluo_time = CAttached{index}.fluo_time';
            end
            
            if size(CAttached{index}.fluo_mean,2) > 1
                CAttached{index}.fluo_mean = CAttached{index}.fluo_mean';
            end
            
            % remove NaN values
            good_indices = ~isnan(fluo_time) & ~isnan(fluo_trace);
            fluo_time = fluo_time(good_indices);
            fluo_trace = fluo_trace(good_indices);
            
            % adjust time and AP times
            fluo_time = fluo_time - fluo_time(1) + total_time;
            AP_times = AP_times - AP_times(1) + total_time;
            
            % combine data
            combined_fluo_time = [combined_fluo_time; fluo_time];
            combined_fluo_trace = [combined_fluo_trace; fluo_trace];
            combined_AP_times = [combined_AP_times; AP_times];
            
            % update total time
            total_time = combined_fluo_time(end);
        end
        %}

        % select a random recording for each neuron
        recording_index = randi(numel(CAttached));
        
        % extract data
        fluo_time = CAttached{recording_index}.fluo_time;
        fluo_trace = CAttached{recording_index}.fluo_mean;
        AP_times = CAttached{recording_index}.events_AP / 1e4;
        
        % adjust vectors
        if size(CAttached{index}.fluo_time,2) > 1
            CAttached{index}.fluo_time = CAttached{index}.fluo_time';
        end

        if size(CAttached{index}.fluo_mean,2) > 1
            CAttached{index}.fluo_mean = CAttached{index}.fluo_mean';
        end
        
        % remove NaN values
        good_indices = ~isnan(fluo_time) & ~isnan(fluo_trace);
        fluo_time = fluo_time(good_indices);
        fluo_trace = fluo_trace(good_indices);
        
        % compute dt and resample if necessary
        dt = median(diff(fluo_time));
        dt0 = dt_all(folder_index);
        
        if abs(dt - dt0) / dt0 > 0.05
            fluo_trace_resampled = resample(double(fluo_trace), round(1/dt0*100), round(1/dt*100));
            fluo_time_resampled = (dt0:dt0:dt0*numel(fluo_trace_resampled))';
        else
            fluo_trace_resampled = double(fluo_trace);
            fluo_time_resampled = fluo_time + dt0 - fluo_time(1);
        end
        
        % generate simulated trace
        [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, ~] = ...
            Gradient_Descent(fluo_time_resampled, AP_times, fluo_trace_resampled);
        
        simulated_trace = zeros(size(fluo_time_resampled));
        for j = 1:length(AP_times)
            t_since_ap = fluo_time_resampled - AP_times(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace = simulated_trace + optimized_amplitude * ...
                (exp(-t_since_ap / optimized_tau_decay) .* (1 - exp(-t_since_ap / optimized_tau_rise)));
        end
        
        % plot results in subplot
        subplot(subplot_rows, subplot_cols, neuron_index);
        plot(fluo_time_resampled, fluo_trace_resampled, 'b', fluo_time_resampled, simulated_trace, 'r');
        title(sprintf('Neuron %d', neuron_index));
        if neuron_index == num_neurons  % Only add legend to the last subplot
            legend('Measured', 'Simulated', 'Location', 'southoutside');
        end
        xlabel('Time (s)');
        ylabel('Fluorescence');
    end
    
    % add title for the figure
    sgtitle(sprintf('Calcium Traces - %s (dt = %.4f s)', dataset_name, dt_all(folder_index)));

    cd ..

end


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
    max_iterations = 5e3;  % maximum number of iterations
    convergence_threshold = 1e-6;  % threshold for change in error
    
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
        if mod(iteration, 100) == 0
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
