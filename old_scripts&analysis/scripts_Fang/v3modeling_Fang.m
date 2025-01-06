%% Modeling for a single neuron (evaluate the performance of GD)


% load the data
data = load('CAttached_jGCaMP8s_472181_1_mini.mat');
CAttached = data.CAttached;

figure;
nRows = ceil(length(CAttached) / 2);
nCols = min(length(CAttached), 2);

error_bef = [];

% loop through each cell in CAttached
for i = 1:length(CAttached)
    cell_data = CAttached{i};
    
    % determine dt
    time = cell_data.fluo_time;
    dt = nanmedian(diff(time));
    ap_events = cell_data.events_AP / 1e4;

    % define parameters (optimize later)
    amplitude = 1;
    tau_rise = 0.05;
    tau_decay = 0.5;

    % generate the simulated trace for the current cell
    simulated_trace = zeros(size(time));

    for j = 1:length(ap_events)
        t_since_ap = time - ap_events(j);
        t_since_ap(t_since_ap < 0) = 1e12;

        % Double exponential template
        simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* ...
            (1 - exp(-t_since_ap / (tau_rise))));
    end

    % calculate baseline
    measured_trace = cell_data.fluo_mean;
    baseline = nanmedian(measured_trace);
    
    % add baseline fluctuation
    baseline_fluctuation = baseline * (1 + 0.1 * sin(2 * pi * time / max(time)));
    %simulated_trace = simulated_trace + baseline_fluctuation;

    % calculate the standard deviation of the noise based on the baseline
    noise_level = 0.1;
    noise_std = noise_level * baseline;
    sigma = noise_std * randn(size(simulated_trace));
    %simulated_trace = simulated_trace + sigma;

    % drift
    drift = movquant(measured_trace', 0.10, 4000, 1,'omitnan','zeropad');
    measured_trace_high_pass_filtered = measured_trace' - drift;
    simulated_trace_drift = simulated_trace' + drift;
    simulated_trace = simulated_trace_drift';

    % plot both traces for the current cell
    subplot(nRows, nCols, i); 
    plot(time, measured_trace, 'b', time, simulated_trace, 'r');
    legend('Measured', 'Simulated');
    xlabel('Time (s)');
    ylabel('ΔF/F');
    title(['Measured vs Simulated Calcium Trace for recording ', num2str(i)]);

    % compute error (MSE)
    error = mean((measured_trace - simulated_trace).^2);
    fprintf('Mean Squared Error for recording %d: %f\n', i, error);
    error_bef = [error_bef, error];
end

%% Optimize parameters using Gradient Descent with Adaptive Iterations

error_aft = [];
figure;
nRows = ceil(length(CAttached) / 2);
nCols = min(length(CAttached), 2);


% loop through each cell in CAttached
for i = 1:length(CAttached)
    cell_data = CAttached{i};
    
    time = cell_data.fluo_time;
    dt = nanmedian(diff(time));
    
    ap_events = cell_data.events_AP / 1e4;
    measured_trace = cell_data.fluo_mean;

    % initial parameter values
    amplitude = 1.0;
    tau_rise = 0.05;
    tau_decay = 0.5;
    baseline = nanmedian(measured_trace);

    % Gradient Descent parameters
    learning_rate = 0.01;
    max_iterations = 5000;  % maximum number of iterations
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

    % plot the optimized simulated trace
    subplot(nRows, nCols, i); 
    plot(time, measured_trace, 'b', time, simulated_trace, 'g');
    legend('Measured', 'Simulated');
    xlabel('Time (s)');
    ylabel('ΔF/F');
    title(['Measured vs Simulated Calcium Trace for recording ', num2str(i)]);

    error_aft = [error_aft, error];

    % display the optimized parameters and MSE
    fprintf('Recording %d: Optimized Amplitude: %f, Optimized Tau Rise: %f, Optimized Tau Decay: %f, Optimized MSE: %f\n', ...
        i, amplitude, tau_rise, tau_decay, error);

    fprintf('Optimization completed in %d iterations\n', iteration);

end


% Compare the improvement

error_table = table((1:length(CAttached))', error_bef', error_aft', ...
    'VariableNames', {'Recording', 'MSE_Before', 'MSE_After'});
disp(error_table)

