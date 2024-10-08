%% Modeling method

% load the data
data = load('CAttached_jGCaMP8s_472181_1_mini.mat');

% access the CAttached field
CAttached = data.CAttached;

% plot each cell in a separate subplot
nRows = ceil(length(CAttached) / 2);
nCols = min(length(CAttached), 2);
figure;

% loop through each cell in CAttached
for i = 1:length(CAttached)
    cell_data = CAttached{i};
    
    % determine dt
    time = cell_data.fluo_time;
    dt = nanmedian(diff(time));
    
    % extract AP events for the current cell
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
        simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
    end

    % calculate baseline
    measured_trace = cell_data.fluo_mean;
    baseline = nanmedian(measured_trace);
    
    % add baseline fluctuation?
    baseline_fluctuation = baseline * (1 + 0.1 * sin(2 * pi * time / max(time)));
    %simulated_trace = simulated_trace + baseline_fluctuation;

    % calculate the standard deviation of the noise based on the baseline
    noise_level = 0.1;
    noise_std = noise_level * baseline;
    sigma = noise_std * randn(size(simulated_trace));
    simulated_trace = simulated_trace + sigma; 

    % plot both traces for the current cell
    subplot(nRows, nCols, i); 
    plot(time, measured_trace, 'b', time, simulated_trace, 'r');
    legend('Measured', 'Simulated');
    xlabel('Time (s)');
    ylabel('Fluorescence');
    title(['Measured vs Simulated Calcium Trace for recording ', num2str(i)]);

    % compute error (MSE)
    error = mean((measured_trace - simulated_trace).^2);
    fprintf('Mean Squared Error for Cell %d: %f\n', i, error);
end

%% Optimize parameters using gradient descent

% loop through each cell in CAttached
for i = 1:length(CAttached)
    cell_data = CAttached{i};
    
    time = cell_data.fluo_time;
    dt = nanmedian(diff(time));
    
    ap_events = cell_data.events_AP / 1e4;
    measured_trace = cell_data.fluo_mean;

    % Initial parameter values
    amplitude = 1;
    tau_rise = 0.05;
    tau_decay = 0.5;
    baseline = nanmedian(measured_trace);

    % Gradient Descent parameters
    learning_rate = 0.01;
    n_iterations = 100;

    for iter = 1:n_iterations
        simulated_trace = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
        end
        
        % add baseline fluctuation
        baseline_fluctuation = baseline * (1 + 0.1 * sin(2 * pi * time / max(time)));
        %simulated_trace = simulated_trace + baseline_fluctuation;

        % add noise
        noise_level = 0.1;
        noise_std = noise_level * baseline;
        sigma = noise_std * randn(size(simulated_trace));
        simulated_trace = simulated_trace + sigma;

        % compute error (MSE)
        error = mean((measured_trace - simulated_trace).^2);

        % calculate gradients
        grad_amplitude = 0;
        grad_tau_rise = 0;
        grad_tau_decay = 0;
        
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            % use * or - here
            grad_amplitude = grad_amplitude - 2 * mean((measured_trace - simulated_trace) .* (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise)))));
            grad_tau_rise = grad_tau_rise + 2 * amplitude * mean((measured_trace - simulated_trace) .* (t_since_ap / (tau_rise^2)) .* exp(-t_since_ap / (tau_rise)));
            grad_tau_decay = grad_tau_decay - 2 * amplitude * mean((measured_trace - simulated_trace) .* (t_since_ap / (tau_decay^2)) .* exp(-t_since_ap / (tau_decay)));
        end

        % update parameters
        amplitude = amplitude - learning_rate * grad_amplitude;
        tau_rise = tau_rise - learning_rate * grad_tau_rise;
        tau_decay = tau_decay - learning_rate * grad_tau_decay;

        % ensure parameters stay positive
        amplitude = max(amplitude, 0.01);
        tau_rise = max(tau_rise, 0.01);
        tau_decay = max(tau_decay, 0.01);
    end



    % plot the optimized simulated trace
    figure;
    plot(time, measured_trace, 'b', time, simulated_trace, 'r');
    legend('Measured', 'Optimized Simulated');
    xlabel('Time (s)');
    ylabel('Fluorescence');
    title(['Optimized Simulated Calcium Trace for recording ', num2str(i)]);

    % display the optimized parameters and MSE
    fprintf('Cell %d: Optimized Amplitude: %f, Optimized Tau Rise: %f, Optimized Tau Decay: %f, Optimized MSE: %f\n', i, amplitude, tau_rise, tau_decay, error);
    end

