%% Modeling method_v2

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
    %simulated_trace = simulated_trace + sigma; 

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
    learning_rate = 0.1;
    n_iterations = 100;

    for iter = 1:n_iterations
        
        if mod(iter,40) == 0 && iter > 1
            learning_rate =  learning_rate/2;
        end

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
        %simulated_trace = simulated_trace;% + sigma;

        % compute error (MSE)
        error = mean((measured_trace - simulated_trace).^2);


        % gradient with respect to amplitude
        change_amplitude = 0.01*amplitude;

        simulated_trace_positive = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace_positive = simulated_trace_positive + (amplitude+change_amplitude) * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
        end
        simulated_trace_negative = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace_negative = simulated_trace_negative + (amplitude-change_amplitude) * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
        end
        
        error_positive = mean((measured_trace - simulated_trace_positive).^2);
        error = mean((measured_trace - simulated_trace).^2);
        error_negative = mean((measured_trace - simulated_trace_negative).^2);

        amplitude = amplitude - learning_rate* (error_positive - error_negative)/change_amplitude;

        % gradient with respect to amplitude
        change_tau_rise = 0.01*tau_rise;

        simulated_trace_positive = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace_positive = simulated_trace_positive + (amplitude) * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise+change_tau_rise))));
        end
        simulated_trace_negative = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace_negative = simulated_trace_negative + (amplitude) * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise-change_tau_rise))));
        end
        
        error_positive = mean((measured_trace - simulated_trace_positive).^2);
        error = mean((measured_trace - simulated_trace).^2);
        error_negative = mean((measured_trace - simulated_trace_negative).^2);

        tau_rise = tau_rise - learning_rate*(error_positive - error_negative)/change_tau_rise;


        % gradient with respect to amplitude
        change_tau_decay = 0.01*tau_decay;

        simulated_trace_positive = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace_positive = simulated_trace_positive + (amplitude) * (exp(-t_since_ap / (tau_decay+change_tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
        end
        simulated_trace_negative = zeros(size(time));
        for j = 1:length(ap_events)
            t_since_ap = time - ap_events(j);
            t_since_ap(t_since_ap < 0) = 1e12;
            simulated_trace_negative = simulated_trace_negative + (amplitude) * (exp(-t_since_ap / (tau_decay-change_tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
        end
        
        error_positive = mean((measured_trace - simulated_trace_positive).^2);
        error = mean((measured_trace - simulated_trace).^2);
        error_negative = mean((measured_trace - simulated_trace_negative).^2);

        tau_decay = tau_decay - learning_rate*(error_positive - error_negative)/change_tau_decay;



        % ensure parameters stay positive
        amplitude = max(amplitude, 0.01);
        tau_rise = max(tau_rise, 0.002);
        tau_decay = max(tau_decay, 0.01);


        disp([amplitude,tau_rise,tau_decay])

        disp(error)

    end


    simulated_trace = zeros(size(time));
    for j = 1:length(ap_events)
        t_since_ap = time - ap_events(j);
        t_since_ap(t_since_ap < 0) = 1e12;
        simulated_trace = simulated_trace + amplitude * (exp(-t_since_ap / (tau_decay)) .* (1 - exp(-t_since_ap / (tau_rise))));
    end
        

    % plot the optimized simulated trace
    figure;
    measured_trace_high_pass_filtered = measured_trace' - movquant(measured_trace', 0.10, 4000, 1,'omitnan','zeropad');
    plot(time, measured_trace, 'b', time, simulated_trace, 'r');
    legend('Measured', 'Optimized Simulated');
    xlabel('Time (s)');
    ylabel('Fluorescence');
    title(['Optimized Simulated Calcium Trace for recording ', num2str(i)]);

    % display the optimized parameters and MSE
    fprintf('Cell %d: Optimized Amplitude: %f, Optimized Tau Rise: %f, Optimized Tau Decay: %f, Optimized MSE: %f\n', i, amplitude, tau_rise, tau_decay, error);
end