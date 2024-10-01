%% Modeling method

% Load the data
data = load('CAttached_jGCaMP8s_472181_1_mini.mat');

% Access the CAttached field
CAttached = data.CAttached;

% Initialize arrays to store data from all cells
all_fluo_time = [];
all_fluo_mean = [];
all_events_AP = [];

% Loop through each cell in CAttached
for i = 1:length(CAttached)
    cell_data = CAttached{i};
    
    % Concatenate data from each cell
    all_fluo_time = [all_fluo_time; cell_data.fluo_time];
    all_fluo_mean = [all_fluo_mean; cell_data.fluo_mean];
    all_events_AP = [all_events_AP; cell_data.events_AP];
end

% Extract action potential events
ap_events = all_events_AP;

% Find the time points of action potentials
ap_times = find(ap_events);

% Define parameters (optimize later)
amplitude = 0.5;  
tau = 0.5;

% Generate the full trace
time = all_fluo_time;
simulated_trace = zeros(size(time));

for i = 1:length(ap_times)
    t_since_ap = time - time(ap_times(i));
    simulated_trace = simulated_trace + calcium_template(t_since_ap, amplitude, tau);
end

% Compare with measured calcium trace
measured_trace = all_fluo_mean;

% Plot both traces
figure;
plot(time, measured_trace, 'b', time, simulated_trace, 'r');
legend('Measured', 'Simulated');
xlabel('Time (s)');
ylabel('Fluorescence');
title('Measured vs Simulated Calcium Trace');


%% Compute error (MSE)
error = compute_mse(measured_trace, simulated_trace);
fprintf('Mean Squared Error: %f\n', error);

% Optimize parameters using gradient descent


% Optimize parameters using grid search



%% Functions
% Build the template for traces
function template = calcium_template(t, amplitude, tau)
    template = amplitude * exp(-t / tau);
end


% Calculates the Mean Squared Error between the measured and simulated calcium traces
function mse = compute_mse(measured, simulated)
    mse = mean((measured - simulated).^2);
end



