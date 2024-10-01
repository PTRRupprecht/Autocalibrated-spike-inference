%% Deconvolution method

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

% Extract calcium imaging data
calcium_trace = all_fluo_mean;
time = all_fluo_time;

% Extract the calcium kernel


