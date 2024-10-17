%% Autocalibration

% load the data
data = load('CAttached_jGCaMP8s_472181_1_mini.mat');

% access the CAttached field
CAttached = data.CAttached;

% first recording
i = 1;
cell_data = CAttached{i};

% assign time and dt
time = cell_data.fluo_time;
dt = nanmedian(diff(time));

% ground truth AP events
ap_events = cell_data.events_AP / 1e4;

% experimentally measured calcium trace
measured_trace = cell_data.fluo_mean;

% parameters
threshold = 0.05; % arbitrary!
smoothing_value = 5; % arbitrary!
duration_threshold = 3;  % arbitrary!
offset_time = 3; % arbitrary!

% detect transients
transients = diff(smooth(measured_trace,smoothing_value))> threshold;

% detect connected transients (in order to exclude large transients)
% bwlabel and regionprops are very powerful methods in Matlab for general segmentation in 1D or 2D
transient_labels = bwlabel(transients); 
detected_components = regionprops(transient_labels);

% allocate matrix where future isolated events will be stored
detected_events = zeros(size(measured_trace));

% go through all event candidates
for i = 1:numel(detected_components)

    % only keep short events (that is, events that are probably small)
    if detected_components(i).Area < duration_threshold
        
        % centroids of these events
        centroid = round(detected_components(i).Centroid);
        
        % set this event time point to 1 (all other time points will remain 0)
        detected_events(centroid) = 1;
    end
end

% find times of detected small events
all_event_times = find(detected_events);

% allocate matrix of detected event amplitudes
all_amplitude_changes = NaN*zeros(size(all_event_times));

% go through all detected small events
for k = 1:numel(all_event_times)

    if all_event_times(k) > 5 % if all_event_times(k) is very small, there is no previous time point to compare to

        % retrieve amplitude change: difference of trace after
        % (all_event_times(k)+offset_time) and the trace before
        % (all_event_times(k)-offset_time).
        all_amplitude_changes(k) = measured_trace(all_event_times(k)+offset_time) - measured_trace(all_event_times(k)-offset_time);
    end
end

% compute the median of the distribution of detected amplitudes
nanmedian(all_amplitude_changes)

figure(12); hist(all_amplitude_changes,20)

