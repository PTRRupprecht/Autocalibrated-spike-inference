
%%

% 1. Extract calcium trace and ground truth such that it can be inserted into Cascade
% 2. Apply Cascade
% 3. Use the output of Cascade to detect events
% 4. From the detected events, make a histogram
% 5. Automatically (or visually) define unitary events (i.e., the smallest observed events)
% 6. Re-scale inferred spike rate with this scaling factor

load('Example_inferred_spike rates.mat')


%%

% threshold deconvolved trace --> threshold not optimized!
event_detection = spike_rates_GC8 > 0.3;

% copy variable and convert from boolean to double
event_detection_scaled = double(event_detection);

% detect contiguous events --> same as before
labels = bwlabel(event_detection);
A = regionprops(labels);

% initialize matrix that will contain the inferred number of spikes per event
spike_rate_per_event = zeros(numel(A),1);

% go through all events
for k = 1:numel(A)

    % get bounding box (left and right time points) for the current event
    range_values = round(A(k).BoundingBox(1):(A(k).BoundingBox(1)+A(k).BoundingBox(3)));

    % compute the number of inferred spikes (sum over the detected event,
    % extended by 2 time points to the left and right)
    spike_rate_per_event(k) = sum(spike_rates_GC8(range_values(1)-2:range_values(end)+2));
    
    % this is just for visualization
    event_detection_scaled(range_values) =  event_detection_scaled(range_values).*spike_rate_per_event(k);

end

figure, plot(spike_rates_GC8); hold on; plot(event_detection_scaled,'r')
plot(ground_truth-1,'b');

figure, hist(spike_rate_per_event,200)

figure(4), subplot(1,2,1); plot(spike_rates_GC8,'r','LineWidth',3); hold on; plot(ground_truth,'b'); xlim([2000 4000])
 subplot(1,2,2); plot(spike_rates_GC8/2.5,'r','LineWidth',3); hold on; plot(ground_truth,'b'); xlim([2000 4000])


