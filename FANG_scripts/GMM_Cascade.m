%% GMM

% 1. Extract calcium trace and ground truth such that it can be inserted into Cascade 
% 2. Apply Cascade
% 3. Use the output of Cascade to detect events
% 4. From the detected events, make a histogram
% 5. Automatically (or visually) define unitary events (i.e., the smallest observed events)!
% 6. Re-scale inferred spike rate with this scaling factor!


% load the dataset
load('CAttached_jGCaMP8s_472182_7_mini')

% threshold deconvolved trace --> threshold not optimized!
event_detection = spike_rates_GC8 > 0.3;

% copy variable and convert from boolean to double
event_detection_scaled = double(event_detection);

% detect contiguous events --> same as before
labels = bwlabel(event_detection);
A = regionprops(labels);

% initialize matrix that will contain the inferred number of spikes per event
spike_rate_per_event = zeros(numel(A),1);
spike_rate_per_event_GT = zeros(numel(A),1);

% go through all events
for k = 1:numel(A)

    % get bounding box (left and right time points) for the current event
    range_values = round(A(k).BoundingBox(1):(A(k).BoundingBox(1)+A(k).BoundingBox(3)));

    % compute the number of inferred spikes (sum over the detected event,
    % extended by 2 time points to the left and right)
    spike_rate_per_event(k) = sum(spike_rates_GC8(range_values(1)-2:range_values(end)+2));
    spike_rate_per_event_GT(k) = sum(ground_truth(range_values(1)-2:range_values(end)+2));
    
    % this is just for visualization
    event_detection_scaled(range_values) =  event_detection_scaled(range_values).*spike_rate_per_event(k);

end

%[unitary_amp, calib_factor] = analyze_events_gmm(spike_rates_GC8);

analyzeSpikesGMM(spike_rate_per_event);

%{
figure, plot(spike_rate_per_event,spike_rate_per_event_GT,'.')

figure, plot(spike_rates_GC8); hold on; plot(event_detection_scaled,'r')
plot(ground_truth-1,'b');

figure, subplot(1,2,1); plot(spike_rates_GC8,'r','LineWidth',3); hold on; plot(ground_truth,'b'); xlim([2000, 4000]);
 subplot(1,2,2); plot(spike_rates_GC8/2.5,'r','LineWidth',3); hold on; plot(ground_truth,'b'); xlim([2000 4000])

figure, hist(spike_rate_per_event,200)


%% Improvements?

% try to improve with the estimated amplitude
amplitude_GMM = 10.87;
corrected_spike_rate_per_event = spike_rate_per_event/amplitude_GMM;

detect_events = find(abs(spike_rate_per_event_GT - 1) < 0.5);

% quantative accuracy
metric = nanmedian(corrected_spike_rate_per_event(detect_events)) - 1;
%}

%% GMM Functions

function analyzeSpikesGMM(spike_rate_per_event)

    % fit GMM to analyze spike rate distribution
    X = spike_rate_per_event(:);
    
    % parameters
    max_components = 3;
    BIC = zeros(max_components-1, 1);
    GMModels = cell(max_components-1, 1);
    
    % find optimal number of components using BIC
    for k = 1:max_components
        GMModels{k} = fitgmdist(X, k, 'Replicates', 10);
        BIC(k) = GMModels{k}.BIC;
    end
    
    [~, optimal_k] = min(BIC);
    gmm_model = GMModels{optimal_k};
    
    % sort components by mean
    [sorted_means, sort_idx] = sort(gmm_model.mu);
    sorted_weights = gmm_model.ComponentProportion(sort_idx);
    sorted_sigmas = sqrt(squeeze(gmm_model.Sigma(:,:,sort_idx)));
    
    % calculate ratios between consecutive means
    ratios = sorted_means(2:end) ./ sorted_means(1);
    
    % Visualization
    figure('Position', [100, 100, 1200, 800]);
    
    % Subplot 1: Histogram with GMM fit
    subplot(2,1,1);
    histogram(X, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.3);
    hold on;
    
    x = linspace(min(X), max(X), 200);
    y_total = zeros(size(x));
    colors = {'r', 'g', 'b', 'm'};
    legend_entries = {'Data'};
    
    % plot individual components
    for i = 1:optimal_k
        y_i = sorted_weights(i) * normpdf(x, sorted_means(i), sorted_sigmas(i));
        plot(x, y_i, colors{i}, 'LineWidth', 2);
        y_total = y_total + y_i;
        legend_entries{end+1} = sprintf('Component %d (μ=%.2f)', i, sorted_means(i));
    end
    
    % plot sum of components
    plot(x, y_total, 'k--', 'LineWidth', 2);
    legend_entries{end+1} = 'Sum of Components';
    
    title('CASCADE Spike Rate Distribution with GMM Fit');
    xlabel('Spike Rate per Event');
    ylabel('Probability Density');
    legend(legend_entries, 'Location', 'best');
    grid on;
    
    % Subplot 2: Component analysis
    subplot(2,1,2);
    bar(1:length(sorted_means), sorted_means, 'FaceColor', [0.8 0.8 0.8]);
    hold on;
    
    % plot ideal integer multiples of first component
    ideal_multiples = sorted_means(1) * (1:length(sorted_means));
    plot(1:length(sorted_means), ideal_multiples, 'r--o', 'LineWidth', 2);
    
    title('Analysis of GMM Components');
    xlabel('Component Number');
    ylabel('Mean Value');
    legend('Actual Component Means', 'Ideal Integer Multiples');
    grid on;
    
    % Print analysis
    fprintf('Number of components: %d\n', optimal_k);
    fprintf('Component means: ');
    fprintf('%.2f ', sorted_means);
    fprintf('\nComponent weights: ');
    fprintf('%.2f ', sorted_weights);
    fprintf('\n\nDeviation from integer multiples:\n');
    for i = 2:length(sorted_means)
        expected = i;
        actual = ratios(i-1);
        fprintf('Component %d: Expected ratio = %d, Actual ratio = %.2f, Error = %.1f%%\n', ...
            i, expected, actual, abs(actual-expected)/expected*100);
    end
end
