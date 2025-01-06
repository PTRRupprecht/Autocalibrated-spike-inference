%% Optimization

% try to improve inference with amplitude
unitary_amplitude = 1.2001;
corrected_spike_rate_per_event = spike_rate_per_event/unitary_amplitude;

detect_events = find(abs(spike_rate_per_event_GT - 1) < 0.5);

% quantative accuracy
raw_metric = nanmedian(spike_rate_per_event(detect_events)) - 1;
raw_metric_log = nanmedian(log(spike_rate_per_event(detect_events)));
metric = nanmedian(corrected_spike_rate_per_event(detect_events)) - 1;
metric_log = nanmedian(log(corrected_spike_rate_per_event(detect_events)));

figure; 
plot(spike_rate_per_event_GT,'r'); 
hold on; 
plot(spike_rate_per_event,'b'); 
plot(corrected_spike_rate_per_event,'g');
xlabel('Time(s)');
ylabel('\DeltaF/F');

%% Variational Bayesian GMM for a single neuron

% load the dataset
load('CAttached_jGCaMP8s_472181_1_mini')

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

end

%figure, histogram(spike_rate_per_event, 100);
%figure, histogram(spike_rate_per_event_GT, 100);
[unitary_amplitude, hb_gmm_model] = AnalyzeSpikesHBGMM(spike_rate_per_event);


%% Function

function [unitary_amplitude, hb_gmm_model] = AnalyzeSpikesHBGMM(spike_rate, options)

    % default options based on our prior knowledge
    if ~exist('options', 'var')
        options = struct();
    end
    
    options = SetDefault(options);
    
    % load data and handle NaN
    X = spike_rate(~isnan(spike_rate));
    
    % initialize model with priors
    [F, max_components] = InitializeModel(X, options);
    
    % fit model with all initialization
    [best_model, best_ll] = FitModel(X, F, max_components, options);
    
    % analyze results
    [unitary_amplitude, hb_gmm_model] = AnalyzeResult(best_model, X, options);

    Visualization(X, hb_gmm_model, unitary_amplitude, options);
end


% set defaults based on priors
function options = SetDefault(options)
    
    % maximum number of spikes typically can be seen in one event
    if ~isfield(options, 'max_spikes')
        options.max_spikes = 8;
    end
    
    % typical variance scaling with amplitude
    if ~isfield(options, 'variance_scaling')
        options.variance_scaling = 0.2; % Variance increases with mean
    end
    
    % component weights (single spikes more common than doubles, doubles more than triples...)
    if ~isfield(options, 'component_prior')
        options.component_prior = exp(-(0:7));
        options.component_prior = options.component_prior / sum(options.component_prior);
    end
    
    % minimum weight for a component
    if ~isfield(options, 'weight_threshold')
        options.weight_threshold = 0.09;
    end

    % priors combine gradient descent modeling and deconvolution
    if ~isfield(options, 'expected_unit_amp')
        options.expected_unit_amp = 1.5;
    end

end


function [F, max_components] = InitializeModel(X, options)

    % apply prior knowledge
    max_components = options.max_spikes;
    initial_unit = options.expected_unit_amp;

    % use the median of the smallest 25% of positive events as initial estimate (abandon after using priors)
    %initial_unit = median(X(X < prctile(X, 25)));
    
    % initialize means at integer multiples
    F.mu = initial_unit * (1:max_components)';
    
    % initialize covariances
    F.Sigma = zeros(1,1,max_components);
    
    for i = 1:max_components

        % variance increases with mean
        F.Sigma(1,1,i) = (options.variance_scaling * F.mu(i))^2;
    end
    
    % initialize weights with priors too
    F.PComponents = options.component_prior(:);
end


function [best_model, best_ll] = FitModel(X, F, max_components, options)
    n_replicates = 20;
    best_ll = inf;
    best_model = [];
    
    % fit model
    for rep = 1:n_replicates
        try
            model = fitgmdist(X, max_components, ...
                'CovarianceType', 'diagonal', ...
                'RegularizationValue', 1e-6, ...
                'Options', statset('MaxIter', 1000, 'TolFun', 1e-6), ...
                'Start', F);
            
            if model.NegativeLogLikelihood < best_ll
                best_ll = model.NegativeLogLikelihood;
                best_model = model;
            end

        catch ER
            fprintf('Warning: Iteration %d failed: %s\n', rep, ER.message);
            continue;
        end
    end
    
    if isempty(best_model)
        error('Failed to fit model after all replicates');
    end
end


function [unitary_amplitude, vb_gmm_model] = AnalyzeResult(model, X, options)

    % component parameters
    weights = model.PComponents;
    means = model.mu;
    
    % filter components
    good_idx = weights > options.weight_threshold;
    good_means = means(good_idx);
    good_weights = weights(good_idx);
    
    % sort components by mean
    [sorted_means, sort_idx] = sort(good_means);
    sorted_weights = good_weights(sort_idx);
    
    % first component mean as unitary amplitude estimate
    unitary_amplitude = sorted_means(1);
    vb_gmm_model = model;
    
    % print analysis
    fprintf('\nHB-GMM Analysis Results:\n');
    fprintf('Number of effective components: %d\n', sum(good_idx));
    fprintf('Unitary amplitude estimate: %.3f\n', unitary_amplitude);
    
    fprintf('\nComponent analysis:\n');
    for i = 1:length(sorted_means)
        expected = i;
        actual = sorted_means(i)/unitary_amplitude;
        error = abs(actual - expected)/expected * 100;
        fprintf(['Component %d: Mean=%.3f, Weight=%.3f\n' ...
                '   Expected ratio=%.1f, Actual ratio=%.2f (error=%.1f%%)\n'],i, ...
                sorted_means(i), sorted_weights(i), expected, actual, error);
    end
end


function Visualization(X, model, unitary_amplitude, options)

    figure('Position', [100, 100, 1200, 800]);
    
    % plot 1: data and fitted components
    subplot(2,1,1);
    X_plot = X(X > 0);
    histogram(X_plot, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.3);
    hold on;
    
    x = linspace(min(X_plot), max(X_plot), 200);
    y_total = zeros(size(x));
    weights = model.PComponents;
    means = model.mu;
    sigmas = sqrt(squeeze(model.Sigma));
    good_idx = weights > options.weight_threshold;
    cmap = lines(sum(good_idx));
    component_idx = 1;
    
    for i = 1:length(weights)
        if weights(i) > options.weight_threshold
            y_i = weights(i) * normpdf(x, means(i), sigmas(i));
            plot(x, y_i, 'Color', cmap(component_idx,:), 'LineWidth', 2, ...
                'DisplayName', sprintf('Component %d (μ=%.2f, w=%.2f)', ...
                component_idx, means(i), weights(i)));
            y_total = y_total + y_i;
            component_idx = component_idx + 1;
        end
    end
    
    plot(x, y_total, 'k--', 'LineWidth', 2, 'DisplayName', 'Sum of Components');
    title('Spike Rate Distribution (HB-GMM Fit)');
    xlabel('Spike Rate');
    ylabel('Probability Density');
    text(-0.05, 1.05, 'A', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
    legend('show', 'Location', 'best');
    grid on;
    
    % plot 2: component analysis
    subplot(2,1,2);
    good_means = means(good_idx);
    [sorted_means, sort_idx] = sort(good_means);
    bar(sorted_means, 'FaceColor', [0.4 0.6 0.8]);
    hold on;
    
    % plot expected integer multiples
    expected_means = unitary_amplitude * (1:length(sorted_means));
    plot(1:length(sorted_means), expected_means, 'r--o', 'LineWidth', 2);
    
    title('Component Mean Analysis');
    xlabel('Component Number');
    ylabel('Mean Value');
    text(-0.05, 1.05, 'B', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Actual Means', 'Expected Integer Multiples');
    grid on;
end

