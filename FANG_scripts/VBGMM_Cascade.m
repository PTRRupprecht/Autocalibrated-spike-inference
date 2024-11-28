%% Improvement

% try to improve inference with amplitude
unitary_amplitude = 1.1416;
corrected_spike_rate_per_event = spike_rate_per_event/unitary_amplitude;

detect_events = find(abs(spike_rate_per_event_GT - 1) < 0.5);

% quantative accuracy
metric = nanmedian(corrected_spike_rate_per_event(detect_events)) - 1;


%% Variational Bayesian GMM

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

end

%figure, histogram(spike_rate_per_event, 100);
%figure, histogram(spike_rate_per_event_GT, 100);
[unitary_amp, model] = analyzeSpikesVBGMM(spike_rate_per_event);


%% Function

function [unitary_amplitude, vbgmm_model] = analyzeSpikesVBGMM(spike_rate, options)

    % default options based on prior knowledge
    if ~exist('options', 'var')
        options = struct();
    end
    
    options = setDefaultOptions(options);
    
    % set random seed for reproducibility
    rng(2024);
    
    % load data and handle NaN
    X = spike_rate(~isnan(spike_rate));
    
    % initialize model with priors
    [S, max_components] = initializeModel(X, options);
    
    % fit model with multiple initializations
    [best_model, best_ll] = fitModelWithReplicates(X, S, max_components, options);
    
    % analyze results
    [unitary_amplitude, vbgmm_model] = analyzeResults(best_model, X, options);
    
    % visualize results
    visualizeResults(X, vbgmm_model, unitary_amplitude, options);
end

% Set defaults based on prior knowledge
function options = setDefaultOptions(options)
    
    % maximum number of spikes typically seen in one event
    if ~isfield(options, 'max_spikes')
        options.max_spikes = 8;
    end
    
    % typical variance scaling with amplitude
    if ~isfield(options, 'variance_scaling')
        options.variance_scaling = 0.2; % Variance increases with mean
    end
    
    % component weights (single spikes more common than doubles, doubles more than triples, etc.)
    if ~isfield(options, 'component_prior')
        options.component_prior = exp(-(0:7));
        options.component_prior = options.component_prior / sum(options.component_prior);
    end
    
    % minimum weight for considering a component active
    if ~isfield(options, 'weight_threshold')
        options.weight_threshold = 0.09;
    end

    % priors combine gradient descent modeling and deconvolution
    if ~isfield(options, 'expected_unit_amp')
        options.expected_unit_amp = 1.3013;
    end

end

function [S, max_components] = initializeModel(X, options)

    % apply prior knowledge
    max_components = options.max_spikes;
    initial_unit = options.expected_unit_amp;

    % use the median of the smallest 25% of positive events as initial estimate
    %initial_unit = median(X(X < prctile(X, 25)));
    
    % initialize means at integer multiples
    S.mu = initial_unit * (1:max_components)';
    
    % initialize covariances with specific scaling
    S.Sigma = zeros(1,1,max_components);
    for i = 1:max_components

        % variance increases with mean
        S.Sigma(1,1,i) = (options.variance_scaling * S.mu(i))^2;
    end
    
    % initialize weights with prior knowledge
    S.PComponents = options.component_prior(:);
end

function [best_model, best_ll] = fitModelWithReplicates(X, S, max_components, options)
    n_replicates = 20;
    best_ll = inf;
    best_model = [];
    
    for rep = 1:n_replicates
        try
            % add small random perturbations to means
            curr_S = S;
            %curr_S.mu = S.mu .* (1 + 0.1*randn(size(S.mu)));
            
            % constrain means to be positive and ordered
            curr_S.mu = sort(max(curr_S.mu, 0));
            
            % fit model with constraints
            model = fitgmdist(X, max_components, ...
                'CovarianceType', 'diagonal', ...
                'RegularizationValue', 1e-6, ...
                'Options', statset('MaxIter', 1000, 'TolFun', 1e-6), ...
                'Start', curr_S);
            
            if model.NegativeLogLikelihood < best_ll
                best_ll = model.NegativeLogLikelihood;
                best_model = model;
            end
        catch ME
            fprintf('Warning: Iteration %d failed: %s\n', rep, ME.message);
            continue;
        end
    end
    
    if isempty(best_model)
        error('Failed to fit model after all replicates');
    end
end

function [unitary_amplitude, vbgmm_model] = analyzeResults(model, X, options)

    % get component parameters
    weights = model.PComponents;
    means = model.mu;
    
    % find active components
    active_idx = weights > options.weight_threshold;
    active_means = means(active_idx);
    active_weights = weights(active_idx);
    
    % sort components by mean
    [sorted_means, sort_idx] = sort(active_means);
    sorted_weights = active_weights(sort_idx);
    
    % first component mean as unitary amplitude estimate
    unitary_amplitude = sorted_means(1);
    vbgmm_model = model;
    
    % Print analysis
    fprintf('\nVB-GMM Analysis Results:\n');
    fprintf('Number of effective components: %d\n', sum(active_idx));
    fprintf('Unitary amplitude estimate: %.3f\n', unitary_amplitude);
    
    fprintf('\nComponent analysis:\n');
    for i = 1:length(sorted_means)
        expected = i;
        actual = sorted_means(i)/unitary_amplitude;
        error = abs(actual - expected)/expected * 100;
        fprintf(['Component %d: Mean=%.3f, Weight=%.3f\n' ...
                '   Expected ratio=%.1f, Actual ratio=%.2f (error=%.1f%%)\n'], ...
            i, sorted_means(i), sorted_weights(i), expected, actual, error);
    end
end

function visualizeResults(X, model, unitary_amplitude, options)

    % create figure
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Data and fitted components
    subplot(2,1,1);
    
    % plot histogram of positive values only
    X_plot = X(X > 0);
    histogram(X_plot, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.3);
    hold on;
    
    % generate points for plotting the GMM
    x = linspace(min(X_plot), max(X_plot), 200);
    y_total = zeros(size(x));
    
    % get active components
    weights = model.PComponents;
    means = model.mu;
    sigmas = sqrt(squeeze(model.Sigma));
    active_idx = weights > options.weight_threshold;
    
    % plot components
    cmap = lines(sum(active_idx));
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
    
    % plot sum of components
    plot(x, y_total, 'k--', 'LineWidth', 2, 'DisplayName', 'Sum of Components');
    
    title('Spike Rate Distribution (VB-GMM Fit)');
    xlabel('Spike Rate');
    ylabel('Probability Density');
    legend('show', 'Location', 'best');
    grid on;
    
    % Plot 2: Component analysis
    subplot(2,1,2);
    
    % get sorted active components
    active_means = means(active_idx);
    [sorted_means, sort_idx] = sort(active_means);
    
    % bar plot of component means
    bar(sorted_means, 'FaceColor', [0.8 0.8 0.8]);
    hold on;
    
    % plot expected integer multiples
    expected_means = unitary_amplitude * (1:length(sorted_means));
    plot(1:length(sorted_means), expected_means, 'r--o', 'LineWidth', 2);
    
    title('Component Mean Analysis');
    xlabel('Component Number');
    ylabel('Mean Value');
    legend('Actual Means', 'Expected Integer Multiples');
    grid on;
end

