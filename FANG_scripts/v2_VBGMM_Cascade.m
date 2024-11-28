%% Apply VB-GMM for all Datasets

% navigate to the main folder
cd('CASCADE/Results_for_autocalibration')
dataset_folder = 'DS32-GCaMP8s-m-V1';
dataset_name = 'GCaMP8s';
cd(dataset_folder)

% find all neuron files
neuron_files = dir('CAttached*.mat');

% initialize arrays to store results
all_metrics = [];
all_unitary_amplitudes = [];
all_component_counts = [];
all_component_weights = {};
all_component_means = {};

% initialize improvement metrics
all_raw_metrics = [];
all_corrected_metrics = [];
all_improvement_ratios = [];
neuron_quality = struct('good', [], 'moderate', [], 'poor', []);

fprintf('\nStarting analysis of %d files...\n', length(neuron_files));

% process each neuron file
for file_idx = 1:length(neuron_files)
    fprintf('\nProcessing file %d/%d: %s\n', file_idx, length(neuron_files), neuron_files(file_idx).name);
    
    % load the dataset
    load(neuron_files(file_idx).name);
    
    % threshold deconvolved trace
    event_detection = spike_rates_GC8 > 0.3;
    
    % detect contiguous events
    labels = bwlabel(event_detection);
    A = regionprops(labels);
    
    % initialize matrices for current file
    spike_rate_per_event = zeros(numel(A), 1);
    spike_rate_per_event_GT = zeros(numel(A), 1);
    
    % process all events in current file
    for k = 1:numel(A)

        % get bounding box for the current event
        range_values = round(A(k).BoundingBox(1):(A(k).BoundingBox(1)+A(k).BoundingBox(3)));
        
        % ensure indices are within bounds
        left_idx = max(1, range_values(1)-2);
        right_idx = min(length(spike_rates_GC8), range_values(end)+2);
        
        % compute number of inferred spikes
        spike_rate_per_event(k) = sum(spike_rates_GC8(left_idx:right_idx));
        spike_rate_per_event_GT(k) = sum(ground_truth(left_idx:right_idx)); 
    end
    
    % VBGMM fitting
    try
        [unitary_amp, model] = analyzeSpikesVBGMM(spike_rate_per_event);
        all_unitary_amplitudes(file_idx) = unitary_amp;
        
        % store component information
        active_idx = model.PComponents > 0.09;
        all_component_counts(file_idx) = sum(active_idx);
        all_component_weights{file_idx} = model.PComponents(active_idx);
        all_component_means{file_idx} = model.mu(active_idx);
        
        % calculate improvements
        corrected_spike_rate = spike_rate_per_event/unitary_amp;
        
        % find single spike events
        single_spike_events = abs(spike_rate_per_event_GT - 1) < 0.5;
        
        % calculate metrics
        if sum(single_spike_events) > 0
            raw_metric = nanmedian(spike_rate_per_event(single_spike_events)) - 1;
            corrected_metric = nanmedian(corrected_spike_rate(single_spike_events)) - 1;
            
            all_raw_metrics(file_idx) = raw_metric;
            all_corrected_metrics(file_idx) = corrected_metric;
            all_improvement_ratios(file_idx) = abs(corrected_metric)/abs(raw_metric);
            
            % classify neuron quality based on corrected metric
            if abs(corrected_metric) < 0.05
                neuron_quality.good = [neuron_quality.good, file_idx];
            elseif abs(corrected_metric) < 0.1
                neuron_quality.moderate = [neuron_quality.moderate, file_idx];
            else
                neuron_quality.poor = [neuron_quality.poor, file_idx];
            end
        end
        
    catch ME
        fprintf('Warning: Analysis failed for file %d: %s\n', file_idx, ME.message);
        all_unitary_amplitudes(file_idx) = NaN;
        all_raw_metrics(file_idx) = NaN;
        all_corrected_metrics(file_idx) = NaN;
        all_improvement_ratios(file_idx) = NaN;
        all_component_counts(file_idx) = NaN;
        all_component_weights{file_idx} = {};
        all_component_means{file_idx} = {};
    end
end

%% Visualization and Statistical Analysis

figure('Position', [100, 100, 1200, 800]);

% Plot 1: Distribution of unitary amplitudes
subplot(2,2,1);
histogram(all_unitary_amplitudes, 'Normalization', 'probability', 'FaceColor', [0.3 0.6 0.9]);
title(sprintf('Distribution of Unitary Amplitudes (%s)', dataset_name));
xlabel('Unitary Amplitude');
ylabel('Probability');
grid on;

% add mean and median lines
hold on;
mean_amp = mean(all_unitary_amplitudes, 'omitnan');
median_amp = median(all_unitary_amplitudes, 'omitnan');
ylim_vals = ylim;
plot([mean_amp mean_amp], ylim_vals, 'r--', 'LineWidth', 2);
plot([median_amp median_amp], ylim_vals, 'g--', 'LineWidth', 2);
legend({'Distribution', 'Mean', 'Median'});

% Plot 2: Raw vs Corrected Metrics
subplot(2,2,2);
boxplot([all_raw_metrics', all_corrected_metrics'], ...
    'Labels', {'Raw', 'Corrected'}, ...
    'Notch', 'on');
title('Raw vs Corrected Metrics');
ylabel('Error from Ground Truth');
grid on;

% Plot 3: Improvement Distribution
subplot(2,2,3);
histogram(all_improvement_ratios, 20, 'FaceColor', [0.3 0.6 0.9]);
title('Distribution of Improvement Ratios');
xlabel('Improvement Ratio');
ylabel('Count');
grid on;

% Plot 4: Quality Distribution
subplot(2,2,4);
quality_counts = [length(neuron_quality.good), ...
                 length(neuron_quality.moderate), ...
                 length(neuron_quality.poor)];
bar(quality_counts);
title('Neuron Quality Distribution');
xticklabels({'Good', 'Moderate', 'Poor'});
ylabel('Number of Neurons');
grid on;


% Print Analysis Summary

fprintf('\n=== Analysis Summary for %s ===\n', dataset_name);
fprintf('Total number of neurons analyzed: %d\n', length(neuron_files));
fprintf('Successfully analyzed neurons: %d\n', sum(~isnan(all_unitary_amplitudes)));

% unitary amplitude statistics
fprintf('\n--- Unitary Amplitude Statistics ---\n');
fprintf('Mean: %.4f\n', mean_amp);
fprintf('Median: %.4f\n', median_amp);
fprintf('Standard deviation: %.4f\n', std(all_unitary_amplitudes, 'omitnan'));
fprintf('Range: %.4f to %.4f\n', min(all_unitary_amplitudes), max(all_unitary_amplitudes));

% improvement statistics
fprintf('\n--- Improvement Statistics ---\n');
fprintf('Raw Error (median ± std): %.4f ± %.4f\n', ...
    median(all_raw_metrics, 'omitnan'), std(all_raw_metrics, 'omitnan'));
fprintf('Corrected Error (median ± std): %.4f ± %.4f\n', ...
    median(all_corrected_metrics, 'omitnan'), std(all_corrected_metrics, 'omitnan'));
fprintf('Median improvement ratio: %.2f\n', median(all_improvement_ratios, 'omitnan'));
fprintf('Neurons with improvement: %.1f%%\n', ...
    100 * sum(all_improvement_ratios < 1)/sum(~isnan(all_improvement_ratios)));

% quality distribution
fprintf('\n--- Quality Distribution ---\n');
fprintf('Good quality neurons (error < 0.05): %d (%.1f%%)\n', ...
    length(neuron_quality.good), ...
    100 * length(neuron_quality.good)/sum(~isnan(all_raw_metrics)));
fprintf('Moderate quality neurons (0.05 ≤ error < 0.1): %d (%.1f%%)\n', ...
    length(neuron_quality.moderate), ...
    100 * length(neuron_quality.moderate)/sum(~isnan(all_raw_metrics)));
fprintf('Poor quality neurons (error ≥ 0.1): %d (%.1f%%)\n', ...
    length(neuron_quality.poor), ...
    100 * length(neuron_quality.poor)/sum(~isnan(all_raw_metrics)));


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
        options.expected_unit_amp = 1.3013; % priors from deconvolution and gradient descent
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

