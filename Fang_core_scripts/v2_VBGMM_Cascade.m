%% Apply VB-GMM for GCaMP8


% preparations
cd('CASCADE/Results_for_autocalibration')
dataset_folder = 'DS31-GCaMP8m-m-V1';
dataset_name = 'GCaMP8m';
cd(dataset_folder)
neuron_files = dir('CAttached*.mat');

% initialize arrays
all_metrics = [];
all_unitary_amplitudes = [];
all_component_counts = [];
all_component_weights = {};
all_component_means = {};

all_raw_metrics = [];
all_optimized_metrics = [];
all_improvement_ratios = [];
neuron = struct('good', [], 'moderate', [], 'poor', []);


for file_idx = 1:length(neuron_files)
    fprintf('\nProcessing file %d/%d: %s\n', file_idx, length(neuron_files), neuron_files(file_idx).name);
    
    load(neuron_files(file_idx).name);
    
    % threshold deconvolved trace
    event_detection = spike_rates_GC8 > 0.3;
    
    % detect contiguous events
    labels = bwlabel(event_detection);
    A = regionprops(labels);
    
    % initialize matrices 
    spike_rate_per_event = zeros(numel(A), 1);
    spike_rate_per_event_GT = zeros(numel(A), 1);

    % process current file
    for k = 1:numel(A)

        % get bounding box (left and right time points) for the current event
        range_values = round(A(k).BoundingBox(1):(A(k).BoundingBox(1)+A(k).BoundingBox(3)));

        % compute the number of inferred spikes (sum over the detected event,
        % extended by 2 time points to the left and right)
        spike_rate_per_event(k) = sum(spike_rates_GC8(range_values(1)-2:range_values(end)+2));
        spike_rate_per_event_GT(k) = sum(ground_truth(range_values(1)-2:range_values(end)+2)); 
    end


    % VBGMM fitting
    try
        [unitary_amplitude, model] = AnalyzeSpikesVBGMM(spike_rate_per_event);
        all_unitary_amplitudes(file_idx) = unitary_amplitude;
        
        % store component info
        active_idx = model.PComponents > 0.09;
        all_component_counts(file_idx) = sum(active_idx);
        all_component_weights{file_idx} = model.PComponents(active_idx);
        all_component_means{file_idx} = model.mu(active_idx);
        
        % calculate optimization
        optimized_spike_rate = spike_rate_per_event/unitary_amplitude;
        
        % find single spike events
        detect_events = abs(spike_rate_per_event_GT - 1) < 0.5;

        % calculate metrics
        if sum(detect_events) > 0
            raw_metric = nanmedian(spike_rate_per_event(detect_events)) - 1;
            optimized_metric = nanmedian(optimized_spike_rate(detect_events)) - 1;
            
            all_raw_metrics(file_idx) = raw_metric;
            all_optimized_metrics(file_idx) = optimized_metric;
            all_improvement_ratios(file_idx) = abs(optimized_metric)/abs(raw_metric);
            
            % classify neuron based on requirements
            if abs(optimized_metric) < 0.05
                neuron.good = [neuron.good, file_idx];
            elseif abs(optimized_metric) < 0.1
                neuron.moderate = [neuron.moderate, file_idx];
            else
                neuron.poor = [neuron.poor, file_idx];
            end

        end

    catch ER
        fprintf('Warning: Analysis failed for file %d: %s\n', file_idx, ER.message);
        all_unitary_amplitudes(file_idx) = NaN;
        all_raw_metrics(file_idx) = NaN;
        all_optimized_metrics(file_idx) = NaN;
        all_improvement_ratios(file_idx) = NaN;
        all_component_counts(file_idx) = NaN;
        all_component_weights{file_idx} = NaN;
        all_component_means{file_idx} = NaN;
    end

end


% Visualization and Statistical Analysis

figure('Position', [100, 100, 1000, 1000]);

% plot 1: distribution of unitary amplitudes
subplot(2,2,1);
histogram(all_unitary_amplitudes, 'Normalization', 'probability', 'FaceColor', [0.4 0.6 0.8]);
title(sprintf('Distribution of Unitary Amplitudes (%s)', dataset_name));
xlabel('Unitary Amplitude');
ylabel('Probability');
grid on;

% add mean and median lines
hold on;
mean_amplitudes = mean(all_unitary_amplitudes, 'omitnan');
median_amplitudes = median(all_unitary_amplitudes, 'omitnan');
ylims = ylim;
plot([mean_amplitudes mean_amplitudes], ylims, 'r--', 'LineWidth', 1);
plot([median_amplitudes median_amplitudes], ylims, 'g--', 'LineWidth', 1);
legend({'Distribution', 'Mean', 'Median'});

% plot 2: raw and optimized metrics
subplot(2,2,2);
boxplot([all_raw_metrics', all_optimized_metrics'],'Labels', {'Raw', 'Optimized'},'Notch', 'on');
title('Raw vs Optimized Metrics');
ylabel('Error Calculated from Ground Truth');
grid on;

% plot 3: optimization distribution
subplot(2,2,3);
histogram(all_improvement_ratios, 10, 'FaceColor', [0.4 0.6 0.8]);
title('Distribution of Optimization Ratios');
xlabel('Optimization Ratio');
ylabel('Count');
grid on;

% plot 4: distribution of all neurons
subplot(2,2,4);
result_counts = [length(neuron.good),length(neuron.moderate),length(neuron.poor)];
b = bar(result_counts, 'FaceColor', [0.4 0.6 0.8]);
title('Neuron Result Distribution');
xticklabels({'Good', 'Moderate', 'Poor'});
ylabel('Number of Neurons');
grid on;


% print analysis summary
fprintf('\n$$$ Analysis Summary for %s $$$\n', dataset_name);
fprintf('Total number of neurons analyzed: %d\n', length(neuron_files));
fprintf('Successfully analyzed neurons: %d\n', sum(~isnan(all_unitary_amplitudes)));

% unitary amplitude stats
fprintf('\n[Unitary Amplitude Statistics]\n');
fprintf('Mean: %.4f\n', mean_amplitudes);
fprintf('Median: %.4f\n', median_amplitudes);
fprintf('Standard deviation: %.4f\n', std(all_unitary_amplitudes, 'omitnan'));
fprintf('Range: %.4f to %.4f\n', min(all_unitary_amplitudes), max(all_unitary_amplitudes));

% improvement stats
fprintf('\n[Improvement Statistics]\n');
fprintf('Raw Error (median ± std): %.4f ± %.4f\n', ...
    median(all_raw_metrics, 'omitnan'), std(all_raw_metrics, 'omitnan'));
fprintf('optimized Error (median ± std): %.4f ± %.4f\n', ...
    median(all_optimized_metrics, 'omitnan'), std(all_optimized_metrics, 'omitnan'));
fprintf('Neurons with improvement: %.1f%%\n', ...
    100 * sum(all_improvement_ratios < 1)/sum(~isnan(all_improvement_ratios)));

% result distribution
fprintf('\n[Result Distribution]\n');
fprintf('Good result neurons (error < 0.05): %d (%.1f%%)\n', ...
    length(neuron.good),100 * length(neuron.good)/sum(~isnan(all_raw_metrics)));
fprintf('Moderate result neurons (0.05 ≤ error < 0.1): %d (%.1f%%)\n', ...
    length(neuron.moderate),100 * length(neuron.moderate)/sum(~isnan(all_raw_metrics)));
fprintf('Poor result neurons (error ≥ 0.1): %d (%.1f%%)\n', ...
    length(neuron.poor),100 * length(neuron.poor)/sum(~isnan(all_raw_metrics)));


%% Function

function [unitary_amplitude, vb_gmm_model] = AnalyzeSpikesVBGMM(spike_rate, options)

    % default options based on our prior knowledge
    if ~exist('options', 'var')
        options = struct();
    end
    
    options = SetDefault(options);
    
    % random seed for reproducibility
    rng(2024);
    
    % load data and handle NaN
    X = spike_rate(~isnan(spike_rate));
    
    % initialize model with priors
    [F, max_components] = InitializeModel(X, options);
    
    % fit model with all initialization
    [best_model, best_ll] = FitModel(X, F, max_components, options);
    
    % analyze results
    [unitary_amplitude, vb_gmm_model] = AnalyzeResult(best_model, X, options);

    Visualization(X, vb_gmm_model, unitary_amplitude, options);
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
        options.expected_unit_amp = 1.3013; % 1.7399 for 8s, 1.3013 for 8m
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
    fprintf('\nVB-GMM Analysis Results:\n');
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
    
    % generate points of plotting
    x = linspace(min(X_plot), max(X_plot), 200);
    y_total = zeros(size(x));
    
    % get components
    weights = model.PComponents;
    means = model.mu;
    sigmas = sqrt(squeeze(model.Sigma));
    good_idx = weights > options.weight_threshold;
    
    % plot components
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
    
    % plot sum of components
    plot(x, y_total, 'k--', 'LineWidth', 2, 'DisplayName', 'Sum of Components');
    title('Spike Rate Distribution (VB-GMM Fit)');
    xlabel('Spike Rate');
    ylabel('Probability Density');
    legend('show', 'Location', 'best');
    grid on;
    
    % plot 2: component analysis
    subplot(2,1,2);
    
    % get sorted good components
    good_means = means(good_idx);
    [sorted_means, sort_idx] = sort(good_means);
    
    % bar plot of component means
    bar(sorted_means, 'FaceColor', [0.4 0.6 0.8]);
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

