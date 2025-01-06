%% Apply (Hybrid)HB-GMM for GCaMP8

% preparations
cd('CASCADE/Results_for_autocalibration')
dataset_folder = 'DS32-GCaMP8s-m-V1';
dataset_name = 'GCaMP8s';
cd(dataset_folder)
neuron_files = dir('CAttached*.mat');


% initialize arrays
all_raw_metrics_log = [];
all_raw_metrics_additive = [];
all_optimized_metrics_log = [];
all_optimized_metrics_additive = [];

all_unitary_amplitudes = [];
all_component_counts = [];
all_component_weights = {};
all_component_means = {};

all_spike_rates = cell(length(neuron_files), 1);
all_spike_rates_GT = cell(length(neuron_files), 1);
all_optimized_rates = cell(length(neuron_files), 1);

for file_idx = 1:length(neuron_files)
    fprintf('\nProcessing file %d/%d: %s\n', file_idx, length(neuron_files), neuron_files(file_idx).name);
    
    load(neuron_files(file_idx).name);
    
    % testing
    %spike_rates_GC8 = spike_rates_GC8 * 10;
    %ground_truth = ground_truth * 10;

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


    % HBGMM fitting
    try
        [unitary_amplitude, model] = AnalyzeSpikesHBGMM(spike_rate_per_event);
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

            % additive metrics (absolute deviations)
            raw_metric_additive = nanmedian(spike_rate_per_event(detect_events)) - 1;
            optimized_metric_additive = nanmedian(optimized_spike_rate(detect_events)) - 1;
    
            % logarithmic metrics (relative scale of errors)
            raw_metric_log = nanmedian(log(spike_rate_per_event(detect_events)));
            optimized_metric_log = nanmedian(log(optimized_spike_rate(detect_events)));
    
            % Store both sets of metrics
            all_raw_metrics_additive(file_idx) = raw_metric_additive;
            all_optimized_metrics_additive(file_idx) = optimized_metric_additive;
            all_raw_metrics_log(file_idx) = raw_metric_log;
            all_optimized_metrics_log(file_idx) = optimized_metric_log;

            all_spike_rates{file_idx} = spike_rate_per_event;
            all_spike_rates_GT{file_idx} = spike_rate_per_event_GT;
            all_optimized_rates{file_idx} = optimized_spike_rate;
            
        end


    catch ER
        fprintf('Warning: Analysis failed for file %d: %s\n', file_idx, ER.message);
        
        all_raw_metrics_log(file_idx) = NaN;
        all_raw_metrics_additive(file_idx) = NaN;
        all_optimized_metrics_log(file_idx) = NaN;
        all_optimized_metrics_additive(file_idx) = NaN;
        all_unitary_amplitudes(file_idx) = NaN;
        all_component_counts(file_idx) = NaN;
        all_component_weights{file_idx} = NaN;
        all_component_means{file_idx} = NaN;
    end
end



% Visualization and Statistical Analysis
figure('Position', [100, 100, 1000, 1000]);


% plot 1 & 2: additive and log metrics comparison
subplot(2,2,1);
boxplot([all_raw_metrics_additive', all_optimized_metrics_additive'], ...
    'Labels', {'Raw', 'Optimized'}, 'Notch', 'on');
title('Additive Metrics: Raw vs Optimized');
ylabel('Additive Error');
text(-0.1, 1.05, 'A', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
grid on;


% plot 2
subplot(2,2,2);
boxplot([all_raw_metrics_log', all_optimized_metrics_log'], ...
    'Labels', {'Raw', 'Optimized'}, 'Notch', 'on');
title('Logarithmic Metrics: Raw vs Optimized');
ylabel('Logarithmic Error');
text(-0.1, 1.05, 'B', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
grid on;



% plot 3 & 4: scatter of metrics
subplot(2,2,3);
scatter(all_raw_metrics_additive, all_optimized_metrics_additive, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
xlabel('Raw Additive Error');
ylabel('Optimized Additive Error');
text(-0.1, 1.05, 'C', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
title('Additive Error Comparison')
% add lines and shadow
hold on;
min_value = min(min(all_raw_metrics_additive), min(all_optimized_metrics_additive));
max_value = max(max(all_raw_metrics_additive), max(all_optimized_metrics_additive));
abs_max = max(abs([min_value, max_value]));
plot([-abs_max, abs_max], [-abs_max, abs_max], 'r--', 'LineWidth', 1);
plot([-abs_max, abs_max], [abs_max, -abs_max], 'r--', 'LineWidth', 1);
x = linspace(-abs_max, abs_max, 100);
y1 = x;
y2 = -x;
fill([x fliplr(x)], [y1 fliplr(y2)], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold off;


% plot 4
subplot(2,2,4);
scatter(all_raw_metrics_log, all_optimized_metrics_log, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
xlabel('Raw Logarithmic Error');
ylabel('Optimized Logarithmic Error');
text(-0.1, 1.05, 'D', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
title('Logarithmic Error Comparison');
hold on;
min_value = min(min(all_raw_metrics_log), min(all_optimized_metrics_log));
max_value = max(max(all_raw_metrics_log), max(all_optimized_metrics_log));
abs_max = max(abs([min_value, max_value]));
plot([-abs_max, abs_max], [-abs_max, abs_max], 'r--', 'LineWidth', 1);
plot([-abs_max, abs_max], [abs_max, -abs_max], 'r--', 'LineWidth', 1);
x = linspace(-abs_max, abs_max, 100);
y1 = x;
y2 = -x;
fill([x fliplr(x)], [y1 fliplr(y2)], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold off;

% example plot
bad_indices = find(abs(all_optimized_metrics_log) > abs(all_raw_metrics_log)); % or plot the bad one
good_results = length(neuron_files) - length(bad_indices);

figure;
plot(all_spike_rates_GT{1},'r'); 
hold on; 
plot(all_spike_rates{1},'b'); 
plot(all_optimized_rates{1},'g');
xlabel('Indices');
ylabel('Spike Rate (Hz)');
legend('Ground Truth','CASCADE','HBGMM');


% stats
% print analysis summary
fprintf('\n$$$ Analysis Summary for %s $$$\n', dataset_name);
fprintf('Total number of neurons analyzed: %d\n', length(neuron_files));
fprintf('Successfully analyzed neurons: %d\n', good_results);

% unitary amplitude stats
fprintf('\n[Unitary Amplitude Statistics]\n');
fprintf('Mean: %.4f\n', nanmean(all_unitary_amplitudes));
fprintf('Median: %.4f\n', nanmedian(all_unitary_amplitudes));
fprintf('Standard deviation: %.4f\n', std(all_unitary_amplitudes, 'omitnan'));
fprintf('Range: %.4f to %.4f\n', min(all_unitary_amplitudes), max(all_unitary_amplitudes));

% improvement stats
fprintf('\n[Optimization of all neurons]\n');
fprintf('Additive Metrics:\n');
fprintf('Raw Error (median ± std): %.4f ± %.4f\n', ...
    nanmedian(all_raw_metrics_additive), std(all_raw_metrics_additive, 'omitnan'));
fprintf('Optimized Error (median ± std): %.4f ± %.4f\n', ...
    nanmedian(all_optimized_metrics_additive), std(all_optimized_metrics_additive, 'omitnan'));

fprintf('\nLogarithmic Metrics:\n');
fprintf('Raw Error (mean ± std): %.4f ± %.4f\n', ...
    nanmean(all_raw_metrics_log), std(all_raw_metrics_log, 'omitnan'));
fprintf('Optimized Error (mean ± std): %.4f ± %.4f\n', ...
    nanmean(all_optimized_metrics_log), std(all_optimized_metrics_log, 'omitnan'));


%% Test the performance of prior parameters (unitary amplitdue & variance scaling) within a specific range across all neurons
%%%
%%%
%%%
%%%
TestPriors();


%% Functions

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
    
    % variance scaling with amplitude
    if ~isfield(options, 'variance_scaling')
        options.variance_scaling = 0.2; % typically, s/m=0.2, f=0.1
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
    % approximately 1.5, the normal range should be 1.0-2.0
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


function [unitary_amplitude, hb_gmm_model] = AnalyzeResult(model, X, options)

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
    hb_gmm_model = model;
    
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
    title('Spike Rate Distribution (HB-GMM Fit)');
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
 



function TestPriors()
   
   cd('CASCADE/Results_for_autocalibration')
   dataset_folder = 'DS32-GCaMP8s-m-V1';
   dataset_name = 'GCaMP8s';
   cd(dataset_folder)
   neuron_files = dir('CAttached*.mat');
   num_neurons = length(neuron_files);

   % test range
   test_ampitude = 0.4:0.2:3;
   test_variance = 0.1:0.1:0.5;
   [Amp, Var] = meshgrid(test_ampitude, test_variance);
   
   % initialize matrices to store results
   additive_metrics = zeros(length(test_variance), length(test_ampitude), num_neurons);
   log_metrics = zeros(length(test_variance), length(test_ampitude), num_neurons);
   
   for file_idx = 1:num_neurons
       fprintf('Processing neuron %d/%d\n', file_idx, num_neurons);
       
       data = load(neuron_files(file_idx).name);
       event_detection = data.spike_rates_GC8 > 0.3;
       labels = bwlabel(event_detection);
       A = regionprops(labels);
       
       spike_rate_per_event = zeros(numel(A), 1);
       spike_rate_per_event_GT = zeros(numel(A), 1);
       
       for k = 1:numel(A)
           range_values = round(A(k).BoundingBox(1):(A(k).BoundingBox(1)+A(k).BoundingBox(3)));
           spike_rate_per_event(k) = sum(data.spike_rates_GC8(range_values(1)-2:range_values(end)+2));
           spike_rate_per_event_GT(k) = sum(data.ground_truth(range_values(1)-2:range_values(end)+2));
       end
       
       % test different parameter combinations
       for i = 1:length(test_variance)
           for j = 1:length(test_ampitude)
               options = struct();
               options.expected_unit_amp = test_ampitude(j);
               options.variance_scaling = test_variance(i);
               
               try
                   X = spike_rate_per_event(~isnan(spike_rate_per_event));
                   options = SetDefault(options);
                   [F, max_components] = InitializeModel(X, options);
                   [model, ~] = FitModel(X, F, max_components, options);
                   [sorted_means, ~] = sort(model.mu);
                   unit_amp = sorted_means(1);
                   optimized_spike_rate = spike_rate_per_event/unit_amp;
                   detect_events = abs(spike_rate_per_event_GT - 1) < 0.5;
                   
                   if sum(detect_events) > 0
                       additive_metrics(i,j,file_idx) = nanmedian(optimized_spike_rate(detect_events)) - 1;
                       log_metrics(i,j,file_idx) = nanmedian(log(optimized_spike_rate(detect_events)));
                   end

               catch ER
                   fprintf('Warning: Failed for neuron %d, amp=%.2f, var=%.2f: %s\n', ...
                       file_idx, test_ampitude(j), test_variance(i), ER.message);
                   additive_metrics(i,j,file_idx) = NaN;
                   log_metrics(i,j,file_idx) = NaN;
               end
           end
       end
   end
   
   % mean across all neurons
   mean_additive = nanmean(additive_metrics,3);
   mean_log = nanmean(log_metrics,3);
   

   figure('Position', [100, 100, 1500, 1000]);
   
   % plot 1: Additive Error
   subplot(1,2,1);
   surf(Amp, Var, abs(mean_additive));
   xlabel('Initial Amplitude');
   ylabel('Variance Scaling');
   zlabel('Additive Error');
   title('Effects on Additive Error');
   colorbar;
   
   % plot 2: Log Error
   subplot(1,2,2);
   surf(Amp, Var, abs(mean_log));
   xlabel('Initial Amplitude');
   ylabel('Variance Scaling');
   zlabel('Log Error');
   title('Effects on Logarithmic Error');
   colorbar;

   sgtitle(sprintf('Parameter Effects Analysis for Dataset %s', dataset_name));
   
   % 2D view
   figure('Position', [100, 100, 1500, 1000]); 

   subplot(1,2,1);
   imagesc(unique(Amp(1,:)), unique(Var(:,1)), abs(mean_additive));
   xlabel('Initial Amplitude');
   ylabel('Variance Scaling');
   title('Effect on Additive Error');
   colorbar;
   axis xy;
   
   subplot(1,2,2);
   imagesc(unique(Amp(1,:)), unique(Var(:,1)), abs(mean_log));
   xlabel('Initial Amplitude');
   ylabel('Variance Scaling');
   title('Effect on Logarithmic Error');
   colorbar;
   axis xy;
   
   sgtitle(sprintf('2D - Parameter Effects Analysis for Dataset %s', dataset_name));

end

