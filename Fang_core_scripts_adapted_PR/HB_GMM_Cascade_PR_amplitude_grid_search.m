
%%

dataset_folder = 'DS32-GCaMP8s-m-V1';

cd('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\')
folders = dir('Results_for_autocalibration_d*');

cd('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\Results_for_autocalibration_d0.47\')
cd(dataset_folder)
file_IDs = dir('CAttached*.mat');

clear systematic_bias_divisions

cmap = parula(21);

clear avg_spikesX slopes_all
for file_ix = 1:numel(file_IDs)

    file_ix
    
    for folder_ix = 1:numel(folders)

        cd([folders(folder_ix).folder,'/',folders(folder_ix).name])
        cd(dataset_folder)

        systematic_bias_divisions(folder_ix) = str2double(folders(folder_ix).name(30:end));
        
        load(file_IDs(file_ix).name)

        spike_rates_GC8 = spike_rates_GC8;
        ground_truth = ground_truth;


        good_ixs = find(~isnan(spike_rates_GC8));

        smoothed_data = conv(ground_truth(good_ixs),fspecial('gaussian',[60 1],30),'same'); 
        
        ground_truth = ground_truth(good_ixs);
        X = conv(spike_rates_GC8,fspecial('gaussian',[60 1],30),'same'); 

        [ix,xi] = sort(smoothed_data,'ascend');
        
        bins = 0.02:0.02:1.0;
        clear avg_spikes
        for jj = 1:numel(bins)
            
            indices = find((smoothed_data<bins(jj)).*(smoothed_data>(bins(jj)-0.02)));

            avg_spikes(jj,2) = nanmean(X(indices));
            avg_spikes(jj,1) = nanmean(smoothed_data(indices));
        end

        y0 = avg_spikes(:,2);
        x0 = avg_spikes(:,1);

        % Fit: 'untitled fit 1'.
        warning('off', 'all')
        [xData, yData] = prepareCurveData( x0, y0 );
            
        % Set up fittype and options.
        ft = fittype( 'a*x', 'independent', 'x', 'dependent', 'y' );
        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
        opts.Display = 'Off';
        opts.StartPoint = 1;
            
        % Fit model to data.
        [fitresult, gof] = fit( xData, yData, ft, opts );
        y_predict = fitresult(xData);
        warning('on', 'all');
        slope = fitresult.a;

        slopes_all(file_ix,folder_ix) = slope;


        avg_spikesX(file_ix,folder_ix,:,:) = avg_spikes;

%         figure(411); subplot(6,6,file_ix); plot(x0,y0,'Color',cmap(folder_ix,:)); hold on;
%         if folder_ix == numel(folders)
%             plot([0 0.5],[0 0.5],'--k')
%             hold off;
%         end

    end

end





x1. Go o a random results folder, get list of files
x2. For each file, find all scaling results, load them into memory
3. Generate Firing Frequency vs. Inferred Firing Frequency plot for each scaling factor
4. Generate (linear?) fit
5. Compare linear slope with the GMM result



%% Apply (Hybrid)HB-GMM for GCaMP8

% preparations
addpath('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Autocalibration\Autocalibrated-spike-inference\Fang_core_scripts_adapted_PR')

cd('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\')

folders = dir('Results_for_autocalibration_d*');


all_raw_metrics_log = [];
all_raw_metrics_additive = [];
all_optimized_metrics_log = [];
all_optimized_metrics_additive = [];

all_unitary_amplitudes = [];

clear systematic_bias_divisions

for jkl = 1:numel(folders)

    cd([folders(jkl).folder,'/',folders(jkl).name])

    disp(folders(jkl).name)

    systematic_bias_divisions(jkl) = str2double(folders(jkl).name(30:end));

    dataset_folder = 'DS32-GCaMP8s-m-V1';
    dataset_name = 'GCaMP8s';
    cd(dataset_folder)
    neuron_files = dir('CAttached*.mat');
    
    
    % initialize arrays

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
            [unitary_amplitude, model] = AnalyzeSpikesHBGMM(spike_rate_per_event,false);
            all_unitary_amplitudes(file_idx,jkl) = unitary_amplitude;
            
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
                all_raw_metrics_additive(file_idx,jkl) = raw_metric_additive;
                all_optimized_metrics_additive(file_idx,jkl) = optimized_metric_additive;
                all_raw_metrics_log(file_idx,jkl) = raw_metric_log;
                all_optimized_metrics_log(file_idx,jkl) = optimized_metric_log;
    
                all_spike_rates{file_idx} = spike_rate_per_event;
                all_spike_rates_GT{file_idx} = spike_rate_per_event_GT;
                all_optimized_rates{file_idx} = optimized_spike_rate;
                
            end
    
    
        catch ER
            fprintf('Warning: Analysis failed for file %d: %s\n', file_idx, ER.message);
            
            all_raw_metrics_log(file_idx,jkl) = NaN;
            all_raw_metrics_additive(file_idx,jkl) = NaN;
            all_optimized_metrics_log(file_idx,jkl) = NaN;
            all_optimized_metrics_additive(file_idx,jkl) = NaN;
            all_unitary_amplitudes(file_idx,jkl) = NaN;
            all_component_counts(file_idx) = NaN;
            all_component_weights{file_idx} = NaN;
            all_component_means{file_idx} = NaN;
        end
    end

end



% Visualization and Statistical Analysis
% figure('Position', [100, 100, 1000, 1000]);
figure('Position', 1e3*[0.5293    1.1047    1.1570    0.9057]);
% set(gcf,'Posiiton',1e3*[0.5293    1.1047    1.1570    0.9057])


% plot 1 & 2: additive and log metrics comparison
subplot(2,2,1);
boxplot([all_raw_metrics_additive', all_optimized_metrics_additive'], ...
    'Labels', {'Raw', 'Optimized'}, 'Notch', 'on');
title('Additive Metrics: Raw vs Optimized');
ylabel('Additive Error');
grid on;


% plot 2
subplot(2,2,2);
boxplot([all_raw_metrics_log', all_optimized_metrics_log'], ...
    'Labels', {'Raw', 'Optimized'}, 'Notch', 'on');
title('Logarithmic Metrics: Raw vs Optimized');
ylabel('Logarithmic Error');
grid on;



% plot 3 & 4: scatter of metrics
subplot(2,2,3);
scatter(all_raw_metrics_additive, all_optimized_metrics_additive, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
xlabel('Raw Additive Error');
ylabel('Optimized Additive Error');
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

% example rates
figure; 
plot(all_spike_rates_GT{1},'r'); 
hold on; 
plot(all_spike_rates{1},'b'); 
plot(all_optimized_rates{1},'g');


% stats
% print analysis summary
fprintf('\n$$$ Analysis Summary for %s $$$\n', dataset_name);
fprintf('Total number of neurons analyzed: %d\n', length(neuron_files));
fprintf('Successfully analyzed neurons: %d\n', sum(~isnan(all_unitary_amplitudes)));

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


