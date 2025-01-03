


function TestPriors()
   
   addpath('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Autocalibration\Autocalibrated-spike-inference\Fang_core_scripts_adapted_PR')
   cd('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\Results_for_autocalibration')
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
   

%    figure('Position', [100, 100, 1500, 1000]);
   figure('Position', 1e3*[0.5280    1.4290    1.1070    0.2597]);

   % plot 1: Additive Error
   subplot(1,2,1);
   imagesc(Amp, Var, abs(mean_additive));
   xlabel('Initial Amplitude');
   ylabel('Variance Scaling');
   zlabel('Additive Error');
   title('Effects on Additive Error');
   colorbar;
   
   % plot 2: Log Error
   subplot(1,2,2);
   imagesc(Amp, Var, abs(mean_log));
   xlabel('Initial Amplitude');
   ylabel('Variance Scaling');
   zlabel('Log Error');
   title('Effects on Logarithmic Error');
   colorbar;

   sgtitle(sprintf('Parameter Effects Analysis for Dataset %s', dataset_name));
   
   % 2D view
%    figure('Position', [100, 100, 1500, 1000]); 
   figure('Position', 1e3*[0.5280    1.4290    1.1070    0.2597]);

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

