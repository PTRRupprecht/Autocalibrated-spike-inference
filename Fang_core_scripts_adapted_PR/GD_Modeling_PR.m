%% Gradient Descent Modeling


cd('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Autocalibration\Autocalibrated-spike-inference\GT_autocalibration')

addpath('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Autocalibration\Autocalibrated-spike-inference\Fang_core_scripts_adapted_PR')

% preparations
dataset_folder = 'DS31-GCaMP8m-m-V1';
dataset_name = 'GCaMP8m';
cd(dataset_folder)

neuron_files = dir('CAttached*.mat');
num_neurons = numel(neuron_files);

modeling_amp = [];
neuron_amplitudes = [];

% randomly select two neurons to plot
selected_neurons = randperm(num_neurons, 2); %change if u wanna more plots

figure('Name','GCaMP8 Selected Neuron Traces', 'Position', [100, 100, 1000, 1000]);

% process each neuron
for neuron_idx = 1:num_neurons

    fprintf('Processing neuron %d of %d\n', neuron_idx, num_neurons);

    load(neuron_files(neuron_idx).name);

    num_recordings = numel(CAttached);
    
    % choose a random recording from selected neurons
    selected = find(selected_neurons == neuron_idx);
    if ~isempty(selected)
        plot_recording_idx = randi(num_recordings);
    end
    
    % process each recording
    for recording_idx = 1:num_recordings
        fprintf('Processing recording %d of %d\n', recording_idx, num_recordings);

        % extract data
        fluo_time = CAttached{recording_idx}.fluo_time;
        fluo_trace = CAttached{recording_idx}.fluo_mean;
        AP_times = CAttached{recording_idx}.events_AP / 1e4;
        
        % generate simulated trace and extract parameters
        [optimized_amplitude, optimized_tau_rise, optimized_tau_decay, final_error] = ...
            Gradient_Descent(fluo_time, AP_times, fluo_trace);
        
        % store all amplitudes from each recording as our modeling amplitudes
        modeling_amp = [modeling_amp; optimized_amplitude];
        
        % plot only the selected random neuron and recording
        if ~isempty(selected) && recording_idx == plot_recording_idx

            % generate simulated trace for plotting
            simulated_trace = zeros(size(fluo_time));

            for j = 1:length(AP_times)
                t_since_ap = fluo_time - AP_times(j);
                t_since_ap(t_since_ap < 0) = 1e12;

                simulated_trace = simulated_trace + optimized_amplitude * ...
                    (exp(-t_since_ap / optimized_tau_decay) .* (1 - exp(-t_since_ap / optimized_tau_rise)));
            end
            
            subplot(1, 2, selected);
            plot(fluo_time, fluo_trace, 'b', fluo_time, simulated_trace, 'r');
            title(sprintf('Neuron %d Recording %d Amp: %.2f', neuron_idx, plot_recording_idx, modeling_amp(plot_recording_idx)));
            xlabel('Time (s)');
            ylabel('Fluorescence');
            legend('Measured', 'Simulated','Location', 'best');

        end
    end

    % store also median amplitude of each neuron
    neuron_median_amplitude = median(modeling_amp);
    neuron_amplitudes = [neuron_amplitudes; neuron_median_amplitude];

end

sgtitle('Measured vs Simulated Calcium Trace - Two Random Examples');



%% Visualization of amplitudes

figure('Name', 'GCaMP8s Amplitude Distribution');

% Histogram
subplot(2,1,1);
histogram(modeling_amp(modeling_amp <= 5), 'BinWidth', 0.05, ...
    'Normalization', 'probability', 'FaceColor', 'g');
title('Distribution of GCaMP8s Neuron Amplitudes');
xlabel('Amplitude (dF/F)');
ylabel('Probability');

% Box plot
subplot(2,1,2);
boxplot(modeling_amp(modeling_amp <= 5));
title('GCaMP8s Amplitude Box Plot');
ylabel('Amplitude (dF/F)');


% statistics
mean_amplitude = mean(modeling_amp);
median_amplitude = median(modeling_amp);
std_amplitude = std(modeling_amp);

fprintf('\nResults for %s:\n', dataset_name);
fprintf('Mean Amplitude: %.2f\n', mean_amplitude);
fprintf('Median Amplitude: %.2f\n', median_amplitude);
fprintf('Standard Deviation: %.2f\n', std_amplitude);



