%% Fig. 3: Kernels and Delays


%% Get an overview of statistics (frame rates and spike rates) of the ground truth data sets
% This script section produces the variable "dt_all", which is then used for the
% next section below

cd('Autocalibrated-spike-inference/GT_autocalibration')

GT_folders = {'DS09-GCaMP6f-m-V1',...
    'DS10-GCaMP6f-m-V1-neuropil-corrected',...
    'DS11-GCaMP6f-m-V1-neuropil-corrected',...
    'DS13-GCaMP6s-m-V1-neuropil-corrected',...
    'DS14-GCaMP6s-m-V1',...
    'DS29-GCaMP7f-m-V1','DS30-GCaMP8f-m-V1','DS31-GCaMP8m-m-V1','DS32-GCaMP8s-m-V1',...
    'DS06-GCaMP6f-zf-aDp','DS07-GCaMP6f-zf-dD','DS08-GCaMP6f-zf-OB'};

% dt = delta_t = 1/(frame rate)
dt_all = zeros(numel(GT_folders),1);

clear spike_rate_GT_all

% Go through all GT datasets
for folder_index =  1:numel(GT_folders)
    
    cd(GT_folders{folder_index})
    
    % Find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    dtX = [];
    spike_rate_GTX = [];

    for neuron_index = 1:numel(neuron_files)

        load(neuron_files(neuron_index).name)

        for index = 1:numel(CAttached)
            
            fluo_time = CAttached{index}.fluo_time; % Time points of the fluorescence trace
            fluo_trace = CAttached{index}.fluo_mean; % Fluorescence trace (in dF/F)
            AP_times = CAttached{index}.events_AP / 1e4; % Action potential times (in seconds)
            
            % compute spike rate for the current neuron
            spike_rate_GT = numel(AP_times)/(max(fluo_time)-min(fluo_time));
            % concatenate current neuron's spike rate to variable "spike_rate_GTX"
            spike_rate_GTX = [spike_rate_GTX;numel(AP_times)/(max(fluo_time)-min(fluo_time))];

            % compute dt from the variable fluo_time (median difference between time points)
            dt = nanmedian(diff(fluo_time));
            dtX = [dtX;dt];
        end
    end

    % Average across neurons for each GT dataset 
    dt_all(folder_index) = nanmedian(dtX);
    spike_rate_GT_all(folder_index) = nanmedian(spike_rate_GTX);

    % Plot sorted distribution of spike rates
    %  %figure(411), plot((1:numel(spike_rate_GTX))/numel(spike_rate_GTX),sort(spike_rate_GTX)); hold on;
    
    % Print summary statistics for the current GT dataset
    disp(['For dataset ',GT_folders{folder_index},', mean spike rate: ',num2str(spike_rate_GT_all(folder_index)),'; average framerate: ',num2str(1/dt_all(folder_index))])

    cd ..


end


%% Extract average spike-evoked kernels for all datasets and neurons

% Clear the variable that will lateron contain all kernels
clear kernel_averaged_all

% Go through all GT datasets
for folder_index = 1:numel(GT_folders)
    
    cd(GT_folders{folder_index})
    
    % Find all GT neurons
    neuron_files = dir('CAttached*.mat');
    
    % Initialize matrix that will lateron contain all kernels for this dataset
    kernel_averaged = [];

    % Go through all neurons of this GT dataset
    for neuron_index = 1:numel(neuron_files)

        load(neuron_files(neuron_index).name)

        kernelX_all = [];
        % Go through all recordings done from the currently analyzed neuron
        for index = 1:numel(CAttached)
            
            % This part is necessary because "fluo_time" is sometimes a
            % column vector and sometimes a row vector, depending on the dataset
            if size(CAttached{index}.fluo_time,2) > 1
                CAttached{index}.fluo_time = CAttached{index}.fluo_time';
            end
            if size(CAttached{index}.fluo_mean,2) > 1
                CAttached{index}.fluo_mean = CAttached{index}.fluo_mean';
            end
            
            fluo_time = CAttached{index}.fluo_time; % Time points of the fluorescence trace
            fluo_trace = CAttached{index}.fluo_mean; % Fluorescence trace (in dF/F)
            AP_times = CAttached{index}.events_AP / 1e4; % Action potential times (in seconds)
            
            % find non-NaN values
            good_indices = find(~isnan(CAttached{index}.fluo_time).*~isnan(CAttached{index}.fluo_mean));
            
            fluo_time = fluo_time(good_indices);
            fluo_trace = fluo_trace(good_indices);
            CAttached{index}.fluo_time = CAttached{index}.fluo_time(good_indices);
            CAttached{index}.fluo_mean = CAttached{index}.fluo_mean(good_indices);

            spikes = AP_times;
            
            % compute dt (time between two imaging frames)
            dt = nanmedian(diff(fluo_time));
            % get (from the script above) the target dt0 for this dataset
            dt0 = dt_all(folder_index);

            % resample the recording to the target frame rate 1/dt0 if
            % necessary; this is important if frame rates vary between
            % recordings of a single dataset
           if abs(dt - dt0)/dt0 > 0.05
                CAttached{index}.fluo_mean_resampled = resample(double(fluo_trace),round(1/dt0*100),round(1/dt*100));
                CAttached{index}.fluo_time_resampled = (dt0:dt0:dt0*numel(CAttached{index}.fluo_mean_resampled)) ;
           else
                CAttached{index}.fluo_mean_resampled = double(fluo_trace);
                CAttached{index}.fluo_time_resampled = CAttached{index}.fluo_time+dt0-CAttached{index}.fluo_time(1);
            end
            
            % The next steps prepare the variables "spike_density" and
            % "fluorescence"; deconvolution is used to obtain the response
            % kernel from those two signals

            % allocate all spikes into bins; the bins are indicated by the
            % frame times, "CAttached{index}.fluo_time_resampled"; 
            spike_density = hist(spikes(spikes<(max(CAttached{index}.fluo_time_resampled))),CAttached{index}.fluo_time_resampled);
            fluorescence = CAttached{index}.fluo_mean_resampled;
            
            good_indices = find(~isnan(fluorescence));
            spike_density = spike_density(good_indices);
            fluorescence = fluorescence(good_indices);
            
            % perform deconvolution to obtain response kernel
            try
%                 kernel = fftshift( ifft(fft(DF)./fft(spike_density)));
                kernel_lucy = deconvreg(fluorescence,spike_density);
            catch
%                 kernel = fftshift( ifft(fft(DF)./fft(HD')));
                kernel_lucy = deconvreg(fluorescence,spike_density');
            end
            
            % get center point of the resulting kernel
            center = round(numel(kernel_lucy)/2);
            
            % extract only the central part of the kernel
            window_extent = 4; % in seconds
            relevant_kernel_excerpt = kernel_lucy(round(center-1/dt0*window_extent):round(center+1/dt0*window_extent));
            
            % concatenate all extracted kernels
            kernelX_all = [kernelX_all, relevant_kernel_excerpt];
        end

        % Treat some special cases ...

        if 1%  (folder_index > 1 && folder_index < 5)

            if size(kernelX_all,1) > 1 && size(kernelX_all,2) > 1
                kernel_averaged = [kernel_averaged,squeeze(nanmean(kernelX_all,2))];
            else
                kernel_averaged = [kernel_averaged,squeeze(kernelX_all)];
            end
        else

            if size(kernelX_all,1) > 1 && size(kernelX_all,2) > 1
                kernel_averaged = [kernel_averaged,squeeze(nanmean(kernelX_all,1))'];
            else
                kernel_averaged = [kernel_averaged,squeeze(kernelX_all)'];
            end

        end



    end
    
    % Write the extracted kernels for each dataset into the pooling
    % variable "kernel_averaged_all"
    kernel_averaged_all{folder_index} = kernel_averaged;


    cd ..

end


%% Dataset names

datasets = {'GC6f_v','GC6f_tg','GC6f_tg','GC6s_tg','GC6s_v','GC7f','GC8f','GC8m','GC8s','GC78_IN','GC6f_zf'};


%% Plot all kernels; for each dataset in a separate subplot

colors = {'c','c','c','c','c','m','r','b','k','g','k','k','k'};

figure(12125);
for k = 1:10
    timeX = ((1:size(kernel_averaged_all{k},1)) - (size(kernel_averaged_all{k},1)-1)/2-1)*dt_all(k)*1000;
%     timeX = timeX;

    % average across datasets 10-12 if k=10
    if k == 10
        transient = nanmedian(([kernel_averaged_all{k}';kernel_averaged_all{k+1}';kernel_averaged_all{k+2}']));
    else
        transient = nanmedian(kernel_averaged_all{k}');
    end

    % subtract pre-spike baseline
    transient = transient - nanmean(transient(1:round(numel(transient)/2)));

%    transient = transient/max(transient);
    subplot(3,4,k)
    plot(timeX,transient,'Color',colors{k}); %hold on;
    
    box off;
    set(gca,'TickDir','out');
    ylim([-0.15 1.05])
    xlim([min(timeX)+2 max(timeX)+0.001])
%     xlim([-0.1 0.4])
    xlim([-0.02 0.3]*1000)
    grid on
    xlabel('Time (ms)')
    ylabel('dF/F')
    title(datasets{k},'Interpreter','None')

end
hold off

set(gcf,'Position', [  360.0000  328.3333  628.3333  369.6667])


%Plotting all kernels in the same plot
figure(12126);
hold on;

colors = {'c','c','c','c','c','m','r','b','k','g'};
legends = {};

for k = 1:10
    timeX = ((1:size(kernel_averaged_all{k},1)) - (size(kernel_averaged_all{k},1)-1)/2-1)*dt_all(k)*1000;
    
    if k == 10
        transient = nanmedian(([kernel_averaged_all{k}';kernel_averaged_all{k+1}';kernel_averaged_all{k+2}']));
    else
        transient = nanmedian(kernel_averaged_all{k}');
    end
    
    transient = transient - nanmean(transient(1:round(numel(transient)/2)));
    plot(timeX, transient, 'Color', colors{k}, 'LineWidth', 2);
    
    legends{end+1} = datasets{k};
end

xlabel('Time (ms)');
ylabel('dF/F');
title('Comparison of Kernels Across Datasets');
legend(legends, 'Location', 'eastoutside');
grid on;
xlim([-50 300]);
ylim([-0.15 1.05]);
hold off


%% Boxplot of half rise times

half_rise_times_all = NaN*zeros(numel(kernel_averaged_all),100);
% figure(121425);
for k = 1:10
    timeX = ((1:size(kernel_averaged_all{k},1)) - (size(kernel_averaged_all{k},1)-1)/2-1)*dt_all(k)*1000;
    

    if k == 10
        kernel_averaged_allX = [kernel_averaged_all{k}';kernel_averaged_all{k+1}';kernel_averaged_all{k+2}'];
    else
        kernel_averaged_allX = [kernel_averaged_all{k}'];
    end


    clear half_rise_times
    for jj = 1:size(kernel_averaged_all{k},2)
        
        transient = kernel_averaged_allX(jj,:);
        transient = transient - nanmean(transient(1:round(numel(transient)/2)));
        transient = transient/max(transient);
    
        change_point = find(transient>0.5,1,'first');
        
        half_rise_time_point = ((change_point-1)/abs(0.5-transient(change_point-1)) + change_point/abs(0.5-transient(change_point)))/(1/abs(0.5-transient(change_point-1)) + 1/abs(0.5-transient(change_point)));
        
        half_rise_times(jj) = interp1((change_point-1):change_point,timeX((change_point-1):change_point),half_rise_time_point);
        
    end

    half_rise_times_all(k,(1:numel(half_rise_times))) = half_rise_times;
end

half_rise_times_all(abs(half_rise_times_all)>0.2*1000) = NaN;

indices = [7 8 9 6 2 3 4 1 5 10];
figure(446), boxplot(half_rise_times_all(indices,:)')
xticklabels(datasets(indices))
xtickangle(45)
box off;
set(gca,'TickDir','out');
set(gcf,'Position', [  360.0000  421.6667  345.0000  276.3333])
ylabel('Half rise time (ms)')

%% Quantitative analysis of kernel variation

% Observe peak amplitudes
peak_amplitudes = [];
for k = 1:10
    if k == 10
        transient = nanmedian(([kernel_averaged_all{k}';kernel_averaged_all{k+1}';kernel_averaged_all{k+2}']));
    else
        transient = nanmedian(kernel_averaged_all{k}');
    end
    transient = transient - nanmean(transient(1:round(numel(transient)/2)));
    peak_amplitudes(k) = max(transient);
end

% Maybe have a look at AUC?
auc = [];
for k = 1:10
    timeX = ((1:size(kernel_averaged_all{k},1)) - (size(kernel_averaged_all{k},1)-1)/2-1)*dt_all(k)*1000;
    if k == 10
        transient = nanmedian(([kernel_averaged_all{k}';kernel_averaged_all{k+1}';kernel_averaged_all{k+2}']));
    else
        transient = nanmedian(kernel_averaged_all{k}');
    end
    transient = transient - nanmean(transient(1:round(numel(transient)/2)));
    auc(k) = trapz(timeX, transient);
end

% Decay time
decay_times = [];
for k = 1:10
    timeX = ((1:size(kernel_averaged_all{k},1)) - (size(kernel_averaged_all{k},1)-1)/2-1)*dt_all(k)*1000;
    if k == 10
        transient = nanmedian(([kernel_averaged_all{k}';kernel_averaged_all{k+1}';kernel_averaged_all{k+2}']));
    else
        transient = nanmedian(kernel_averaged_all{k}');
    end
    transient = transient - nanmean(transient(1:round(numel(transient)/2)));
    peak = max(transient);
    half_peak = peak / 2;
    [~, peak_index] = max(transient);
    decay_index = find(transient(peak_index:end) <= half_peak, 1, 'first') + peak_index - 1;
    decay_times(k) = timeX(decay_index) - timeX(peak_index);
end

% Computing in a table
variation_table = table(datasets(1:10)', peak_amplitudes', auc', decay_times', half_rise_times_all(1:10,1), ...
    'VariableNames', {'Dataset', 'PeakAmplitude', 'AUC', 'DecayTime', 'HalfRiseTime'});

disp(variation_table);

