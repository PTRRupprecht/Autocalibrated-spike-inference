cd('/Users/fvinci6/Downloads')

%% Each file contains the data for a single neuron in a single Matlab "struct" named "CAttached"
% For most neurons, multiple recordings have been performed (with a few minutes between recordings).
% These multiple recordings can be accessed as CAttached{1}, CAttached{2}, etc.

filename = {
    'CAttached_jGCaMP8s_472181_1_mini'
    'CAttached_jGCaMP8s_479120_1_mini'
    'CAttached_jGCaMP8s_479120_8_mini'
    'CAttached_jGCaMP8s_479120_6_mini'
    'CAttached_jGCaMP8s_479120_2_mini'
    'CAttached_jGCaMP8s_472181_3_mini'
};

% Initialize a cell array to store all CAttached data
all_CAttached = cell(1, numel(filename));

%{
% Iterate through each filename
for file_idx = 1:numel(filename)
    
    % Construct full file path
    full_file_path = fullfile([filename{file_idx}, '.mat']);
    
    % Load data from file
    load(full_file_path, 'CAttached');
    
    % Store CAttached data for this file
    all_CAttached{file_idx} = CAttached;
end

% Process all loaded data
for file_idx = 1:numel(all_CAttached)
    CAttached = all_CAttached{file_idx};
    
    nb_recordings = numel(CAttached); 
    for index = 1:nb_recordings

        % read the variables of the current recording
        time = CAttached{index}.fluo_time; % time points of the fluorescence trace
        fluo_trace = CAttached{index}.fluo_mean; % fluorescence trace (in dF/F)
        AP_times = CAttached{index}.events_AP/1e4; % action potential times (in seconds)

        % plot the traces of the current recording
        figure(file_idx * 1000 + index); 
    
        plot(time,fluo_trace); hold on;
        for k = 1:numel(AP_times)
            plot([AP_times(k) AP_times(k)],[-1 -0.5],'k');
        end
        hold off;
        set(gca,'TickDir','out'); box off
        ylim([-1.5 9])

    end
end
%}

%% Exercise 1: Isolated action potentials
% a) For each neuron, find the times of all action potentials with no other action potential 4 seconds before and 4 seconds after it. 
% b) Extract the corresponding calcium transients (4 seconds before until 4 seconds after)
% c) Visualize these transients somehow
% d) Compare the extracted calcium transients across neurons; what are your observations?

% Your code here


% Initialize a cell array to store isolated transients for all neurons
all_isolated_transients = cell(1, numel(filename));

for file_idx = 1:numel(filename)
    
    % Construct full file path
    full_file_path = fullfile([filename{file_idx}, '.mat']);
    
    % Load data from file
    load(full_file_path, 'CAttached');
    
    % Store CAttached data for this file
    all_CAttached{file_idx} = CAttached;
    
    % Initialize storage for isolated transients for this neuron
    neuron_isolated_transients = [];
    
    % Process each recording for the current neuron
    for index = 1:numel(CAttached)
        
        % Read the variables of the current recording
        time = CAttached{index}.fluo_time; % Time points of the fluorescence trace
        fluo_trace = CAttached{index}.fluo_mean; % Fluorescence trace (in dF/F)
        AP_times = CAttached{index}.events_AP / 1e4; % Action potential times (in seconds)

        % a) Find times of all action potentials with no other action potential 4 seconds before and 4 seconds after
        isolated_AP_times = [];
        for k = 1:numel(AP_times)
            if (k == 1 || abs(AP_times(k) - AP_times(k-1)) > 4) && ...
               (k == numel(AP_times) || abs(AP_times(k) - AP_times(k+1)) > 4)
                isolated_AP_times = [isolated_AP_times, AP_times(k)];
            end
        end

        expected_nb_time_points = floor(8*1/nanmedian(diff(time)));

        % b) Extract corresponding calcium transients (4 seconds before until 4 seconds after)
        for k = 1:numel(isolated_AP_times)
            start_time = isolated_AP_times(k) - 4;
            end_time = isolated_AP_times(k) + 4;
            
            % Find indices corresponding to the time window
            transient_indices = find(time >= start_time & time <= end_time);
            
            if ~isempty(transient_indices)
                % Extract the transient
                transient_fluo = fluo_trace(transient_indices);
                
                % Store the transient
                if numel(transient_fluo) > expected_nb_time_points

                    neuron_isolated_transients = [neuron_isolated_transients, transient_fluo(1:expected_nb_time_points)'];

                end
            end
        end
    end
    
    % Store isolated transients for this neuron in the cell array
    all_isolated_transients{file_idx} = neuron_isolated_transients;
end


% c) Visualize isolated transients
for file_idx = 1:numel(all_isolated_transients)
    isolated_transients = all_isolated_transients{file_idx};
    
    if ~isempty(isolated_transients)

        figure(file_idx * 100 + 1); % Separate figure for isolated transients
        
        timeX = (1:size(isolated_transients, 1));
        for k = 1:size(isolated_transients, 2)
            plot(timeX, isolated_transients(:, k)+(k-1)*3); hold on;
        end
        hold off;
        xlabel('Time (a.u.) relative to AP');
        ylabel('Fluorescence (dF/F)');

        title(['Isolated Calcium Transients for Neuron %d', num2str(file_idx)]);
%         title(sprintf('Isolated Calcium Transients for Neuron %d', file_idx));
        grid on;
    end
end

% d) Comparison of extracted calcium transients across neurons

clear average_transient
for file_idx = 1:numel(all_isolated_transients)
    isolated_transients = all_isolated_transients{file_idx};
    if ~isempty(isolated_transients)

        average_transient{file_idx} = mean(isolated_transients(487:600,:),1);
    end
end

mean(average_transient{2})
mean(average_transient{4})
mean(average_transient{5})


figure(31* 100 + 1); % Separate figure for isolated transients
for file_idx = 1:numel(all_isolated_transients)
    isolated_transients = all_isolated_transients{file_idx};
    
    if ~isempty(isolated_transients)
        timeX = (1:size(isolated_transients, 1));
        plot(timeX, nanmean(isolated_transients(:, :),2)); hold on;
    end
end
hold off;
xlabel('Time (a.u.) relative to AP');
ylabel('Fluorescence (dF/F)');

title(['Isolated Calcium Transients for all neurons']);



%% Exercise 2: All action potentials (not only isolated ones)
% a) Try to find a way to extract the "average calcium transient" triggered by an action potential (be it isolated or not)
% b) Compare this average calcium transient to the calcium transient triggered by isolated action potentials (Exercise 1); what are your observations?

% Your code here


1. Modeling: fit to data

template_time = (1:975);
template_calcium_transient = zeros(975,1);
tau = 50;
amplitude = 0.6;
template_calcium_transient(488:975) = amplitude*exp(-((488:975)-488)/tau);
figure, plot(template_calcium_transient)

a) load one recording
b) find all action potential time points
c) for each action potential: add the template to a trace
d) compare it with the measured calcium trace
e) compute error (MSE, mean squared error)
f) i) gradient descent on the parameters (amplitude and tau) ("gradient descent")
   ii) go through all parameter combinations and find the best ("grid search")



2. Deconvolution: Linear kernels were extracted by regularized deconvolution using the deconvreg(Calcium,Spikes) function in MATLAB (MathWorks). 

 What is convolution?  A * B = https://en.wikipedia.org/wiki/Convolution

A = zeros(16,16); A(4,6) = 1; A(7,10) = 1; A = rand(64,64);
B = fspecial('gaussian',[3 3],1);

Blurred_A = conv2(A,B,"same");
figure, imagesc(A)
figure, imagesc(Blurred_A)

 What is deconvolution?

Restore_A = deconvreg(Blurred_A,B);
figure, imagesc(Restore_A)


