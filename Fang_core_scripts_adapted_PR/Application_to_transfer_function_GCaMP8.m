


% x 1. extract and save unitary responses for all neurons
% x 2. plot transfer curves after division by unitary responses
% x 3. compare the deviation from unity for all 3 datasets



results_unitary_respones = {'C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\Results_for_autocalibration\DS30-GCaMP8f-m-V1',...
    'C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\Results_for_autocalibration\DS31-GCaMP8m-m-V1',...
    'C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\Results_for_autocalibration\DS32-GCaMP8s-m-V1'};


%% Second attempt (first attempt below)

folders = dir('C:\Users\peter\Desktop\Spike_inference_with_GCaMP8\Cascade_GC8\Cascade\Result_noise8_framerate30_September24\DS*-*');


full_performance_matrix = [];

clear deviation_from_linearity deviation_from_identity

for kkk = 14:14%[6 7 8 9 10 11 12 13 14 30:31]%numel(folders)]

    cd(results_unitary_respones{kkk-11})
    load('Autocalibration_results.mat')

%     all_unitary_amplitudes
%     all_file_names

    % go to target dataset (analyzed! this is not the GT dataset!)
    cd([folders(kkk).folder,'/',folders(kkk).name]) % go to DS332
    
   
    % make list of neurons within dataset
    files = dir('WithMLSpike*CAttached*.mat');
    filesO = dir('WithMLSpike*CAttached*.mat');
   
    clear full_shifts
    for k = 1:numel(files) % select neuron of choice

        disp([kkk,k])
    
        load(files(k).name)
        load(filesO(k).name)
        
        spike_rates_MLSpike = conv(spike_rates_MLSpike,fspecial('gaussian',[20 1],1.5152),'same');
%         spike_rates_MLSpike_RE = conv(spike_rates_MLSpike_RE,fspecial('gaussian',[20 1],1.5152),'same');
        if kkk > 10
            spike_rates_MLSpike_Tuned = conv(spike_rates_MLSpike_Tuned,fspecial('gaussian',[20 1],1.5152),'same');
        end
        spike_rates_OASIS = conv(spike_rates_OASIS,fspecial('gaussian',[20 1],1.5152),'same');
        spike_rates_OASIS_finetuned = conv(spike_rates_OASIS_finetuned,fspecial('gaussian',[20 1],1.5152),'same');
        
        % compute correlation with ground truth (optimized for temporal shift)
        
        clear performance
        all_shifts = -20:20;
        for shift = 1:numel(all_shifts)
        
            ground_truth_shifted = circshift(ground_truth,all_shifts(shift));
            good_ixs = find(~isnan(spike_rates_GC8));
            
            
            performance(shift,1,k) = corr(ground_truth_shifted(good_ixs),calcium(good_ixs));
            performance(shift,2,k) = corr(ground_truth_shifted(good_ixs),spike_rates_GC8(good_ixs)');
            performance(shift,3,k) = corr(ground_truth_shifted(good_ixs),spike_rates_GLOBAL(good_ixs)');
            performance(shift,4,k) = corr(ground_truth_shifted(good_ixs),spike_rates_OASIS(good_ixs)');
            performance(shift,5,k) = corr(ground_truth_shifted(good_ixs),spike_rates_MLSpike(good_ixs)');
            performance(shift,6,k) = corr(ground_truth_shifted(good_ixs),spike_rates_new(good_ixs)');
            performance(shift,7,k) = corr(ground_truth_shifted(good_ixs),spike_rates_old(good_ixs)');
            performance(shift,8,k) = corr(ground_truth_shifted(good_ixs),spike_rates_transfer(good_ixs)');
            performance(shift,9,k) = corr(ground_truth_shifted(good_ixs),spike_rates_within(good_ixs)');
            if kkk < 11
                performance(shift,10,k) = corr(ground_truth_shifted(good_ixs),spike_rates_MLSpike(good_ixs)');
            else
                performance(shift,10,k) = corr(ground_truth_shifted(good_ixs),spike_rates_MLSpike_Tuned(good_ixs)');
            end
            performance(shift,11,k) = corr(ground_truth_shifted(good_ixs),spike_rates_OASIS_finetuned(good_ixs)');
            
        
        end
        
        [~,shift1] = max(performance(:,1,k));
        [~,shift2] = max(performance(:,2,k));
        [~,shift3] = max(performance(:,3,k));
        [~,shift4] = max(performance(:,4,k));
        [~,shift5] = max(performance(:,5,k));
        [~,shift6] = max(performance(:,6,k));
        [~,shift7] = max(performance(:,7,k));
        [~,shift8] = max(performance(:,8,k));
        [~,shift9] = max(performance(:,9,k));
        [~,shift10] = max(performance(:,10,k));
        [~,shift11] = max(performance(:,11,k));
        full_shifts(k,1) = shift1;
        full_shifts(k,2) = shift2;
        full_shifts(k,3) = shift3;
        full_shifts(k,4) = shift4;
        full_shifts(k,5) = shift5;
        full_shifts(k,6) = shift6;
        full_shifts(k,7) = shift7;
        full_shifts(k,8) = shift8;
        full_shifts(k,9) = shift9;
        full_shifts(k,10) = shift10;
        full_shifts(k,11) = shift11;
        
    end

    best_shifts = nanmedian(full_shifts);
    best_shifts = round(best_shifts);

    clear performance_all



    for k = 1:numel(files) % select neuron of choice
    
        filenameX = files(k).name(25:end);

        ixix = find(strcmp(all_file_names,filenameX));

        if isempty(ixix)
            unitary_scaling = NaN;
        else
            unitary_scaling = all_unitary_amplitudes(ixix);
        end


        clear XX YY
        disp([kkk,k])
        load(files(k).name)
        load(filesO(k).name)
        
        spike_rates_MLSpike = conv(spike_rates_MLSpike,fspecial('gaussian',[20 1],1.5152),'same');
        if kkk > 10
            spike_rates_MLSpike_Tuned = conv(spike_rates_MLSpike_Tuned,fspecial('gaussian',[20 1],1.5152),'same');
        end
        spike_rates_OASIS = conv(spike_rates_OASIS,fspecial('gaussian',[20 1],1.5152),'same');
        spike_rates_OASIS_finetuned = conv(spike_rates_OASIS_finetuned,fspecial('gaussian',[20 1],1.5152),'same');
        
        
        % compute correlation with ground truth (optimized for temporal shift)
        
        clear performance

        good_ixs = find(~isnan(spike_rates_GC8));

        XX(1,:) = circshift(spike_rates_GC8(good_ixs),-all_shifts(best_shifts(2)));
        YY(1,:) = ground_truth(good_ixs)';
        XX(2,:) = circshift(spike_rates_GLOBAL(good_ixs),-all_shifts(best_shifts(3)));
        YY(2,:) = ground_truth(good_ixs)';
        XX(3,:) = circshift(spike_rates_OASIS(good_ixs),-all_shifts(best_shifts(4)));
        YY(3,:) = ground_truth(good_ixs)';
        XX(4,:) = circshift(spike_rates_MLSpike(good_ixs),-all_shifts(best_shifts(5)));
        YY(4,:) = ground_truth(good_ixs)';
        XX(5,:) = circshift(spike_rates_transfer(good_ixs),-all_shifts(best_shifts(8)));
        YY(5,:) = ground_truth(good_ixs)';
        XX(6,:) = circshift(spike_rates_within(good_ixs),-all_shifts(best_shifts(9)));
        YY(6,:) = ground_truth(good_ixs)';
        XX(7,:) = circshift(calcium(good_ixs),-all_shifts(best_shifts(1)));
        YY(7,:) = ground_truth(good_ixs)';
        

        smoothed_data = conv(ground_truth(good_ixs),fspecial('gaussian',[60 1],30),'same'); 
        
%         figure(4545); plot(smoothed_data); hold on;


        select_ixs = 1:numel(smoothed_data);
%         select_ixs = 1:round(numel(smoothed_data)/2);
%         select_ixs = round(numel(smoothed_data)/2):numel(smoothed_data);
        smoothed_data = smoothed_data(select_ixs);
        YY = YY(:,select_ixs);
        XX = XX(:,select_ixs);
        XX(5,:) = XX(6,:);
        XX(6,:) = XX(6,:)/unitary_scaling;

        [ix,xi] = sort(smoothed_data,'ascend');
        
        bins = 0.02:0.02:1.0;
        for jk = 1:7
            figure(1312+kkk), 
            title(folders(kkk).name)
            clear avg_spikes
            for jj = 1:numel(bins)
                
                X = conv(XX(jk,:),fspecial('gaussian',[60 1],30),'same'); 
%                 if jk == 1 && jj == 1
%                     if any(X>0.5)
%                         disp(k)
%                         keyboard
%                     end
%                 end

                
    
                indices = find((smoothed_data<bins(jj)).*(smoothed_data>(bins(jj)-0.02)));
    
                avg_spikes(jj,2) = nanmean(X(indices));
                avg_spikes(jj,1) = nanmean(smoothed_data(indices));
            end

            y0 = avg_spikes(:,2);
            x0 = avg_spikes(:,1);
            
            if sum(~isnan(y0))>0

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
    
                deviation_from_linearity(jk,k,kkk) = sqrt( nanmean( (y_predict-yData).^2./y_predict.^2) );
                
    
    %             deviation_from_identity(jk,k,kkk) = nanmean(yData - xData)/nanmean(xData);
                
                deviation_from_identity(jk,k,kkk) = slope;
                
                
                subplot(2,4,jk);
                if jk < 7
                    plot(avg_spikes(:,1)*30,avg_spikes(:,2)*30,'-'); hold on;
                    if k == 1
                        plot([0 0.8]*30,[0 0.8]*30,'--')
                    end
                else
                    plot(avg_spikes(:,1)*30,avg_spikes(:,2),'-'); hold on;
                end
                set(gca,'TickDir','out');
                box off
    
                xlim([0 0.5]*30)
                xlabel('True spike rate (Hz)')
                ylabel('Predicted spike rate (Hz)')
    %             ylim([0 0.55])
            end

        end
    end


    deviation_from_linearity(deviation_from_linearity == 0) = NaN;
    deviation_from_identity(deviation_from_identity == 0) = NaN;

end



nanmean(squeeze(nanmedian(abs(deviation_from_identity([7 3 4 2 1 6],:,1:3)), 2)),2)
nanmean(squeeze(nanmedian(abs(deviation_from_identity([7 3 4 2 1 6],:,4:9)), 2)),2)
squeeze(nanmedian(abs(deviation_from_identity([7 3 4 2 1  6],:,12:14)), 2))

nanmean(squeeze(nanmedian(abs(deviation_from_linearity([7 3 4 2 1 6],:,1:3)), 2)),2)
nanmean(squeeze(nanmedian(abs(deviation_from_linearity([7 3 4 2 1 6],:,4:9)), 2)),2)
squeeze(nanmedian(abs(deviation_from_linearity([7 3 4 2 1  6],:,12:14)), 2))


%

datasets_selector = {[1:9],12,13,14};
% reordering = [7 3 4 2 1  6];
dataset_ID = {'GCaMP6f','GCaMP8f','GCaMP8m','GCaMP8s'};
% model_ID = {'dF/F','OASIS','MLSpike','Default CASCADE','GC8-tuned CASCADE','Finetuned CASCADE'};
reordering = [7 3 5 2 1  6];
dataset_ID = {'GCaMP6f','GCaMP8f','GCaMP8m','GCaMP8s'};
model_ID = {'dF/F','OASIS','Finetuned CASCADE','Default CASCADE','GC8-tuned CASCADE','Finetuned CASCADE rescaled'};
clear linearity_results

for k = 1:numel(datasets_selector)
    
    datasets = datasets_selector{k};
    temp = [];
    for j = 1:numel(datasets) 
        temp = [temp,deviation_from_linearity(:,:,datasets(j)) ];
    end
    linearity_results{k} = temp;
end

for k = 1:numel(linearity_results)
    figure(414); subplot(1,4,k); boxplot(linearity_results{k}(reordering,:)')
    box off
%     set(gca, 'YScale', 'log')
    ylim([0 2])
    set(gca,'TickDir','out');
    xtickangle(45)
    title(dataset_ID{k});
    xticklabels(model_ID)
    if k == 1
        ylabel('Deviation from linearity')
    end
end
set(gcf,'Position',[181.3333  344.0000  827.6667  243.3334])

for k = 1:4
    signrank(linearity_results{k}(reordering(1),:),linearity_results{k}(reordering(5),:))
end


clear linearity_results2

for k = 1:numel(datasets_selector)
    
    datasets = datasets_selector{k};
    temp = [];
    for j = 1:numel(datasets) 
        temp = [temp,deviation_from_identity(:,:,datasets(j)) ];
    end
    linearity_results2{k} = abs(temp);
end

for k = 1:numel(linearity_results2)
    figure(4114); subplot(1,4,k); boxplot(linearity_results2{k}(reordering(3:end),:)'-1)
    box off
%     set(gca, 'YScale', 'log')
    ylim([-2 2])
    yline(0,'--')
    set(gca,'TickDir','out');
    xtickangle(45)
    title(dataset_ID{k});
    xticklabels(model_ID(3:end))
    if k == 1
        ylabel('Deviation from linearity')
    end

end
set(gcf,'Position',[121.3333  183.6667  634.0000  240.6667])


index = 6 % 6 and 5
for k = 1:4
    N = nansum(~isnan(linearity_results2{k}(reordering(index),:)));
    [nanmedian(linearity_results2{k}(reordering(index),:))-1, nanstd(linearity_results2{k}(reordering(index),:))/sqrt(N)]
end