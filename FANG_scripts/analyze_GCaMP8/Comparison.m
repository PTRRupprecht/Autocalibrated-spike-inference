%% Compare amplitudes between two methods

%modeling_amp = per_recording_amplitudes;
%deconv_amp = neuron_deconv_amp';
plot_amplitude_comparison(modeling_amp, deconv_amp, 3);


function plot_amplitude_comparison(modeling_amplitudes, deconv_amplitudes, threshold)

    % if threshold is not provided, set default to 5
    if nargin < 3
        threshold = 5;
    end
    
    % filter extreme values
    valid_idx = modeling_amplitudes <= threshold & deconv_amplitudes <= threshold;
    
    % get filtered data
    modeling_filtered = modeling_amplitudes(valid_idx);
    deconv_filtered = deconv_amplitudes(valid_idx);
    
    % calculate number of filtered points
    total_points = length(modeling_amplitudes);
    filtered_points = total_points - sum(valid_idx);
    percent_filtered = (filtered_points/total_points) * 100;
    
    % create figure with specified size
    figure('Position', [100, 100, 1000, 800]);
    
    % create scatter plots with different colors
    scatter(modeling_filtered, deconv_filtered, 80, 'b', 'filled', ...
        'MarkerFaceAlpha', 0.5, 'DisplayName', 'Data points');
    hold on;
    
    % calculate linear regression for filtered data
    p = polyfit(modeling_filtered, deconv_filtered, 1);
    y_fit = polyval(p, modeling_filtered);
    
    % calculate R-squared
    yresid = deconv_filtered - y_fit;
    SSresid = sum(yresid.^2);
    SStotal = (length(deconv_filtered)-1) * var(deconv_filtered);
    rsq = 1 - SSresid/SStotal;
    
    % plot linear regression line
    plot(modeling_filtered, y_fit, 'g-', 'LineWidth', 2, ...
        'DisplayName', sprintf('Linear fit (slope=%.2f, R^2=%.3f)', p(1), rsq));
    
    % add labels and title
    xlabel('Modeling Method Amplitudes', 'FontSize', 16);
    ylabel('Deconvolution Method Amplitudes', 'FontSize', 16);
    title({'Comparison of XXX Amplitudes: Modeling vs Deconvolution', ...
           sprintf('(Filtered: amplitude ≤ %.1f)', threshold)}, ...
           'FontSize', 18);
    
    % add legend in upper left corner
    legend('Location', 'northwest', 'FontSize', 14);
    
    % add text box with filtering statistics in lower right corner
    stats_str = sprintf('Total points: %d\nFiltered points: %d (%.1f%%)', ...
                       total_points, filtered_points, percent_filtered);

    text_box = annotation('textbox', [0.76, 0.1, 0.3, 0.1], ...
               'String', stats_str, ...
               'EdgeColor', 'none', ...
               'FontSize', 12);
    
    
    % make plot looks nice
    grid on;
    set(gca, 'FontSize', 14);
    box on;
    
    % set axis limits to show range up to threshold
    xlim([0, threshold]);
    ylim([0, threshold]);
    
    % add diagonal line for reference (y=x) using gray color
    plot([0, threshold], [0, threshold], '--', 'Color', [0.5 0.5 0.5], ...
        'LineWidth', 1.5, 'DisplayName', 'y = x (perfect match)');
    
    hold off;
end

