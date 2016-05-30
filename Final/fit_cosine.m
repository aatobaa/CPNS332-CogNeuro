function [ output_args ] = fit_cosine( meanFR )
%% Fit a cosine tuning curve for mean firing rate 
% Keep track of errors of each fit
allErrors = nan(143,1);
for i = 1:143
    %24 has no data
    if i == 24
        continue
    end
    
    cos_fun = @(p, theta) p(1) + p(2) * cos ( theta - p(3) );
    
    %x values is theta in radians
    x = deg2rad([0 45 90 135 180 225 270 315]);

    %y is observed mean firing rates
    y = meanFRAllNeurons(i,:);
    
    %Fit a cosine function to the mean FR as a function of angles
    p = nlinfit(x, y, cos_fun, [1 1 0] );
%     plot(rad2deg(x), y)
%     hold on    
%     xScale = 0:0.1:2*pi;
%     yScale = cos_fun(p,xScale);
%     plot(rad2deg(xScale),yScale,'r.','MarkerSize',16)
%     pause
%     hold off
    % Compute sum squared error
    yFit = cos_fun(p,x);
    error = y - yFit;
    standardizedError = sum((error.*error)/var(y));
    allErrors(i) = standardizedError;
end




end

