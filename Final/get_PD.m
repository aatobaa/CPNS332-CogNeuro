function [ PD, fitted_cos ] = get_PD( data )
%Gets preferred direction of each neuron

if ndims(data) == 3
    numNeurons = size(data,2);
    numTrials = size(data,3);   
else
    numNeurons = size(data,1);
    numTrials = size(data,2);   
end

meanFR = get_meanFR(data);

PD = zeros(numNeurons,1);
fitted_cos = cell(numNeurons,1);

for i = 1:numNeurons
    %24 has no data
    if i == 24
        continue
    end
    x = deg2rad([0 45 90 135 180 225 270 315]);
    y = meanFR(i,:);
    
    % fit a cosine tuning curve for the mean firing rate as a function of the target angle
    cos_fun = @(p, theta) p(1) + p(2) * cos ( theta - p(3) );
    p = nlinfit(x, y, cos_fun, [1 1 0] );
  
    phase = p(3);
    
    %Ensure phase is between 0 and 2pi.
    while phase < 0
        phase = phase + 2*pi;
    end
    while phase > 2*pi
        phase = phase - 2*pi;
    end
    
    %If the phase is picking out the minimum, correct it by pi
    if cos_fun(p,phase) < cos_fun(p,phase+0.1)
        phase = phase - pi;
        if phase < 0
            phase = phase + 2*pi;
        end
    end
    PD(i) = phase;
    fitted_cos{i} = p;
end

end

