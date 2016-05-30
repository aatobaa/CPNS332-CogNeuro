function [ meanFR ] = get_meanFR( data )
if ndims(data) == 3
    numNeurons = size(data,2);
    numTrials = size(data,3);
else
    numNeurons = size(data,1);
    numTrials = size(data,2);
end

direction = zeros(numTrials,1);
for i = 1:numTrials
    %Figure out which direction the trial was in
    %Get an 8x1 cell of directions, all are empty except 1
    directionCell = data(:,1,i);
    for d = 1:8
        if sum(size(directionCell{d}))~=0
            direction(i) = d;
        end
    end
end    
%% Compute Mean Firing Rates
meanFR = zeros(numNeurons,8);
for directionIdx = 1:8
    for neuronNum = 1:numNeurons
        firingRates = [];
        for trialNum = 1:numTrials
            if direction(trialNum) ~= directionIdx
                continue
            end
            spikes = data{directionIdx,neuronNum,trialNum};
            %Count the number of spikes in the 1-second window
            spikeCounts = length(spikes); 
            firingRates = [firingRates; spikeCounts];
        end
%         neuronNum
%         mean(firingRates)
%         firingRates
        meanFR(neuronNum,directionIdx) = mean(firingRates);
    end
end

end

