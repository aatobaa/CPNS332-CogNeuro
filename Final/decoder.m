% Parameters to adjust:
% Restricting to only looking at neurons that are well-fit to cosine, error
%    < 2
% Compare PMd vs MI neurons 

clear all

load('Lab5_CenterOutTrain')

%Begin by creating a data structure to organize the data. Every element
%contains the spike times corresponding to up to 1 second after instruction
%cue
trainingData = cell(8,143,158);
for neuronNum = 1:143
    allSpikeTimes = unit(neuronNum).times;
    for trialNum = 1:158
        dir = direction(trialNum);
        instructionTime = instruction(trialNum);
        segmentedSpikeTimes = allSpikeTimes(allSpikeTimes > (instructionTime) & allSpikeTimes < (instructionTime + 1));
        centeredSpikes = segmentedSpikeTimes - instructionTime;
        trainingData{dir,neuronNum,trialNum} = centeredSpikes;
    end
end

%% Compute Mean Firing Rates
meanFRAllNeurons = zeros(143,8);
for directionIdx = 1:8
    for neuronNum = 1:143
        firingRates = [];
        for trialNum = 1:158
            if direction(trialNum) ~= directionIdx
                continue
            end
            spikes = trainingData{directionIdx,neuronNum,trialNum};
            %Count the number of spikes in the 1-second window
            spikeCounts = length(spikes); 
            firingRates = [firingRates; spikeCounts];
        end
        meanFRAllNeurons(neuronNum,directionIdx) = mean(firingRates);
    end
end
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



%% Evaluate errors

figure
hist(allErrors)
xlabel('Error')
ylabel('Frequency')
title('Histogram of errors of cosine fit')

%% Check how error and fit correspond
figure
goodNeurons = find(allErrors < 2);
for i = 1:length(goodNeurons)
    y = meanFRAllNeurons(goodNeurons(i),:);
    allErrors(goodNeurons(i))
    plot(rad2deg(x), y)
    hold on
    p = nlinfit(x, y, cos_fun, [1 1 0] );
    xScale = 0:0.1:2*pi;
    yScale = cos_fun(p,xScale);
    plot(rad2deg(xScale),yScale,'r.','MarkerSize',16)
    legend('Actual','Fit')
    xlabel('Target angle (degrees)')
    ylabel('Mean Firing Rate (spikes/second)')
    title(strcat('Tuning curve for Neuron ',int2str(goodNeurons(i))))
    xlim([0 315])
    hold off
    pause
end

%% Pick the neurons that are fitted well
goodNeurons = find(allErrors < 2);
badNeurons = find(allErrors > 2);


%% Test the model
clearvars -except trainingData meanFRAllNeurons preferredDirections goodNeurons allErrors
load('Lab6_CenterOutTest')

numTrials = length(go);
testData = cell(143,numTrials);
for neuronNum = 1:143
    allSpikeTimes = unit(neuronNum).times;
    for trialNum = 1:numTrials
        instructionTime= instruction(trialNum);
        segmentedSpikeTimes = allSpikeTimes(allSpikeTimes > (instructionTime) & allSpikeTimes < (instructionTime + 1));
        centeredSpikes = segmentedSpikeTimes - instructionTime;
        testData{neuronNum,trialNum} = centeredSpikes;
    end
end

%% 
%Compute weights - EXPERIMENT WITH DIFFERENT WEIGHTS
    % Weight by distribution of preferred directions
holdPeriodFR = zeros(143,numTrials);
PDProportions = [];
for i = 1:8
    PDProportions(i) = sum(rad2direction(preferredDirections)==i)/length(preferredDirections);
end
for neuronNum = 1:143
    weights = [];
        for trialNum = 1:numTrials
            spikes = testData{neuronNum,trialNum};
            spikeCounts = length(spikes);
            fr = spikeCounts;
            if sum(neuronNum==goodNeurons) == 0
                fr = 0;
            end
            %Scale the firing rate by the proportion of neurons
            PD = rad2direction(preferredDirections(neuronNum));
            weight = fr / PDProportions(PD);
            weights = [weights; weight];
        end
    holdPeriodFR(neuronNum,:) = weights;
end

% Testing
[pct_correct predictedDirections] = decode_PV(testData, holdPeriodFR, goodNeurons,preferredDirections,direction);
pct_correct

%% Try to pick the best subset of neurons to use.

%% First: Show that error threshold does not predict performance
performance = [];
for i = 0:0.1:10
    goodNeurons = find(allErrors < i);
    [pct_correct ~]= decode_PV(testData,holdPeriodFR,goodNeurons,preferredDirections,direction);
    performance = [performance; pct_correct];
end
plot(0:0.1:10,performance)
xlabel('Error Threshold')
ylabel('Percent Correct')
%%

performance = [];
goodNeurons = [];
current_performance = 0;
for i = 1:143
    testNeurons = [goodNeurons; i];
    pct_correct = decode_PV(testData,holdPeriodFR,testNeurons,preferredDirections,direction);
    if pct_correct > current_performance
        goodNeurons = [goodNeurons; i];
        current_performance = pct_correct;
        performance = [performance; pct_correct];
    end
end
%%    
plot(performance)
xlabel('Number of Neurons')
ylabel('Percent Correct')


%% RANDOMIZE TRIALS TESTING SUITE
[newTrainData,newTestData] = randomize_trials(trainingData,testData,direction);

numNeurons = size(newTrainData,2)
numTrainTrials = size(newTrainData,3)
numTestTrials = size(newTestData,3)
preferredDirections = get_PD(newTrainData);

%%
goodNeurons = 1:143;
%% 
%Compute weights, weight by distribution of preferred directions
holdPeriodFR = zeros(numNeurons,numTestTrials);
PDProportions = [];
for i = 1:8
    PDProportions(i) = sum(rad2direction(preferredDirections)==i)/length(preferredDirections);
end
for neuronNum = 1:numNeurons
    weights = [];
        for trialNum = 1:numTestTrials
            %This should be testData, not trainData. Recall, we are
            %computing the weights as a function of a neuron's PD
            %(determined from training) AND what the neuron's activity
            %currently is (in the testing session). Note, spike counts do
            %NOT utilize any direction information. 
            spikes = [newTestData{:,neuronNum,trialNum}];
            spikeCounts = length(spikes);
            fr = spikeCounts;
            if sum(neuronNum==goodNeurons) == 0
                fr = 0;
            end
            dir = direction(trialNum);
            %Scale the firing rate by the proportion of neurons
            if dir == rad2direction(preferredDirections(neuronNum))
                weight = fr / PDProportions(dir);
            else
                weight = fr;
            end
            weights = [weights; weight];
        end
    holdPeriodFR(neuronNum,:) = weights;
end

% Testing
%%
directions = zeros(numTestTrials,1);
for i = 1:numTestTrials
    %Figure out which direction the trial was in
    %Get an 8x1 cell of directions, all are empty except 1
    directionCell = newTestData(:,1,i);
    for d = 1:8
        if sum(size(directionCell{d}))~=0
            directions(i) = d;
        end
    end
end    
%%
[pct_correct,predictedDirections] = decode_PV(newTestData, holdPeriodFR, goodNeurons,preferredDirections,directions);
pct_correct





