%function [ predictedDir, actualDir, pct_correct] = iterate_decoder( )
%This is simply the decoder_2 file, put into a function to return its
%results to check for systematic biases

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

%% RANDOMIZE TRIALS TESTING SUITE
[newTrainData,newTestData] = randomize_trials(trainingData,testData,direction);

numNeurons = size(newTrainData,2);
numTrainTrials = size(newTrainData,3);
numTestTrials = size(newTestData,3);
[preferredDirections fitted_cosines] = get_PD(newTrainData);

%%
goodNeurons = 1:143;
goodNeurons = [     1
     2
     3
     4
     5
     6
     7
     9
    10
    11
    12
    16
    17
    18
    19
    21
    22
    23
    25
    26
    30
    32
    33
    34
    35
    37
    38
    39
    40
    41
    44
    47
    48
    49
    50
    51
    53
    55
    62
    63
    64
    65
    70
    71
    72
    73
    74
    75
    76
    77
    78
    79
    81
    82
    86
    87
    89
    90
    91
    92
    94
    96
    97
    98
   102
   103
   104
   105
   108
   109
   111
   112
   113
   114
   115
   116
   118
   119
   120
   121
   122
   124
   125
   127
   128
   129
   130
   131
   132
   133
   134
   135
   136
   138
   139
   140
   141
   143];
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
            %Scale the firing rate by the proportion of neurons
            PD = rad2direction(preferredDirections(neuronNum));
            weight = fr / PDProportions(PD);
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
predictedDir = predictedDirections;
actualDir = directions; 
pct_correct


%end

