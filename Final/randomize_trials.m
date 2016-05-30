function [ randomizedTrainData randomizedTestData ] = randomize_trials( trainingData, testData, directions)
%Feed all data we have to generate a random set of "trial" vs "test"

% First, combine all the trials
allTrials = cell(8,143,158+80);
for dir = 1:8
    for neuronNum = 1:143
        for trialNum = 1:158
            allTrials{dir,neuronNum,trialNum} = trainingData{dir,neuronNum,trialNum};
        end
    end
end
for neuronNum = 1:143
    for trialNum = 159:158+80
        dir = directions(trialNum-158);
        allTrials{dir,neuronNum,trialNum} = testData{neuronNum,trialNum-158};
    end
end

% Now, randomly assign trials to train or test
allTrialsIdx = 1:158+80;
trainTrialIdx = [rand(158+80,1)<0.7];
testTrialIdx = ~trainTrialIdx;
trainTrials = allTrialsIdx(trainTrialIdx);
testTrials = allTrialsIdx(testTrialIdx);
        
randomizedTrainData = allTrials(:,:,trainTrials);
randomizedTestData = allTrials(:,:,testTrials);

end

