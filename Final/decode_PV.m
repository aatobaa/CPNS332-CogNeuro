function [ pct_correct predictedDirections ] = decode_PV( data,weights, neurons,PD,CD )
if ndims(data) == 3
    numTrials = size(data,3);
else
    numTrials = size(data,2);
end

testData = data;
goodNeurons = neurons;
preferredDirections = PD;
direction = CD;

%% POPULATION VECTOR METHOD 
% Compute Population Vector 
PopulationVectorXY = zeros(numTrials,2);
for trialNum = 1:numTrials
    Px = nansum(weights(neurons,trialNum).*cos(preferredDirections(neurons)));
    Py = nansum(weights(neurons,trialNum).*sin(preferredDirections(neurons)));
    PopulationVectorXY(trialNum,:) = [Px Py];
end
% Convert components to one vector
PopulationVectorRad = atan2(PopulationVectorXY(:,2),PopulationVectorXY(:,1));
% In accordance with lab hint, add 2*pi to negative radian values. 
PopulationVectorRad(PopulationVectorRad < 0) = PopulationVectorRad(PopulationVectorRad < 0) + 2*pi;
% Convert to target direction
x = 0:pi/4:2*pi;
for i = 1:numTrials
    d = abs(PopulationVectorRad(i)-x);
    [m predictedTargetPV(i)] = min(d);
end
% Convert direction 9 to direction 1
predictedTargetPV(predictedTargetPV == 9) = 1;

%Compute percent correct
pct_correct = sum(predictedTargetPV' == direction) / numTrials * 100;
predictedDirections = predictedTargetPV;
end

