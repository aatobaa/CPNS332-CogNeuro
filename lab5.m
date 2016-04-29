%%  EXERCISES 1 & 2 Perceptrons

set1 = [2,2; -2, 2; 0.5, 1.5];
set2 = [1, -2; -1, 1; -0.5, -0.5];
setAll = [set1; set2];
% 
% figure
% plot(set1(:,1),set1(:,2),'*')
% hold on
% plot(set2(:,1),set2(:,2),'*','color','r')
% axis([-2.5 2.5 -2.5 2.5])
% axis square

W = [0 0];
b = 0;
targets = [1 1 1 0 0 0];
lr = 0.01;
numEpochs = 100;
numInputs = 6;

allErrors = [];
for i = 1:numEpochs
    totalError = 0;
    for j = 1:numInputs
        inp = setAll(j,:);
        net_input = W*inp' + b;
        out = net_input;
        
        W = W+lr*(targets(j)-out)*inp;
        b = b+lr*(targets(j)-out);
        
        totalError = totalError + (targets(j)-out)^2;
    end
    allErrors = [allErrors; totalError];
end

plot(allErrors)

%% EXERCISE 3 Greeble Classification with Perceptron

badGreeblesTrain = xlsread('BadGreeblesTraining.xls');
goodGreeblesTrain = xlsread('GoodGreeblesTraining.xls');
trainData = [goodGreeblesTrain;badGreeblesTrain];

% 1 rerepsents a good Greeble, 0 represents a bad Greeble
targets = [repmat(1,200,1)' repmat(0,200,1)'];

W = [1 1 1];
b = 0;
num_inputs = size(trainData,1);

initialClassification = W*trainData' + b;

for i = 1:numEpochs
    for j = 1:numInputs
        input = trainData(j,:);
        net_input = W*input' + b;
        out = net_input;
        
        W = W + lr*(targets(j) - out)*input;
        b = b + lr*(targets(j) - out);
    end
end

trainedClassification = W*trainData' + b;

plot(trainedClassification,'.','MarkerSize',16)


%% EXERCISE 4: Backpropagation
NHIDDEN = 1;
NINP = 3;
NOUT = 2;

Wh = rand(NHIDDEN,NINP); %weight matrix feeding hidden nodes
Wo = rand(NOUT, NHIDDEN); %weight matrix feeding output nodes
bh = zeros(NHIDDEN,1); %bias weights feeding hidden nodes
bo = zeros(NOUT,1); %bias weights feeding output nodes
transfer_fn=@(x,alpha) 1./(1+exp(alpha*x));

% Learning rule applied to weights feeding the output nodes (from hidden
% layer to output layer) will be the non-linear Widrow-Hoff learning rule.
% From inputs to hidden layer will be __________...

%TODO: Classify the Greebles using the above network and a sigmoid transfer
%function & backpropagation learning rule. 



