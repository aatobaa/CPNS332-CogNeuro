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


%%

%% EXERCISE 4: Backpropagation
badGreeblesTrain = xlsread('BadGreeblesTraining.xls');
goodGreeblesTrain = xlsread('GoodGreeblesTraining.xls');
trainData = [goodGreeblesTrain;badGreeblesTrain];
%%
% 1 rerepsents a good Greeble, 0 represents a bad Greeble
target = [repmat(1,200,1)' repmat(0,200,1)'];

NHIDDEN = 2;
NINP = 3;
NOUT = 1;

Wh = rand(NHIDDEN,NINP); %weight matrix feeding hidden nodes
Wo = rand(NOUT, NHIDDEN); %weight matrix feeding output nodes
bh = zeros(NHIDDEN,1); %bias weights feeding hidden nodes
bo = zeros(NOUT,1); %bias weights feeding output nodes
lr = 0.01;
transfer_fn=@(x,alpha) 1./(1+exp(alpha*x));

inp_input = trainData;
%TODO: Determine targets for output layer (from training) 

%Derivations below assume sigmoid transfer fxn with alpha of -1;

numEpochs = 100;
allErrors = [];

numTrainingInput = size(inp_input,1);

massiveSub = 9;
massiveSubPlot = [];
for Z = 1:massiveSub
    for j = 1:numEpochs
        epochError = 0;
        for i = 1:numTrainingInput
            net_input_h = Wh*inp_input(i,:)' + bh;
            output_h = transfer_fn(net_input_h,-1);
            inp_hidden = output_h; 
            %The output activation of the hidden layer is the input to the output layer

            net_input_o = Wo*inp_hidden + bo; 
            output_o = transfer_fn(net_input_o,-1);
            Wo = Wo + lr*transfer_fn(net_input_o,-1).*(1-transfer_fn(net_input_o,-1))*(target(i)-output_o)*inp_hidden';
            % Can we just use:
            % Wo = Wo + lr*errors_o*inp_hidden;?

            bo = bo + lr*transfer_fn(net_input_o,-1).*(1-transfer_fn(net_input_o,-1))*(target(i)-output_o);

            %Should this be before or after net_input_o is calculated?
            % Compute the errors from the Output layer
            errors_o = transfer_fn(net_input_o,-1).*(1-transfer_fn(net_input_o,-1))*(target(i)-output_o);

            % Adjust the weights feeding into the Hidden layer from the errors at the Output layer. 
            output_h = transfer_fn(net_input_h,-1);
            Wh = Wh + lr*transfer_fn(net_input_h,-1).*(1-transfer_fn(net_input_h,-1))*sum((Wo*errors_o))*inp_input(i,:);
            bh = bh + lr*transfer_fn(net_input_h,-1).*(1-transfer_fn(net_input_h,-1))*sum((Wo*errors_o));
        end
        for k = 1:numTrainingInput
            net_input_h = Wh*inp_input(k,:)' + bh;
            output_h = transfer_fn(net_input_h,-1);
            inp_hidden = output_h; 
            net_input_o = Wo*inp_hidden + bo; 
            output_o = transfer_fn(net_input_o,-1);
            epochError = epochError + (target(k)-output_o)^2;
        end
        allErrors = [allErrors; epochError];
    end
    %%
    net_input_h = Wh*trainData' + repmat(bh,1,400);
    output_h = transfer_fn(net_input_h,-1);
    inp_hidden = output_h; 
    net_input_o = Wo*inp_hidden + bo; 
    bpGreeblesClassification = transfer_fn(net_input_o,-1);
    massiveSubPlot = [massiveSubPlot; bpGreeblesClassification];
end
%%
figure
subplot(3,3,1)
plot(massiveSubPlot(1,:),'.')
axis([0 400 0 1])
subplot(3,3,2)
plot(massiveSubPlot(2,:),'.')
axis([0 400 0 1])
subplot(3,3,3)
plot(massiveSubPlot(3,:),'.')
axis([0 400 0 1])
subplot(3,3,4)
plot(massiveSubPlot(4,:),'.')
axis([0 400 0 1])
subplot(3,3,5)
plot(massiveSubPlot(5,:),'.')
axis([0 400 0 1])
subplot(3,3,6)
plot(massiveSubPlot(6,:),'.')
axis([0 400 0 1])
subplot(3,3,7)
plot(massiveSubPlot(7,:),'.')
axis([0 400 0 1])
subplot(3,3,8)
plot(massiveSubPlot(8,:),'.')
axis([0 400 0 1])
subplot(3,3,9)
plot(massiveSubPlot(9,:),'.')
axis([0 400 0 1])

%%%%%%
%% PROJECT

SAMPLING_RATE = 22050;

kira1 = wavread('Kira_Training1.wav');
kira2 = wavread('Kira_Training2.wav');
kira3 = wavread('Kira_Training3.wav');
pascal1 = wavread('Pascal_Training1.wav');
pascal2 = wavread('Pascal_Training2.wav');
pascal3 = wavread('Pascal_Training3.wav');















