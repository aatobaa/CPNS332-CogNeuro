%% PROJECT

SAMPLING_RATE = 22050;

kira1 = wavread('Kira_Training1.wav');
kira2 = wavread('Kira_Training2.wav');
kira3 = wavread('Kira_Training3.wav');
pascal1 = wavread('Pascal_Training1.wav');
pascal2 = wavread('Pascal_Training2.wav');
pascal3 = wavread('Pascal_Training3.wav');
%%

kira1_c = extract_features(kira1);
kira2_c = extract_features(kira2);
kira3_c = extract_features(kira3);
pascal1_c = extract_features(pascal1);
pascal2_c = extract_features(pascal2);
pascal3_c = extract_features(pascal3);

female_train = [kira1_c kira2_c kira3_c];
male_train = [pascal1_c pascal2_c pascal3_c];

trainingData = [female_train male_train];

k1s = size(kira1_c,2);
k2s = size(kira2_c,2);
k3s = size(kira3_c,2);
p1s = size(pascal1_c,2);
p2s = size(pascal2_c,2);
p3s = size(pascal3_c,2);

% 1 represents female, 0 represents male
target = [repmat(1,k1s+k2s+k3s,1)' repmat(0,p1s+p2s+p3s,1)'];

minVoice = min(trainingData');
maxVoice = max(trainingData');

normFemale = zeros(size(female_train));
normMale = zeros(size(male_train));

NUM_FEATURES = size(kira1_c,1); 

a=-1; b=1;
% Normalizing using formula from http://www.mathworks.com/matlabcentral/fileexchange/5103-toolbox-diffc/content/toolbox_diffc/toolbox/rescale.m
for i = 1:NUM_FEATURES
    normFemale(i,:) = (b-a) .* (female_train(i,:) - minVoice(i))./(maxVoice(i)-minVoice(i)) + a;
    normMale(i,:) = (b-a) .* (male_train(i,:) - minVoice(i))./(maxVoice(i)-minVoice(i)) + a; 
end

%%
trainingData = [normFemale normMale];

%%

NHIDDEN = 10;
NINP = NUM_FEATURES;
NOUT = 1;

Wh = rand(NHIDDEN,NINP); %weight matrix feeding hidden nodes
Wo = rand(NOUT, NHIDDEN); %weight matrix feeding output nodes
bh = zeros(NHIDDEN,1); %bias weights feeding hidden nodes
bo = zeros(NOUT,1); %bias weights feeding output nodes
lr = 0.1;
transfer_fn=@(x,alpha) 1./(1+exp(alpha*x));

inp_input = trainingData;

%Derivations below assume sigmoid transfer fxn with alpha of -1;
numEpochs = 1000;
allErrors = [];

numTrainingInput = size(inp_input,2);

for j = 1:numEpochs
    epochError = 0;
    for i = 1:numTrainingInput
        net_input_h = Wh*inp_input(:,i) + bh;
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
        Wh = Wh + lr*transfer_fn(net_input_h,-1).*(1-transfer_fn(net_input_h,-1))*sum((Wo*errors_o))*inp_input(:,i)';
        bh = bh + lr*transfer_fn(net_input_h,-1).*(1-transfer_fn(net_input_h,-1))*sum((Wo*errors_o));
        
    end
    for k = 1:numTrainingInput
        net_input_h = Wh*inp_input(:,k) + bh;
        output_h = transfer_fn(net_input_h,-1);
        inp_hidden = output_h; 
        net_input_o = Wo*inp_hidden + bo; 
        output_o = transfer_fn(net_input_o,-1);
        epochError = epochError + (target(k)-output_o)^2;
    end
    epochError
    allErrors = [allErrors; epochError];
end
%%
net_input_h = Wh*trainingData + repmat(bh,1,numTrainingInput);
output_h = transfer_fn(net_input_h,-1);
inp_hidden = output_h; 
net_input_o = Wo*inp_hidden + bo; 
bpGreeblesClassification = transfer_fn(net_input_o,-1);
plot(bpGreeblesClassification,'.')



