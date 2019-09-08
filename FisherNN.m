load fisheriris
% Extract two attributes.
pl = meas(:,3); % petal length
pw = meas(:,4); % petal width
% Prepare data.
X = [pl,pw];
X = X';
classes = unique(species);
t = [];
for j = 1:numel(classes)
   indx = strcmp(species,classes(j)); 
   t = [t indx];
end
t = t';

% Create classifier.
hiddenSizes = [10]; % row vector of one or more hidden layer sizes
netModel = patternnet(hiddenSizes);
% Use all samples for training.
%netModel.divideFcn = 'dividetrain';
% Use 70% of samples for training, 15% for validation, 15% for testing.
netModel.divideFcn = 'dividerand';
netModel.divideParam.trainRatio = 0.7;
netModel.divideParam.valRatio = 0.15;
netModel.divideParam.testRatio = 0.15;
% Set transfer function of the set of hidden neurons to the hard-limit
% transfer function. This is equivalent to a particular hidden neuron 
% returning whether its input lies to the left or right of the 
% line given by the weights.
netModel.layers{1}.transferFcn = 'hardlim';
% By default the transfer function of the last layer is set to the
% softmax function. Dpending on which sides of the lines produced in
% the first layer a sample lies, a probability score for each class
% is given.

netModel = train(netModel,X,t);

% Lay grid over the region.
d = 0.01;
[x1Grid,x2Grid] = meshgrid(0.8:d:7,0:d:3);
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);

% For each grid point calculate the score of each class.
score = netModel(xGrid');

% Classify according to the maximum score.
[~,maxScore] = max(score,[],1);

% Plot classifier regions.
figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.5 0.5 0.5; 0.7 0.7 0.7; 0.9 0.9 0.9]);
hold on

% Plot data.
h(4:6) = gscatter(pl, pw, species,'rgb','os^');
xlabel('Petal length');
ylabel('Petal width');
legend(h,{'Setosa region','Versicolor region','Virginica region',...
    'Setosa','Versicolor','Virginica'},...
    'Location','Northwest');
axis([0.8 7 0 3])