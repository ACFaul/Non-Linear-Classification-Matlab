load fisheriris
% Extract two attributes.
pl = meas(:,3); % petal length
pw = meas(:,4); % petal width
X = [pl,pw];

% Create classifier.
k = 5;
kNNModel = fitcknn(X,species,...
    'NumNeighbors',k,...        % number of neighbours
    'Standardize',true);        % standardize data

% Lay grid over the region.
d = 0.01;
[x1Grid,x2Grid] = meshgrid(0.8:d:7,0:d:3);
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);

% For each grid point calculate the score of each class.
% 'predict' returns the predicted class labels corresponding to the 
% minimum misclassification cost, the score (posterior probability) 
% for each class as well as the expected classification cost for
% each class
[~,score,~] = predict(kNNModel,xGrid);

% Classify according to the maximum score.
[~,maxScore] = max(score,[],2);

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

% Plot new points with ellipses containing nearest neighbours.
newpoints = [2.5 .75;...
    5 1.4;...
    6 2];
plot(newpoints(:,1),newpoints(:,2),'xk','linewidth',1.5);
% Find nearest neighbours.
[idx,d] = knnsearch(X,newpoints,...
    'k',k,...                % number of neighbours
    'Distance','seuclidean');% Euclidean distance on standardized data
% Mark neighbours.
plot(X(idx,1),X(idx,2),'ok','markersize',10);
% Plot ellipses.
for i=1:3
    s = kNNModel.Sigma *d(i,end); % scale standardized coordinates
    c = newpoints(i,:) - s;   % corner of rectangle containing ellipse
	% Draw an ellipse around the nearest neighbours.
    h = rectangle('position',[c,2*s(1),2*s(2)],...
        'curvature',[1 1],'Linestyle','--');
end
hold off